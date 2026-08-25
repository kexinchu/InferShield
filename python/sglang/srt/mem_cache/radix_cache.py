from __future__ import annotations

from zmq import NULL

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The radix tree data structure for managing the KV cache.
"""

import heapq
import hashlib
import threading
from functools import partial
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Tuple

import torch

from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool, TokenToKVPoolAllocator
from sglang.srt.mem_cache.safekv_policy import (
    DurableLedger,
    NamespaceKey,
    PublicAuthorization,
    PublicRegistry,
    SafeKVMetrics,
    Visibility,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

from sglang.srt.managers.private_service.private_client import PrivateJudgeClient
from sglang.srt.mem_cache.tree_node import TreeNode


def _key_match_page_size1(key0: List, key1: List):
    """Exact prefix match. A prior fuzzy (≤10-mismatch) variant let a
    near-copy or shorter probe observe another tenant's KV and split it.
    """
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i


def _key_match_paged(key0: List, key1: List, page_size: int):
    min_len = min(len(key0), len(key1))

    i = 0
    while i < min_len:
        if key0[i : i + page_size] != key1[i : i + page_size]:
            break
        i += page_size

    return i


class RadixCache(BasePrefixCache):
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
        page_size: int,
        private_judge_client: PrivateJudgeClient,
        disable: bool = False,
        enable_kv_cache_events: bool = False,
        access_budget_B: int = 10,
        creator_threshold_K: int = 2,
        safekv_mode: str = "strict",
        operator_key: Optional[str] = None,
        policy_epoch: int = 1,
        model_id: Optional[str] = None,
        tokenizer_version: Optional[str] = None,
        ledger_path: Optional[str] = None,
        experiment_autoshare: bool = False,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = page_size
        self.disable = disable
        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue = []

        self.private_judge_client = private_judge_client

        # SafeKV safeguard parameters
        self.access_budget_B = access_budget_B
        self.creator_threshold_K = creator_threshold_K
        self.safekv_mode = safekv_mode
        self.experiment_autoshare = experiment_autoshare
        self.model_id = model_id or "unknown-model"
        self.tokenizer_version = tokenizer_version or "unknown-tokenizer"
        self.metrics = SafeKVMetrics()
        self.ledger = DurableLedger(path=ledger_path)
        self.registry = PublicRegistry(
            (operator_key or "safekv-development-key").encode("utf-8"),
            policy_epoch=policy_epoch,
        )
        self._namespace_lock = threading.RLock()
        self.private_roots: Dict[str, TreeNode] = {}
        self.public_roots: Dict[str, TreeNode] = {}
        self.public_tokens: Dict[str, Tuple[int, ...]] = {}

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = lambda key: key[0]
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=page_size)
            self.get_child_key_fn = lambda key: tuple(key[:page_size])
        self.reset()

    ##### Public API #####

    def reset(self):
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.lock_ref = 1
        self.root_node.visibility = "root"
        with self._namespace_lock:
            self.private_roots = {}
            self.public_roots = {}
            self.public_tokens = {}
            self.registry.reset()
            self.metrics.reset()
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self._record_all_cleared_event()

    def _new_namespace_root(self, namespace_key: NamespaceKey) -> TreeNode:
        root = TreeNode()
        root.key = []
        root.value = []
        root.lock_ref = 1
        root.parent = self.root_node
        root.namespace_key = namespace_key
        root.visibility = namespace_key.visibility.value
        return root

    def _private_root(self, user_id: str, create: bool) -> Optional[TreeNode]:
        with self._namespace_lock:
            root = self.private_roots.get(user_id)
            if root is None and create:
                root = self._new_namespace_root(
                    NamespaceKey.private(user_id, "namespace-root")
                )
                self.private_roots[user_id] = root
            return root

    def _parse_authorization(
        self, authorization: Optional[Mapping[str, object]]
    ) -> Optional[PublicAuthorization]:
        if not authorization:
            return None
        try:
            return PublicAuthorization.from_mapping(authorization)
        except (KeyError, TypeError, ValueError) as exc:
            self.metrics.record_event(
                "authorization_rejected", reason="malformed", detail=str(exc)
            )
            return None

    def _install_authorization(
        self, authorization: Optional[Mapping[str, object]], key: List[int]
    ) -> Tuple[Optional[PublicAuthorization], str]:
        parsed = self._parse_authorization(authorization)
        if parsed is None:
            return None, "none" if not authorization else "malformed"
        result = self.registry.install(
            parsed,
            key,
            model_id=self.model_id,
            tokenizer_version=self.tokenizer_version,
        )
        self.metrics.record_event(
            "authorization_verified",
            object_id=parsed.public_object_id,
            accepted=result.valid,
            reason=result.reason,
        )
        if not result.valid:
            return None, result.reason
        with self._namespace_lock:
            self.public_tokens[parsed.public_object_id] = tuple(key)
            if parsed.public_object_id not in self.public_roots:
                self.public_roots[parsed.public_object_id] = (
                    self._new_namespace_root(
                        NamespaceKey.public(
                            parsed.public_object_id,
                            parsed.prefix_fingerprint,
                        )
                    )
                )
        return parsed, result.reason

    def _matching_public_root(
        self, key: List[int]
    ) -> Tuple[Optional[TreeNode], Optional[str]]:
        key_tuple = tuple(key)
        with self._namespace_lock:
            matches = [
                (len(tokens), object_id)
                for object_id, tokens in self.public_tokens.items()
                if not self.registry.is_revoked(object_id)
                and len(tokens) <= len(key_tuple)
                and key_tuple[: len(tokens)] == tokens
            ]
            if not matches:
                return None, None
            _, object_id = max(matches)
            return self.public_roots[object_id], object_id

    def _node_fingerprint(self, key: Optional[List]) -> str:
        """Return the accounting-epoch key for a cached token prefix."""
        if not key:
            return "empty"
        material = "\0".join(
            (
                str(self.registry.policy_epoch),
                self.model_id,
                self.tokenizer_version,
                ",".join(str(k) for k in key),
            )
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    @staticmethod
    def _value_identity(value) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            raw = ",".join(str(item) for item in value.detach().cpu().tolist())
        else:
            raw = ",".join(str(item) for item in value)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def flush_budgets_to_ledger(self) -> None:
        """Flush the reserve-before-hit ledger.

        Every Balanced hit is persisted before its KV address is returned, so
        eviction no longer reconstructs charges from in-node counters.
        """
        self.ledger.flush()

    def promote_to_budgeted(self, user_id: str, fingerprint: str) -> int:
        """Promote matching private nodes to BUDGETED_SHARED, restoring ledger budget.

        Used by the async detection pipeline when it determines a node is
        non-sensitive (false-negative case) and Balanced Mode is active.

        Returns the number of nodes promoted.
        """
        if self.safekv_mode != "balanced":
            return 0
        root = self._private_root(user_id, create=False)
        if root is None:
            return 0
        promoted = 0
        stack = list(root.children.values())
        while stack:
            node = stack.pop()
            if (
                node.visibility == Visibility.PRIVATE.value
                and not node.permanently_private
                and node.namespace_key is not None
                and node.namespace_key.fingerprint == fingerprint
            ):
                with node.transition_lock:
                    if (
                        node.visibility == Visibility.PRIVATE.value
                        and not node.permanently_private
                    ):
                        # Unknown ledger state must fail closed.
                        node_fp = self._node_fingerprint(node.key)
                        if not self.ledger.is_operational:
                            node.visibility = Visibility.EXHAUSTED_PRIVATE.value
                            node.permanently_private = True
                        else:
                            charged = self.ledger.charged_hits(node_fp)
                            node.access_budget = charged
                            if charged >= self.access_budget_B:
                                node.visibility = Visibility.EXHAUSTED_PRIVATE.value
                                node.permanently_private = True
                            else:
                                node.visibility = Visibility.BUDGETED_SHARED.value
                        promoted += 1
            stack.extend(node.children.values())
        return promoted

    def safekv_snapshot(self) -> Dict[str, object]:
        """Return JSON-compatible policy state for invariant experiments."""

        variants = []
        roots = list(self.private_roots.values()) + list(
            self.public_roots.values()
        )
        for root in roots:
            stack = list(root.children.values())
            while stack:
                node = stack.pop()
                namespace = node.namespace_key or root.namespace_key
                variants.append(
                    {
                        "node_id": node.id,
                        "object_id": node.object_id,
                        "visibility": node.visibility,
                        "namespace_visibility": (
                            namespace.visibility.value if namespace else None
                        ),
                        "namespace_identity": (
                            namespace.identity if namespace else None
                        ),
                        "creator_id": node.creator_id,
                        "token_count": len(node.key),
                        "kv_address_id": self._value_identity(node.value),
                    }
                )
                stack.extend(node.children.values())
        metrics = self.metrics.snapshot()
        return {
            "mode": self.safekv_mode,
            "model_id": self.model_id,
            "tokenizer_version": self.tokenizer_version,
            "policy_epoch": self.registry.policy_epoch,
            "access_budget_B": self.access_budget_B,
            "counters": dict(metrics["counters"]),
            "ledger": dict(self.ledger.snapshot()),
            "events": [
                {
                    "name": event.name,
                    "timestamp": event.timestamp,
                    "attributes": dict(event.attributes),
                }
                for event in metrics["events"]
            ],
            "variants": variants,
            "private_principals": sorted(self.private_roots),
            "public_object_ids": sorted(self.public_roots),
        }

    def match_prefix(
        self,
        key: List[int],
        user_id: Optional[str] = None,
        authorization: Optional[Mapping[str, object]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, int]:
        """Find the matching prefix from the radix tree.
        Args:
            key: A list of token IDs to find a matching prefix.
        Returns:
            A tuple of a tensor of matching prefix token IDs and
            the last node that contains the prefix values. Note that
            this API can modify the internal state of the Radix tree.
            The last node create a new child if the prefix is shorter
            than the last node's value.
        """
        if self.disable or len(key) == 0:
            return (
                torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                self.root_node,
            )

        # "none" mode: vanilla SGLang global cache sharing, no namespace isolation.
        if self.safekv_mode == "none":
            if self.page_size != 1:
                page_aligned_len = len(key) // self.page_size * self.page_size
                key = key[:page_aligned_len]
            value, last_node = self._match_prefix_helper(self.root_node, key, user_id)
            if value:
                tensors = [
                    v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.int64, device=self.device)
                    for v in value
                ]
                return torch.cat(tensors), last_node
            return torch.empty((0,), dtype=torch.int64, device=self.device), self.root_node

        if not user_id:
            self.metrics.record_event("lookup_denied", reason="missing_principal")
            return (
                torch.empty((0,), dtype=torch.int64, device=self.device),
                self.root_node,
            )

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        installed_auth, _ = self._install_authorization(authorization, key)
        if installed_auth is not None:
            public_root = self.public_roots[installed_auth.public_object_id]
            public_object_id = installed_auth.public_object_id
        else:
            public_root, public_object_id = self._matching_public_root(key)

        if public_root is not None:
            value, last_node = self._match_prefix_helper(
                public_root, key, user_id
            )
            if value or installed_auth is not None:
                self.metrics.record_event(
                    "lookup",
                    requester=user_id,
                    served_namespace="verified_public",
                    public_object_id=public_object_id,
                    hit=bool(value),
                )
                if value:
                    return torch.cat(value), last_node
                return (
                    torch.empty(
                        (0,), dtype=torch.int64, device=self.device
                    ),
                    last_node,
                )

        private_root = self._private_root(user_id, create=False)
        if private_root is not None:
            value, last_node = self._match_prefix_helper(
                private_root, key, user_id
            )
            self.metrics.record_event(
                "lookup",
                requester=user_id,
                served_namespace="private",
                owner=user_id,
                hit=bool(value),
            )
            if value:
                tensors = [
                    v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.int64, device=self.device)
                    for v in value
                ]
                return torch.cat(tensors), last_node

        # ── Balanced Mode: search other principals' BUDGETED_SHARED nodes ──
        # Walk only shareable nodes. Unauthorized children stop the walk
        # without splitting or returning KV, so a Private tree cannot leak
        # through a shorter or near-matching probe.
        if self.safekv_mode == "balanced" and user_id:
            with self._namespace_lock:
                other_roots = [
                    (uid, root)
                    for uid, root in self.private_roots.items()
                    if uid != user_id
                ]
            for owner_id, other_root in other_roots:
                value, last_node = self._match_prefix_helper(
                    other_root, key, user_id
                )
                if value:
                    self.metrics.record_event(
                        "lookup",
                        requester=user_id,
                        served_namespace="budgeted_shared",
                        owner=owner_id,
                        hit=True,
                    )
                    tensors = [
                        v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.int64, device=self.device)
                        for v in value
                    ]
                    return torch.cat(tensors), last_node

        self._scrub_cross_principal_residual()
        return (
            torch.empty((0,), dtype=torch.int64, device=self.device),
            self.root_node,
        )

    def insert(
        self,
        key: List,
        value=None,
        prompt="",
        user_id: Optional[str] = None,
        authorization: Optional[Mapping[str, object]] = None,
    ):
        if self.disable:
            return 0

        # "none" mode: global cache insertion (vanilla SGLang).
        if self.safekv_mode == "none":
            if value is None:
                value = [x for x in key]
            return self._insert_helper(self.root_node, key, value, prompt, user_id=user_id)

        if not user_id:
            return 0

        if value is None:
            value = [x for x in key]
        parsed = self._parse_authorization(authorization)
        installed_auth = None
        if parsed is not None and parsed.prefix_token_length <= len(key):
            installed_auth, _ = self._install_authorization(
                authorization, key[: parsed.prefix_token_length]
            )
        if installed_auth is not None:
            root = self.public_roots[installed_auth.public_object_id]
            key = key[: installed_auth.prefix_token_length]
            value = value[: installed_auth.prefix_token_length]
            was_empty = not root.children
        else:
            root = self._private_root(user_id, create=True)
            was_empty = False
        inserted = self._insert_helper(
            root, key, value, prompt, user_id=user_id
        )
        if installed_auth is not None and was_empty and root.children:
            self.metrics.increment(
                "public_object_created",
                object_id=installed_auth.public_object_id,
            )
        return inserted

    def cache_finished_req(self, req: Req):
        """Cache request when it finishes."""
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(req.origin_input_ids) + len(req.output_ids) - 1
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            self.req_to_token_pool.free(req.req_pool_idx)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:-1]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        if self.page_size != 1:
            page_aligned_len = len(kv_indices) // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len].clone()
            self.token_to_kv_pool_allocator.free(kv_indices[page_aligned_len:])
        else:
            page_aligned_len = len(kv_indices)
            page_aligned_kv_indices = kv_indices.clone()

        # Radix Cache takes one ref in memory pool
        new_prefix_len = self.insert(
            token_ids[:page_aligned_len],
            page_aligned_kv_indices,
            req.origin_input_text,
            user_id=req.user_id,
            authorization=req.safekv_public_authorization,
        )
        self.token_to_kv_pool_allocator.free(
            kv_indices[len(req.prefix_indices) : new_prefix_len]
        )

        # Remove req slot release the cache lock
        self.req_to_token_pool.free(req.req_pool_idx)
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        token_ids = req.fill_ids
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        if self.page_size != 1:
            page_aligned_len = len(kv_indices) // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len].clone()
        else:
            page_aligned_len = len(kv_indices)
            page_aligned_kv_indices = kv_indices.clone()
        page_aligned_token_ids = token_ids[:page_aligned_len]

        # Radix Cache takes one ref in memory pool
        new_prefix_len = self.insert(
            page_aligned_token_ids,
            page_aligned_kv_indices,
            req.origin_input_text,
            user_id=req.user_id,
            authorization=req.safekv_public_authorization,
        )
        self.token_to_kv_pool_allocator.free(
            kv_indices[len(req.prefix_indices) : new_prefix_len]
        )

        # The prefix indices could be updated, reuse it
        new_indices, new_last_node = self.match_prefix(
            page_aligned_token_ids,
            user_id=req.user_id,
            authorization=req.safekv_public_authorization,
        )
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(len(req.prefix_indices), len(new_indices))),
            new_indices[len(req.prefix_indices) :],
        )

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        if self.page_size != 1:
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices
        req.last_node = new_last_node

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper()

    def evict(self, num_tokens: int):
        if self.disable:
            return

        leaves = self._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)
            if x == self.root_node:
                break
            if x.lock_ref > 0:
                continue

            num_evicted += len(x.value)
            self.token_to_kv_pool_allocator.free(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0:
                heapq.heappush(leaves, x.parent)

            self._record_remove_event(x)

    def inc_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.value)
                self.protected_size_ += len(node.value)
                delta -= len(node.value)
            node.lock_ref += 1
            node = node.parent
        return delta

    def dec_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.value)
                self.protected_size_ -= len(node.value)
                delta += len(node.value)
            node.lock_ref -= 1
            node = node.parent
        return delta

    def evictable_size(self):
        return self.evictable_size_

    def protected_size(self):
        # protected size refers to the size of the cache that is locked
        return self.protected_size_

    def all_values_flatten(self):
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        for root in list(self.private_roots.values()) + list(
            self.public_roots.values()
        ):
            _dfs_helper(root)
        if not values:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        return torch.cat(values)

    ##### Internal Helper Functions #####

    def _visible_to(self, node: TreeNode, user_id: Optional[str]) -> bool:
        """Whether this requester may observe *any* KV from ``node``.

        Visibility is checked before split, hit-count, or value return so a
        denied probe cannot mutate another principal's tree or take a partial
        hit on a shared first token (BOS / chat template).
        """
        # Vanilla SGLang: one global radix tree, no principal isolation.
        if self.safekv_mode == "none":
            return True
        if node.visibility == Visibility.VERIFIED_PUBLIC.value:
            return True
        if user_id is not None and user_id == node.creator_id:
            return True
        if (
            self.safekv_mode == "balanced"
            and node.visibility == Visibility.BUDGETED_SHARED.value
            and not node.permanently_private
        ):
            return True
        return False

    def _reserve_balanced_hit(self, child: TreeNode, user_id: str) -> bool:
        """Reserve one ledger unit for a cross-tenant Budgeted-Shared hit."""
        with child.transition_lock:
            if child.visibility != Visibility.BUDGETED_SHARED.value:
                return False
            fingerprint = self._node_fingerprint(child.key)
            accepted, charged, reason = self.ledger.reserve_hit(
                fingerprint, self.access_budget_B
            )
            if not accepted:
                child.visibility = Visibility.EXHAUSTED_PRIVATE.value
                child.permanently_private = True
                self.metrics.record_event(
                    "balanced_hit_denied",
                    fingerprint=fingerprint,
                    reason=reason,
                    charged_hits=charged,
                )
                return False
            child.access_budget = charged
            self.metrics.increment(
                "cross_tenant_balanced_hits",
                fingerprint=fingerprint,
                budget=child.access_budget,
            )
            if child.access_budget >= self.access_budget_B:
                child.visibility = Visibility.EXHAUSTED_PRIVATE.value
                child.permanently_private = True
                self.metrics.increment(
                    "budget_exhausted_nodes",
                    fingerprint=fingerprint,
                )
            return True

    def _scrub_cross_principal_residual(self) -> None:
        """Overwrite a small device buffer after an isolated miss.

        Victim prefill of the target tokens can leave GPU L2 / allocator
        warmth that makes a later same-prefix miss slightly faster. A short
        write after a denied lookup reduces that residual channel. Vanilla
        ``none`` mode keeps the unprotected timing surface.
        """
        if self.safekv_mode == "none":
            return
        device = getattr(self, "device", None)
        if device is None or (hasattr(device, "type") and device.type == "cpu"):
            return
        try:
            buf = torch.empty(2 * 1024 * 1024, device=device, dtype=torch.float32)
            buf.fill_(0)
            del buf
        except Exception:
            return

    def _match_prefix_helper(self, node: TreeNode, key: List, user_id: Optional[str] = None):
        child_key = self.get_child_key_fn(key)

        value = []
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            if not self._visible_to(child, user_id):
                # Denied: do not split, count, or return any KV.
                break

            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len <= 0:
                break

            if (
                self.safekv_mode == "balanced"
                and child.visibility == Visibility.BUDGETED_SHARED.value
                and user_id is not None
                and user_id != child.creator_id
            ):
                if not self._reserve_balanced_hit(child, user_id):
                    break

            child.hit_count += 1
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node
                break

            value.append(child.value)
            node = child
            key = key[prefix_len:]
            if len(key):
                child_key = self.get_child_key_fn(key)

        return value, node

    def _split_node(self, key, child: TreeNode, split_len: int):
        # new_node -> child
        self._record_remove_event(child)
        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]
        # SafeKV: inherit privacy metadata
        new_node.creator_id = child.creator_id
        new_node.private_tag = child.private_tag
        new_node.need_check_privacy = child.need_check_privacy
        new_node.creator_set = child.creator_set.copy()
        new_node.creator_count = child.creator_count
        new_node.access_budget = child.access_budget
        new_node.permanently_private = child.permanently_private
        new_node.visibility = child.visibility
        new_node.namespace_key = child.namespace_key
        new_node.hit_count = child.hit_count
        new_node.prompt = child.prompt
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        self._record_store_event(new_node)
        self._record_store_event(child)

        return new_node

    def _insert_helper(self, node: TreeNode, key: List, value, prompt: str, user_id: Optional[str] = None):
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]

            prefix_len = self.key_match_fn(node.key, key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            NUM_ = 1024
            while len(value) > 0:
                new_node = TreeNode()
                new_node.parent = node
                new_node.key = key[:NUM_]
                new_node.value = value[:NUM_]
                # SafeKV: new nodes are private by default
                new_node.prompt = prompt
                new_node.private_tag = 1
                new_node.access_budget = 0
                new_node.creator_id = user_id
                new_node.creator_set = {user_id} if user_id else set()
                new_node.creator_count = len(new_node.creator_set)
                namespace = node.namespace_key
                new_node.namespace_key = namespace
                if (
                    namespace is not None
                    and namespace.visibility == Visibility.VERIFIED_PUBLIC
                ):
                    new_node.private_tag = 0
                    new_node.visibility = Visibility.VERIFIED_PUBLIC.value
                    new_node.creator_id = None
                else:
                    new_node.visibility = Visibility.PRIVATE.value
                    if (
                        self.experiment_autoshare
                        and self.safekv_mode == "balanced"
                    ):
                        new_node.visibility = Visibility.BUDGETED_SHARED.value
                        new_node.private_tag = 0
                        new_node.need_check_privacy = False
                    elif self.safekv_mode != "strict":
                        self.private_judge_client.update_privacy(
                            node=new_node,
                            context=prompt,
                            prompt=prompt,
                        )
                node.children[child_key] = new_node
                self.evictable_size_ += min(NUM_, len(value))
                self._record_store_event(new_node)
                node = new_node
                key = key[NUM_:]
                value = value[NUM_:]
                if len(key):
                    child_key = self.get_child_key_fn(key)
        return total_prefix_length

    def _print_helper(self, node: TreeNode, indent: int):
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                len(current_node.key),
                current_node.key[:10],
                f"r={current_node.lock_ref}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == self.get_child_key_fn(
                    child.key
                ), f"{key=}, {self.get_child_key_fn(child.key)=}"

    def _delete_leaf(self, node):
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        self.evictable_size_ -= len(node.key)

    def _total_size_helper(self):
        total_size = 0
        stack = list(self.private_roots.values()) + list(
            self.public_roots.values()
        )
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size

    def _collect_leaves(self):
        ret_list = []
        namespace_roots = list(self.private_roots.values()) + list(
            self.public_roots.values()
        )
        stack = []
        for root in namespace_roots:
            stack.extend(root.children.values())

        while stack:
            cur_node = stack.pop()
            if len(cur_node.children) == 0:
                ret_list.append(cur_node)
            else:
                stack.extend(cur_node.children.values())

        return ret_list

    def _record_store_event(self, node: TreeNode):
        if self.enable_kv_cache_events:
            block_hash = hash(tuple(node.key))
            parent_block_hash = hash(tuple(node.parent.key))
            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=[block_hash],
                    parent_block_hash=parent_block_hash,
                    token_ids=node.key,
                    block_size=len(node.key),
                    lora_id=None,
                )
            )

    def _record_remove_event(self, node: TreeNode):
        if self.enable_kv_cache_events:
            block_hash = hash(tuple(node.key))
            self.kv_event_queue.append(BlockRemoved(block_hashes=[block_hash]))

    def _record_all_cleared_event(self):
        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

    def take_events(self):
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events


if __name__ == "__main__":
    tree = RadixCache(None, None, page_size=1, disable=False)

    tree.insert("Hello")
    tree.insert("Hello")
    tree.insert("Hello_L.A.!")
    tree.pretty_print()
