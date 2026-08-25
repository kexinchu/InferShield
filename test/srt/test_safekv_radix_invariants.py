import time
import unittest
import sys
import types

import torch

# RadixCache only needs these names for annotations in this CPU-only invariant
# test. Stubbing avoids importing the full CUDA model stack during collection.
memory_pool_stub = types.ModuleType("sglang.srt.mem_cache.memory_pool")
memory_pool_stub.ReqToTokenPool = object
memory_pool_stub.TokenToKVPoolAllocator = object
sys.modules["sglang.srt.mem_cache.memory_pool"] = memory_pool_stub

from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.safekv_policy import PublicRegistry


class _NoopDetector:
    def update_privacy(self, **kwargs):
        raise AssertionError("Strict mode must not submit detector promotion")


class TestSafeKVRadixInvariants(unittest.TestCase):
    def setUp(self):
        self.cache = RadixCache(
            None,
            None,
            page_size=1,
            private_judge_client=_NoopDetector(),
            safekv_mode="strict",
            operator_key="test-operator-key",
            policy_epoch=1,
            model_id="phi4",
            tokenizer_version="tokenizer-v1",
        )
        self.tokens = [11, 12, 13, 14]

    @staticmethod
    def _values(start):
        return torch.tensor(
            [start, start + 1, start + 2, start + 3],
            dtype=torch.int64,
        )

    def test_equal_private_prefixes_use_distinct_namespaces(self):
        self.cache.insert(
            self.tokens, self._values(100), user_id="victim"
        )
        attacker_hit, _ = self.cache.match_prefix(
            self.tokens, user_id="attacker"
        )
        self.assertEqual(len(attacker_hit), 0)

        self.cache.insert(
            self.tokens, self._values(200), user_id="attacker"
        )
        victim_hit, _ = self.cache.match_prefix(
            self.tokens, user_id="victim"
        )
        attacker_hit, _ = self.cache.match_prefix(
            self.tokens, user_id="attacker"
        )
        self.assertEqual(victim_hit.tolist(), [100, 101, 102, 103])
        self.assertEqual(attacker_hit.tolist(), [200, 201, 202, 203])

        snapshot = self.cache.safekv_snapshot()
        private_variants = [
            variant
            for variant in snapshot["variants"]
            if variant["namespace_visibility"] == "private"
        ]
        self.assertEqual(
            {variant["namespace_identity"] for variant in private_variants},
            {"victim", "attacker"},
        )
        self.assertEqual(
            len({variant["kv_address_id"] for variant in private_variants}), 2
        )
        self.assertTrue(
            all(value == 0 for value in snapshot["counters"].values())
        )

    def test_valid_authorization_creates_separate_public_object(self):
        self.cache.insert(
            self.tokens, self._values(100), user_id="victim"
        )
        signer = PublicRegistry(b"test-operator-key", policy_epoch=1)
        authorization = signer.issue(
            "public-object-1",
            "operator",
            "phi4",
            "tokenizer-v1",
            self.tokens,
            time.time() + 60,
        ).to_dict()

        prewarm_hit, _ = self.cache.match_prefix(
            self.tokens,
            user_id="operator",
            authorization=authorization,
        )
        self.assertEqual(len(prewarm_hit), 0)
        self.cache.insert(
            self.tokens,
            self._values(300),
            user_id="operator",
            authorization=authorization,
        )

        attacker_hit, _ = self.cache.match_prefix(
            self.tokens, user_id="attacker"
        )
        victim_private_root = self.cache.private_roots["victim"]
        victim_node = next(iter(victim_private_root.children.values()))
        public_node = next(
            iter(self.cache.public_roots["public-object-1"].children.values())
        )
        self.assertEqual(attacker_hit.tolist(), [300, 301, 302, 303])
        self.assertNotEqual(victim_node.id, public_node.id)
        self.assertNotEqual(
            self.cache._value_identity(victim_node.value),
            self.cache._value_identity(public_node.value),
        )
        snapshot = self.cache.safekv_snapshot()
        self.assertEqual(snapshot["counters"]["public_object_created"], 1)
        self.assertEqual(
            snapshot["counters"]["unauth_public_promotions"], 0
        )
        self.assertIn("victim", snapshot["private_principals"])
        self.assertEqual(
            snapshot["public_object_ids"], ["public-object-1"]
        )

    def test_invalid_authorizations_fail_closed(self):
        signer = PublicRegistry(b"wrong-key", policy_epoch=1)
        forged = signer.issue(
            "forged",
            "attacker",
            "phi4",
            "tokenizer-v1",
            self.tokens,
            time.time() + 60,
        ).to_dict()
        self.cache.match_prefix(
            self.tokens, user_id="attacker", authorization=forged
        )
        self.cache.insert(
            self.tokens,
            self._values(400),
            user_id="attacker",
            authorization=forged,
        )
        snapshot = self.cache.safekv_snapshot()
        self.assertEqual(snapshot["public_object_ids"], [])
        self.assertEqual(snapshot["private_principals"], ["attacker"])
        reasons = [
            event["attributes"].get("reason")
            for event in snapshot["events"]
            if event["name"] == "authorization_verified"
        ]
        self.assertIn(PublicRegistry.INVALID_MAC, reasons)

    def test_revocation_hides_public_object_and_blocks_reinstall(self):
        signer = PublicRegistry(b"test-operator-key", policy_epoch=1)
        valid = signer.issue(
            "revocable-object",
            "operator",
            "phi4",
            "tokenizer-v1",
            self.tokens,
            time.time() + 60,
        ).to_dict()
        self.cache.insert(
            self.tokens,
            self._values(500),
            user_id="operator",
            authorization=valid,
        )
        before, _ = self.cache.match_prefix(
            self.tokens, user_id="attacker"
        )
        self.assertEqual(before.tolist(), [500, 501, 502, 503])

        revoked = signer.issue(
            "revocable-object",
            "operator",
            "phi4",
            "tokenizer-v1",
            self.tokens,
            time.time() + 60,
            revoked=True,
        ).to_dict()
        denied, _ = self.cache.match_prefix(
            self.tokens,
            user_id="operator",
            authorization=revoked,
        )
        self.assertEqual(len(denied), 0)

        post_revoke, _ = self.cache.match_prefix(
            self.tokens, user_id="attacker"
        )
        self.assertEqual(len(post_revoke), 0)
        stale_reinstall, _ = self.cache.match_prefix(
            self.tokens,
            user_id="operator",
            authorization=valid,
        )
        self.assertEqual(len(stale_reinstall), 0)
        reasons = [
            event["attributes"].get("reason")
            for event in self.cache.safekv_snapshot()["events"]
            if event["name"] == "authorization_verified"
        ]
        self.assertEqual(reasons[-2:], ["revoked", "revoked"])


class _SilentDetector:
    def update_privacy(self, **kwargs):
        return


class TestSafeKVLookupIsolation(unittest.TestCase):
    """Denied probes must not observe or mutate another principal's KV."""

    def _cache(self, mode: str) -> RadixCache:
        return RadixCache(
            None,
            None,
            page_size=1,
            private_judge_client=_SilentDetector(),
            safekv_mode=mode,
            operator_key="test-operator-key",
            policy_epoch=1,
            model_id="phi4",
            tokenizer_version="tokenizer-v1",
            access_budget_B=10,
        )

    @staticmethod
    def _values(start, length=8):
        return torch.arange(start, start + length, dtype=torch.int64)

    def test_exact_key_match_rejects_near_copies(self):
        from sglang.srt.mem_cache.radix_cache import _key_match_page_size1

        key = [1, 2, 3, 4, 5, 6, 7, 8]
        near = [1, 9, 9, 9, 9, 9, 9, 9]
        self.assertEqual(_key_match_page_size1(key, key), 8)
        self.assertEqual(_key_match_page_size1(key, near), 1)

    def test_strict_shorter_probe_does_not_split_or_hit(self):
        cache = self._cache("strict")
        tokens = [11, 12, 13, 14, 15, 16, 17, 18]
        cache.insert(tokens, self._values(100), user_id="victim")
        victim_root = cache.private_roots["victim"]
        victim_node = next(iter(victim_root.children.values()))
        original_key = list(victim_node.key)

        hit, _ = cache.match_prefix(tokens[:3], user_id="attacker")
        self.assertEqual(len(hit), 0)
        self.assertEqual(list(victim_node.key), original_key)
        self.assertEqual(len(victim_root.children), 1)

    def test_strict_near_copy_does_not_hit(self):
        cache = self._cache("strict")
        tokens = [11, 12, 13, 14, 15, 16, 17, 18]
        near = [11, 22, 23, 24, 25, 26, 27, 28]
        cache.insert(tokens, self._values(100), user_id="victim")
        hit, _ = cache.match_prefix(near, user_id="attacker")
        self.assertEqual(len(hit), 0)

    def test_balanced_private_prefix_does_not_leak(self):
        cache = self._cache("balanced")
        tokens = [11, 12, 13, 14, 15, 16, 17, 18]
        cache.insert(tokens, self._values(100), user_id="victim")
        exact, _ = cache.match_prefix(tokens, user_id="attacker")
        prefix, _ = cache.match_prefix(tokens[:4], user_id="attacker")
        near, _ = cache.match_prefix([11, 99, 98, 97, 96, 95, 94, 93], user_id="attacker")
        self.assertEqual(len(exact), 0)
        self.assertEqual(len(prefix), 0)
        self.assertEqual(len(near), 0)

    def test_balanced_budgeted_full_prefix_still_hits(self):
        cache = self._cache("balanced")
        tokens = [11, 12, 13, 14, 15, 16, 17, 18]
        cache.insert(tokens, self._values(200), user_id="victim")
        node = next(iter(cache.private_roots["victim"].children.values()))
        node.visibility = "budgeted_shared"
        hit, _ = cache.match_prefix(tokens, user_id="attacker")
        self.assertEqual(hit.tolist(), list(range(200, 208)))
        snapshot = cache.safekv_snapshot()
        self.assertEqual(snapshot["counters"]["cross_tenant_balanced_hits"], 1)


if __name__ == "__main__":
    unittest.main()
