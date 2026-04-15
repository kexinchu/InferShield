from __future__ import annotations

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
from collections import defaultdict
from typing import Optional


class TreeNode:

    counter = 0

    def __init__(self, id: Optional[int] = None):
        self.children = defaultdict(TreeNode)
        self.parent = None
        self.key = None
        self.value = None
        self.lock_ref = 0

        self.hit_count = 0
        # indicating the node is loading KV cache from host
        self.loading = False
        # store the host indices of KV cache
        self.host_value = None

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

        # SafeKV privacy metadata
        self.private_tag = 1  # 1 = private (default), 0 = shareable
        self.need_check_privacy = True
        self.creator_id = None  # user_id of the creator
        self.creator_set = set()  # distinct user IDs who created this prefix
        self.creator_count = 0  # len(creator_set)
        self.access_budget = 0  # cross-tenant access budget (set to B on promotion)
        self.permanently_private = False  # prevents re-promotion after budget demotion

        # prompt - user's input
        self.prompt = ""

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    def __lt__(self, other: "TreeNode"):
        return self.hit_count < other.hit_count
