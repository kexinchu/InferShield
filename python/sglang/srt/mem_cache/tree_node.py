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

# add by kexinchu --- start
from sglang import get_epoch
# add by kexinchu --- end

class TreeNode:

    counter = 0

    def __init__(self, id: Optional[int] = None):
        self.children = defaultdict(TreeNode)
        self.parent = None
        self.key = None
        self.value = None
        self.lock_ref = 0
        # self.last_access_time = time.monotonic()

        self.hit_count = 0 # already have hit_count
        # indicating the node is loading KV cache from host
        self.loading = False
        # store the host indices of KV cache
        self.host_value = None

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

        # add by kexinchu --- start
        self.private = True  # default is private node
        self.need_check_privacy = True
        self.owner_id = None
        self.u_cnt_l = set() # the hit user_id list
        self.hit_pre = 0
        self.u_cnt_pre = 0 # 上一个time_window的hit_user_count
        self.epoch = get_epoch()
        self.after_merged = False # 是否被合并过 (合并)
        self.merged_key = []
        self.merged_value = []
        self.is_sub_root = False # 是否是隐私节点合并后的根节点

        # prompt - user's input
        self.prompt = ""
        # add by kexinchu --- end

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    # add by kexinchu --- start
    def __lt__(self, other: "TreeNode"):
        # 改为使用epoch
        # return self.last_access_time < other.last_access_time
        return self.epoch < other.epoch
    # add by kexinchu --- end


