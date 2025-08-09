import queue

tier_1_task_queue = queue.Queue()
tier_2_task_queue = queue.Queue()
tier_3_task_queue = queue.Queue()

tier_2_result_queue = queue.Queue()
tier_3_result_queue = queue.Queue()

result_final_queue = queue.Queue()
