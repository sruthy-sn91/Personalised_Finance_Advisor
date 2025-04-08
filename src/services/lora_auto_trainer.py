# import os
# import time
# import threading
# from src.llm.finetune_lora import LoraFineTuner

# class LoraAutoTrainer:
#     def __init__(self, training_file_path, check_interval=60):
#         self.training_file_path = training_file_path
#         self.check_interval = check_interval  # seconds
#         self.last_modified_time = None
#         self.trainer = LoraFineTuner()
#         self.running = False

#     def _has_file_changed(self):
#         try:
#             current_mod_time = os.path.getmtime(self.training_file_path)
#             if self.last_modified_time is None:
#                 self.last_modified_time = current_mod_time
#                 return False
#             if current_mod_time != self.last_modified_time:
#                 self.last_modified_time = current_mod_time
#                 return True
#         except FileNotFoundError:
#             return False
#         return False

#     def _watch_file_and_train(self):
#         print(f"[LoraAutoTrainer] Watching '{self.training_file_path}' for changes...")
#         while self.running:
#             if self._has_file_changed():
#                 print(f"[LoraAutoTrainer] Detected change in '{self.training_file_path}', retraining...")
#                 self.trainer.finetune(self.training_file_path)
#                 print("[LoraAutoTrainer] Fine-tuning complete.")
#             time.sleep(self.check_interval)

#     def start(self):
#         if not self.running:
#             self.running = True
#             thread = threading.Thread(target=self._watch_file_and_train, daemon=True)
#             thread.start()
#             print("[LoraAutoTrainer] Started auto-retraining thread.")

#     def stop(self):
#         self.running = False
#         print("[LoraAutoTrainer] Stopped auto-retraining.")