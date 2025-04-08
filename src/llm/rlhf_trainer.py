import json
import os

class RLHFTrainer:
    def __init__(self):
        self.feedback_file = "feedback.json"

    def record_feedback(self, user_query, model_response, user_feedback):
        # Create file if it doesn't exist
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file, "w") as f:
                json.dump([], f)

        # Load existing feedback data
        try:
            with open(self.feedback_file, "r") as f:
                content = f.read().strip()
                data = json.loads(content) if content else []
        except json.JSONDecodeError:
            data = []

        # Append new feedback
        data.append({
            "user_query": user_query,
            "model_response": model_response,
            "user_feedback": user_feedback
        })

        # Save updated feedback
        with open(self.feedback_file, "w") as f:
            json.dump(data, f, indent=2)



# import json
# import os
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from torch.optim import AdamW  # Using AdamW from torch.optim
# from config.config import config

# class RLHFTrainer:
#     def __init__(self):
#         self.feedback_file = "feedback.json"
#         # Initialize the sentiment analysis pipeline with an explicit model for production use.
#         self.sentiment_analyzer = pipeline(
#             "sentiment-analysis",
#             model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
#             revision="main"
#         )

#     def record_feedback(self, user_query, model_response, user_feedback):
#         # Create the feedback file if it doesn't exist.
#         if not os.path.exists(self.feedback_file):
#             with open(self.feedback_file, "w") as f:
#                 json.dump([], f)

#         # Load existing feedback data.
#         try:
#             with open(self.feedback_file, "r") as f:
#                 content = f.read().strip()
#                 data = json.loads(content) if content else []
#         except json.JSONDecodeError:
#             data = []

#         # Append new feedback.
#         data.append({
#             "user_query": user_query,
#             "model_response": model_response,
#             "user_feedback": user_feedback
#         })

#         # Save the updated feedback.
#         with open(self.feedback_file, "w") as f:
#             json.dump(data, f, indent=2)

#     def retrain_with_feedback(self):
#         """
#         This method implements a simplified PPO update to fine-tune the model based on human feedback.
#         For each feedback entry, it:
#           1. Uses a sentiment analysis model to automatically assign a reward.
#           2. Computes the log probabilities of the target response under both an old (frozen) policy and the current policy.
#           3. Computes a PPO loss with clipping and updates the model.
        
#         Note: This is a simplified demonstration.
#         """
#         # Load feedback data.
#         try:
#             with open(self.feedback_file, "r") as f:
#                 feedback_data = json.load(f)
#         except Exception as e:
#             print("Error loading feedback data:", e)
#             return

#         if not feedback_data:
#             print("No feedback data available for retraining.")
#             return

#         model_name = config.MODEL_NAME
#         # Retrieve token from environment; if token is not set, assume a public model.
#         token = os.environ.get("HF_TOKEN")

#         print("Loading model:", model_name)
#         if token:
#             model = AutoModelForCausalLM.from_pretrained(model_name)
#             tokenizer = AutoTokenizer.from_pretrained(model_name)
#         else:
#             model = AutoModelForCausalLM.from_pretrained(model_name)
#             tokenizer = AutoTokenizer.from_pretrained(model_name)
#         model.train()

#         optimizer = AdamW(model.parameters(), lr=5e-5)
#         epsilon = 0.2  # PPO clipping parameter.
#         num_epochs = 1  # For demonstration purposes.

#         # Create a frozen copy of the model (old policy) for computing old log probabilities.
#         if token:
#             old_model = AutoModelForCausalLM.from_pretrained(model_name, token=token)
#         else:
#             old_model = AutoModelForCausalLM.from_pretrained(model_name)
#         old_model.load_state_dict(model.state_dict())
#         old_model.eval()

#         for epoch in range(num_epochs):
#             epoch_loss = 0.0
#             for entry in feedback_data:
#                 user_query = entry["user_query"]
#                 model_response = entry["model_response"]
#                 user_feedback = entry["user_feedback"]

#                 # Use sentiment analysis to assign a reward.
#                 sentiment_result = self.sentiment_analyzer(user_feedback)[0]
#                 label = sentiment_result["label"]
#                 score = sentiment_result["score"]
#                 if label.upper() == "POSITIVE":
#                     reward = score  # Reward between 0 and 1.
#                 elif label.upper() == "NEGATIVE":
#                     reward = -score  # Negative reward.
#                 else:
#                     reward = 0.0

#                 # Debug: print reward information.
#                 print(f"Processing feedback: '{user_feedback}' -> Reward: {reward}")

#                 # Prepare inputs: treat user_query as the prompt and model_response as the target.
#                 inputs = tokenizer(user_query, return_tensors="pt")
#                 labels = tokenizer(model_response, return_tensors="pt")["input_ids"]

#                 # Compute old log probabilities (without gradient) using old_model.
#                 with torch.no_grad():
#                     old_outputs = old_model(**inputs, labels=labels)
#                     old_logits = old_outputs.logits
#                     old_log_probs = torch.nn.functional.log_softmax(old_logits, dim=-1)
#                     old_target_log_probs = old_log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
#                     old_log_prob = old_target_log_probs.sum(dim=-1)

#                 # Compute new log probabilities using the current model.
#                 new_outputs = model(**inputs, labels=labels)
#                 new_logits = new_outputs.logits
#                 new_log_probs = torch.nn.functional.log_softmax(new_logits, dim=-1)
#                 new_target_log_probs = new_log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
#                 new_log_prob = new_target_log_probs.sum(dim=-1)

#                 # Compute the probability ratio for PPO.
#                 ratio = torch.exp(new_log_prob - old_log_prob)

#                 # Use the reward as the advantage (baseline assumed to be 0).
#                 advantage = reward

#                 # Compute PPO loss with clipping.
#                 clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
#                 ppo_loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()

#                 # Backpropagation and update.
#                 ppo_loss.backward()
#                 optimizer.step()
#                 optimizer.zero_grad()
#                 epoch_loss += ppo_loss.item()

#             print(f"Epoch {epoch + 1}: Total PPO Loss = {epoch_loss}")
#             # Update the old_model with new weights after each epoch.
#             old_model.load_state_dict(model.state_dict())

#         # Save the retrained model for future inference.
#         output_dir = "./data/rlhf_model"
#         os.makedirs(output_dir, exist_ok=True)
#         model.save_pretrained(output_dir)
#         tokenizer.save_pretrained(output_dir)
#         print("RLHF retraining with PPO completed.")
