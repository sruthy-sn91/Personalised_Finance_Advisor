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
