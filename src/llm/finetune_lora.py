import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from config.config import config

class LoraFineTuner:
    def __init__(self):
        self.base_model_name = config.MODEL_NAME
        self.output_dir = config.LORA_OUTPUT_DIR

    def finetune(self, training_dataset_path):
        # Example uses a local text dataset
        dataset = load_dataset("text", data_files={"train": training_dataset_path})
        model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj","v_proj"],  # Example for a GPT model
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        peft_model = get_peft_model(model, peft_config)

        # This is a very rough skeleton; normally you'd use a Trainer or a full training loop
        for batch in dataset["train"]:
            inputs = tokenizer(batch["text"], return_tensors="pt", truncation=True, max_length=512)
            outputs = peft_model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()
            peft_model.step()

        peft_model.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        print("LoRA Fine-tuning completed.")
