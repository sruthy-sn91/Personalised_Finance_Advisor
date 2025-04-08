import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from config.config import config

class LoraFineTuner:
    def __init__(self):
        self.base_model_name = config.MODEL_NAME
        self.output_dir = config.LORA_OUTPUT_DIR
        self.device = torch.device("cpu")

    def preprocess_function(self, example, tokenizer):
        prompt = f"Instruction: {example['instruction']}\nInput: {example.get('input', '')}\nOutput:"
        target = example['output']
        inputs = tokenizer(prompt, truncation=True, padding='max_length', max_length=256)
        targets = tokenizer(target, truncation=True, padding='max_length', max_length=256)
        inputs["labels"] = targets["input_ids"]
        return inputs

    def finetune(self, training_dataset_path):
        print(f"[INFO] Finetuning model with LoRA using {training_dataset_path}")
        dataset = load_dataset('json', data_files=training_dataset_path, split='train')
        print(f"[INFO] Loaded dataset with {len(dataset)} examples")

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(self.base_model_name).to(self.device)

        target_modules = []
        for name, _ in model.named_modules():
            if any(sub in name for sub in ['q_proj', 'v_proj', 'c_attn']):
                target_modules.append(name.split('.')[-1])
        target_modules = list(set(target_modules)) or ["c_attn"]

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, lora_config)
        tokenized_dataset = dataset.map(lambda x: self.preprocess_function(x, tokenizer), batched=False)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            save_total_limit=1,
            learning_rate=2e-4,
            logging_dir="./logs",
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer
        )

        print("[INFO] Starting training...")
        trainer.train()
        model.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        print("[✅ SUCCESS] LoRA fine-tuning completed.")

if __name__ == "__main__":
    LoraFineTuner().finetune("datasets/lora_training_data.json")
