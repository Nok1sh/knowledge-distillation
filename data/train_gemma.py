import os

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from trl import SFTConfig, SFTTrainer

wandb.login()

os.environ["WANDB_PROJECT"] = "gemma3-countdown-distillation"


class GemmaCompletionCollator(DataCollatorForLanguageModeling):
    def __init__(
        self, tokenizer, response_template="<start_of_turn>model\n", *args, **kwargs
    ):
        super().__init__(tokenizer=tokenizer, mlm=False, *args, **kwargs)
        self.response_template = response_template

    def torch_call(self, examples):
        batch = super().torch_call(examples)

        response_token_ids = self.tokenizer.encode(
            self.response_template, add_special_tokens=False
        )
        response_tensor = torch.tensor(response_token_ids)

        for i in range(len(batch["labels"])):
            labels = batch["labels"][i]

            for j in range(len(labels) - len(response_token_ids)):
                if torch.equal(
                    labels[j : j + len(response_token_ids)],
                    response_tensor.to(labels.device),
                ):
                    labels[: j + len(response_token_ids)] = -100
                    break
        return batch


model_id = "google/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

collator = GemmaCompletionCollator(tokenizer=tokenizer)


data_path = "dataset_3.jsonl"

raw_dataset = load_dataset("json", data_files=data_path, split="train")

dataset_split = raw_dataset.train_test_split(test_size=0.02, seed=42)
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"]


def formatting_prompts_func(example):
    return tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )


lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    task_type="CAUSAL_LM",
)

training_args = SFTConfig(
    output_dir="checkpoints",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    max_grad_norm=1.0,
    num_train_epochs=2,
    logging_steps=5,
    save_steps=50,
    eval_strategy="steps",
    eval_steps=80,
    report_to="wandb",
    run_name="distill-gemma1b-run-5",
    bf16=True,
    max_length=2048,
    dataset_text_field="text",
    optim="paged_adamw_32bit",
    gradient_checkpointing=True,
    prediction_loss_only=True,
    eval_accumulation_steps=1,
    per_device_eval_batch_size=1,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    peft_config=lora_config,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    processing_class=tokenizer,
)

trainer.train()

wandb.finish()
