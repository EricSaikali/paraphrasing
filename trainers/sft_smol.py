import os
import torch
import datasets
from datasets import load_dataset, ClassLabel, DatasetDict
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
import gc
import time

from utils import load_quora_dataset, set_seed

SEED = 42
QUORA_TEST_PROPORTION = 0.2
QUORA_VALID_PROPORTION = 0.5
EPSILON = 1e-6


def load_paraphrase_dataset(tokenizer, max_train=5000, max_eval=500):
    set_seed(SEED)
    quora_dataset = load_quora_dataset(SEED, QUORA_TEST_PROPORTION, QUORA_VALID_PROPORTION)
    dataset = quora_dataset.filter(lambda x: x["is_duplicate"] == 1)

    def preprocess(example):
        source = example["question1"]
        target = example["question2"]
        prompt = f"Paraphrase the following sentence: {source} ### Paraphrase:"

        model_inputs = tokenizer(prompt, padding="max_length", truncation=True, max_length=512)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                target, max_length=128, truncation=True, padding="max_length"
            )
        model_inputs["labels"] = labels["input_ids"].copy()
        return model_inputs

    train_data = dataset["train"].map(preprocess, remove_columns=dataset["train"].column_names)
    eval_data = dataset["valid"].map(preprocess, remove_columns=dataset["valid"].column_names)

    train_data = train_data.select(range(min(len(train_data), max_train)))
    eval_data = eval_data.select(range(min(len(eval_data), max_eval)))

    train_data.set_format("torch")
    eval_data.set_format("torch")
    return train_data, eval_data


def train_paraphraser(output_dir="paraphrase_model_output"):
    model_name = "HuggingFaceTB/SmolLM-135M" #"Qwen/Qwen2.5-0.5B"

    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    train_data, eval_data = load_paraphrase_dataset(tokenizer)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["layers.29.self_attn.q_proj",
                        "layers.29.self_attn.v_proj",
                        "model.layers.29.mlp.up_proj",
                        "model.layers.29.mlp.down_proj"]
        # ["layers.23.self_attn.q_proj", "layers.23.self_attn.v_proj"]
        # ["Wqkv", "fc1", "fc2" ] # ["Wqkv", "out_proj", "fc1", "fc2" ]
    )

    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        eval_accumulation_steps=1,
        logging_steps=10,
        save_steps=250,
        eval_steps=250,
        num_train_epochs=3,
        learning_rate=3e-5,
        bf16=torch.backends.mps.is_available(),
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        eval_strategy="steps",
        save_strategy="steps",
        do_train=True,
        do_eval=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        peft_config=peft_config,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.model.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")

    for _ in range(2):
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(10)

    print(f"Model and tokenizer saved to {output_dir}/final")


if __name__ == "__main__":
    out_dir = "storage/SmolLM-135M-Quora-v2"
    train_paraphraser(output_dir=out_dir)
