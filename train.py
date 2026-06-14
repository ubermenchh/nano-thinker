# Dependencies
import os
import torch
import modal

from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset

from reward_func import (
    accuracy_reward,
    format_reward,
    get_cosine_scaled_reward,
    reasoning_steps_reward,
    get_repetition_penalty_reward
)

# Modal App Image
cuda_version = "12.8.0"
flavour = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavour}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "ninja",
        "packaging",
        "wheel",
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "trl",
        "bitsandbytes",
        "wandb",
        "vllm",
        "latex2sympy2_extended[antlr4_13_2]",
        "math-verify[antlr4_13_2]"
    )
    .pip_install(
        "flash-attn==2.7.4.post1", extra_options="--no-build-isolation"
    )
)

app = modal.App(name="grpo-pilot", image=image)

SYSTEM_PROMPT = """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <reasoning> </reasoning> and
<answer> </answer> tags, respectively, i.e., <reasoning> reasoning process here </reasoning>
<answer> answer here </answer>.
"""

def extract_xml_answer(text: str) -> str:
    if "</answer>" not in text and "<answer>" not in text:
        return ""
    answer_part = text.split("<answer>")[-1]
    answer_part = answer_part.split("</answer>")[0]
    return answer_part.strip()

def get_openr1_math_dataset(split: str="train") -> Dataset:
    data = load_dataset("open-r1/OpenR1-Math-220k", split=split)

    def transform_record(x):
        problem_text = x.get("problem", "")
        reasoning_text = x.get("solution", "")
        final_answer = x.get("answer", "")

        xml_output = f"<reasoning>\n{reasoning_text}\n</reasoning>\n<answer>{final_answer}\n</answer>"

        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem_text}
            ],
            "answer": xml_output
        }

    data = data.map(transform_record).remove_columns("messages")
    return data

@app.function(
    gpu=modal.gpu.A100(count=2),
    image=image,
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret")
    ],
    timeout=3600
)
def train(model_name: str):
    output_dir = "qwen-1.5B-openr1-math"
    run_name = "qwen-1.5B-openr1-math-10%"

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_size="left")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="balanced",
        use_cache=False
    )

    dataset = get_openr1_math_dataset("train[:10%]")
    if len(dataset) == 0:
        raise ValueError("No Dataset Found")

    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=2,
        max_prompt_length=4096,
        max_completion_length=4096,
        num_train_epochs=1,
        save_steps=100,
        max_grad_norm=0.1,
        report_to="wandb",
        log_on_each_node=False,
        use_vllm=False
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            accuracy_reward,
            format_reward,
            reasoning_steps_reward,
            get_cosine_scaled_reward(
                min_value_wrong=-1.0,
                max_value_wrong=-0.5,
                min_value_correct=0.5,
                max_value_correct=1.0,
                max_len=1000,
            ),
            get_repetition_penalty_reward(ngram_size=3, max_penalty=-0.5), 
        ],
        args=training_args,
        train_dataset=dataset
    )

    trainer.train()

    trainer.model.push_to_hub(f"ubermenchh/{run_name}")
    trainer.processing_class.push_to_hub(f"ubermenchh/{run_name}")


@app.local_entrypoint()
def main():
    train.remote(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
