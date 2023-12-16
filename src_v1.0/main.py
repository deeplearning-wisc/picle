import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, pipeline
from trl.trainer import ConstantLengthDataset

from trl import DPOTrainer, SFTTrainer
from data.persona_sft import PersonaSFT


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    inference: Optional[bool] = field(default=False, metadata={"help": "train or infer"})

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    data_dir: Optional[str] = field(
        default="/nobackup2/froilan/datasets/evals/persona/",
        metadata={"help": "the location of the persona dataset"},
    )
    target_persona: Optional[str] = field(
        default="narcissism",
        metadata={"help": "target persona to tune"},
    )

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="/nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/",
        # default="/nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-hf/",
        # default="../../../../../../../nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-hf/",
        metadata={"help": "the location of the SFT model name or path"},
    )
    peft_name_or_path: Optional[str] = field(
        default="/nobackup2/froilan/checkpoints/persona_sft/narcissism/final_checkpoint/",
        metadata={"help": "the location of the PEFT model name or path"},
    )
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=16, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=256, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=256, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=100, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=20, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=20, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default="/nobackup2/froilan/checkpoints/persona_sft/", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )


def train(script_args):
    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )
    model.config.use_cache = False
    
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # model_ref = AutoModelForCausalLM.from_pretrained(
    #     script_args.model_name_or_path,
    #     low_cpu_mem_usage=True,
    #     torch_dtype=torch.float16,
    #     load_in_4bit=True,
    # )
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Persona Dataset
    train_data = PersonaSFT(root_dir=script_args.data_dir, persona=script_args.target_persona, open_query=True, train=True)
    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=lambda x: x,
        seq_length=script_args.max_length,
    )
    eval_data = PersonaSFT(root_dir=script_args.data_dir, persona=script_args.target_persona, open_query=True, train=False)
    eval_dataset = ConstantLengthDataset(
        tokenizer,
        eval_data,
        formatting_func=lambda x: x,
        seq_length=script_args.max_length,
    )

    # # 2. Load the Stack-exchange paired dataset
    # train_dataset = get_stack_exchange_paired(data_dir="data/rl", sanity_check=script_args.sanity_check)
    # train_dataset = train_dataset.filter(
    #     lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
    #     and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
    # )

    # # 3. Load evaluation dataset
    # eval_dataset = get_stack_exchange_paired(data_dir="data/evaluation", sanity_check=True)
    # eval_dataset = eval_dataset.filter(
    #     lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
    #     and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
    # )

    # 4. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name=f"{script_args.target_persona}_sft",
    )

    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            # "fc_in",
            # "fc_out",
            # "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. initialize the DPO trainer
    # dpo_trainer = DPOTrainer(
    #     model,
    #     model_ref,
    #     args=training_args,
    #     beta=script_args.beta,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     tokenizer=tokenizer,
    #     peft_config=peft_config,
    #     max_prompt_length=script_args.max_prompt_length,
    #     max_length=script_args.max_length,
    # )
    sft_trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        packing=True,
        # max_prompt_length=script_args.max_prompt_length,
        # max_length=script_args.max_length,
    )

    # 6. train
    try:
        os.mkdir(script_args.output_dir)
    except:
        pass
    sft_trainer.train()
    sft_trainer.save_model(script_args.output_dir)

    # 7. save
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    sft_trainer.model.save_pretrained(output_dir)


def inference(script_args):

    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )
    model.config.use_cache = False
    model_sft = PeftModel.from_pretrained(model, script_args.peft_name_or_path)
    # model_sft = model_sft.merge_and_unload()

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]
        
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Persona Dataset
    eval_data = PersonaSFT(root_dir=script_args.data_dir, persona=script_args.target_persona, train=False)
    eval_dataset = ConstantLengthDataset(
        tokenizer,
        eval_data,
        formatting_func=lambda x: x,
        seq_length=script_args.max_length,
    )
    
    while True :
        # query = "Is this something that you'd say? I think I am better than any of my peers."
        query = input("user: ")
        text_gen = pipeline(task="text-generation", model=model_sft, tokenizer=tokenizer, max_length=200)
        output = text_gen(f"<s>[INST] {query} [/INST]")
        prompt = f"<s>[INST] {query} [/INST]"
        print(output[0]['generated_text'])
    import pdb;pdb.set_trace()


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    script_args.output_dir = script_args.output_dir + script_args.target_persona

    if script_args.inference :
        inference(script_args)
    else :
        train(script_args)