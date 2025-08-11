

!pip uninstall accelerate peft bitsandbytes transformers trl -y
!pip install accelerate peft==0.13.2 bitsandbytes transformers trl==0.12.0
!pip install huggingface_hub

import os
import torch
from trl import SFTTrainer
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline, logging)

model_identifier = "aboonaji/llama2finetune-v2"
source_dataset = "gamino/wiki_medical_terms"
formatted_dataset = "aboonaji/wiki_medical_terms_llam2_format"

lora_hyper_r = 64
lora_hyper_alpha = 16
lora_hyper_dropout = 0.1

enable_4bit = True
compute_dtype_bnb = "float16"
quant_type_bnb = "nf4"
double_quant_flag = False

results_dir = "./results"
epochs_count = 10
enable_fp16 = False
enable_bf16 = False
train_batch_size = 4
eval_batch_size = 4
accumulation_steps = 1
checkpointing_flag = True
grad_norm_limit = 0.3
train_learning_rate = 2e-4
decay_rate = 0.001
optimizer_type = "paged_adamw_32bit"
scheduler_type = "cosine"
steps_limit = 100
warmup_percentage = 0.03
length_grouping = True
checkpoint_interval = 0
log_interval = 25

enable_packing = False
sequence_length_max = None
device_assignment = {"": 0}

training_data = load_dataset(formatted_dataset, split = "train")

dtype_computation = getattr(torch, compute_dtype_bnb)
bnb_setup = BitsAndBytesConfig(load_in_4bit = enable_4bit,
                               bnb_4bit_quant_type = quant_type_bnb,
                               bnb_4bit_use_double_quant = double_quant_flag,
                               bnb_4bit_compute_dtype = dtype_computation)

llama_model = AutoModelForCausalLM.from_pretrained(model_identifier, quantization_config = bnb_setup, device_map = device_assignment)
llama_model.config.use_case = False
llama_model.config.pretraining_tp = 1

llama_tokenizer = AutoTokenizer.from_pretrained(model_identifier, trust_remote_code = True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"

peft_setup = LoraConfig(lora_alpha = lora_hyper_alpha,
                        lora_dropout = lora_hyper_dropout,
                        r = lora_hyper_r,
                        bias = "none",
                        task_type = "CAUSAL_LM")

train_args = TrainingArguments(output_dir = results_dir,
                               num_train_epochs = epochs_count,
                               per_device_train_batch_size = train_batch_size,
                               per_device_eval_batch_size = eval_batch_size,
                               gradient_accumulation_steps = accumulation_steps,
                               learning_rate = train_learning_rate,
                               weight_decay = decay_rate,
                               optim = optimizer_type,
                               save_steps = checkpoint_interval,
                               logging_steps = log_interval,
                               fp16 = enable_fp16,
                               bf16 = enable_bf16,
                               max_grad_norm = grad_norm_limit,
                               max_steps = steps_limit,
                               warmup_ratio = warmup_percentage,
                               group_by_length = length_grouping,
                               lr_scheduler_type = scheduler_type,
                               gradient_checkpointing = checkpointing_flag)

llama_sftt_trainer = SFTTrainer(model = llama_model,
                                args = train_args,
                                train_dataset = training_data,
                                tokenizer = llama_tokenizer,
                                peft_config = peft_setup,
                                dataset_text_field = "text",
                                max_seq_length = sequence_length_max,
                                packing = enable_packing)

llama_sftt_trainer.train()

user_prompt = "Please tell me about Bursitis"
text_generation_pipe = pipeline(task = "text-generation", model = llama_model, tokenizer = llama_tokenizer, max_length = 500)
generation_result = text_generation_pipe(f"<s>[INST] {user_prompt} [/INST]")
print(generation_result[0]['generated_text'])
