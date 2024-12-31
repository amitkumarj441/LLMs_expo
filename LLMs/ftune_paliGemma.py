import json
import os
from datasets import load_dataset
import loguru
from transformers import PaliGemmaProcessor
from transformers import PaliGemmaForConditionalGeneration
import torch
device = "cuda"
from transformers import  PaliGemmaForConditionalGeneration
from peft import get_peft_model, LoraConfig
from transformers import TrainingArguments

from transformers import Trainer
from swanlab.integration.huggingface import SwanLabCallback
from PIL import Image
import swanlab

swanlab_callback = SwanLabCallback(
    project="paligemema",
    experiment_name="paligemema-3b",
    description="fine-tuning paligemama loar",
    config={
        "model": "paligemma2-3b-pt-224",
        "model_dir": "models/paligemma2-3b-pt-224",
        "dataset": "vqav2-small-sample",
    },
)

# Optional - log this run on a wandb project
os.environ["WANDB_PROJECT"]="pgemema"

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

def load_datasets():
    ds = load_dataset('parquet', data_dir= 'datasets/vqav2-small',split='validation')
    split_ds = ds.train_test_split(test_size=0.95) # we'll use a very small split for demo
    train_ds,test_ds= split_ds["train"],split_ds["test"]
    loguru.logger.info(f"train example:{train_ds[0]},train size:{len(train_ds)}")
    return train_ds,test_ds

def init_model(model_id):
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, device_map="auto")#, quantization_config=bnb_config)
    processor = PaliGemmaProcessor.from_pretrained(model_id)

    for param in model.vision_tower.parameters():
        param.requires_grad = True

    for param in model.multi_modal_projector.parameters():
        param.requires_grad = False
        

    model = get_peft_model(model, lora_config)
    loguru.logger.info(f"train paremeter:{model.print_trainable_parameters()}")
    return model,processor

args=TrainingArguments(
            num_train_epochs=2,
            remove_unused_columns=False,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            warmup_steps=2,
            learning_rate=2e-5,
            weight_decay=1e-6,
            adam_beta2=0.999,
            logging_steps=50,
            optim="adamw_hf", # you can use paged optimizers like paged_adamw_8bit for QLoRA
            save_strategy="steps",
            save_steps=500,
            save_total_limit=1,
            output_dir="paligemma_vqav2",
            bf16=True,
            report_to="none",
            dataloader_pin_memory=False
        )
def eval_model(model,processor,test_ds):
    # Leaving the prompt blank for pre-trained models

    loguru.logger.info(f"eval datasets {len(test_ds)}")
    test_ds_list = list(test_ds)
    test_text_list = []
    for data in test_ds_list[:10]:
        model_inputs = processor(text="<image>" + data["question"], images=data["image"], return_tensors="pt").to(torch.bfloat16).to(device)
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]
            decoded = processor.decode(generation, skip_special_tokens=True)
            predict = {"predict":{decoded}}
            predict=json.dumps(predict,ensure_ascii=False)
            result_text = f"{data}"
            test_text_list.append(swanlab.Text(result_text, caption=predict))
    swanlab.log({"Prediction": test_text_list})
    swanlab.finish()
    
    

def execute_train():
    model_id = "models/paligemma2-3b-pt-224"
    train_ds,test_ds = load_datasets()
    model,processor = init_model(model_id)
    # image_token = processor.tokenizer.convert_tokens_to_ids("<image>")
    def build_datasets(examples):
        texts = ["<image>" + example["question"] for example in examples]
        labels= [example['multiple_choice_answer'] for example in examples]
        ##resize((128,128),Image.Resampling.LANCZOS
        images = [example["image"].convert("RGB") for example in examples]
        # loguru.logger.info(f"text:{texts},lables:{labels},images:{images}")
        tokens = processor(text=texts, images=images, suffix=labels,
                            return_tensors="pt", padding="longest")

        tokens = tokens.to(model.dtype).to(device)
        # loguru.logger.info(f"token shape:{tokens['pixel_values'].shape,tokens['input_ids'].shape,tokens['labels'].shape}")
        return tokens
    trainer = Trainer(
        model=model,
        train_dataset=train_ds ,
        data_collator=build_datasets,
        args=args,
        callbacks=[swanlab_callback]
        )
    trainer.train()
    eval_model(model,processor,test_ds)
    
    
if __name__ == "__main__":
    loguru.logger.info(f"Spin fine tuning of paliGemma")
    execute_train()
    
