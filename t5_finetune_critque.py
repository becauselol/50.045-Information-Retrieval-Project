from datasets import concatenate_datasets
from transformers import TrainingArguments, Trainer
import os
import torch
from peft import prepare_model_for_int8_training,LoraConfig, get_peft_model, TaskType
import json 
import random

with open('wiki_qa_paragraph.json',mode='r') as f:
        term_dict = json.load(f)
    
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "google/flan-t5-xl"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name, load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
TASK_LIST = 
{
}

ds = load_dataset('wiki_qa')

def prompt_template_for_task1(batch):
    q = batch['question']
    answer = batch['answer']
    if batch['document_title'] in term_dict:
        input = f"Context: {term_dict[batch['document_title']]}\nQ: {q}\nReasoning: {answer}"
    else:
        input = f"Context: No context\nQ: {q}\nReasoning: {answer}"

    if batch['label'] == 1:
        output = "The reasoning is correct."
    else:
        output = "The reasoning is wrong."

    
    return {"input":input,"output":output}
    
def prompt_template_for_task3(batch):
    q = batch['question']
    answer = batch['answer']
    random_context = random.choice(list(term_dict.keys()))
    while random_context == batch['document_title']:
        random_context = random.choice(list(term_dict.keys()))
    
    input = f"Context: {term_dict[random_context]}\nQ: {q}\nReasoning: {answer}"

    output = f"I will need more context to answer this question, I should search up more information on TERM[{batch['document_title']}]"
    return {"input":input,"output":output}
    
def prompt_template_for_task5(batch):
    q = batch['question']
    input = f"Q: {q}"
    output = f"In order to answer this question, I need to search for more information on the TERM[{batch['document_title']}]"
    
    return {"input":input,"output":output}
    

text_column = "input"
label_column = "output"
max_length = 512




def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM"
)

model = prepare_model_for_int8_training(model)
model = get_peft_model(model, lora_config)
print_trainable_parameters(model)


def preprocess_function(examples):
    inputs = examples[text_column]
    targets = examples[label_column]
    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(targets, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    
    return model_inputs


def main():
    
    ds = load_dataset('wiki_qa')
    ds_temp = ds['train'].select(range(10000))
    task1 = ds_temp.map(prompt_template_for_task1)
    task3 = ds_temp.map(prompt_template_for_task3)
    task5 = ds_temp.map(prompt_template_for_task5)
    ds_encoded = concatenate_datasets([task1,task3,task5])
    
    processed_datasets = ds_encoded.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset")


    
    training_args = TrainingArguments(
        output_dir="trained_model/flan_t5_multi_task",
        evaluation_strategy="epoch",
        learning_rate=1e-3,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=16,
        num_train_epochs=2,
        evaluation_steps=5000,
        save_steps=100,
        logging_steps=100,
        save_total_limit=8)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_datasets
    )
    trainer.train()

    
if __name__ == '__main__':
    main()
