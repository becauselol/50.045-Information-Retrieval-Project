import torch
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from tqdm.notebook import tqdm
from transformers.utils import logging
from transformers import GenerationConfig
import json
from peft import LoraConfig, get_peft_model, TaskType,PeftModel
from utils import get_gold_passages, response_gen, extract_ans_t5

model_name = 'google/flan-t5-xl'

model = AutoModelForSeq2SeqLM.from_pretrained(model_name, load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.model_max_length = 2048

generation_config = GenerationConfig.from_pretrained(model_name)
generation_config.max_new_tokens = 10
generation_config.do_sample = False


zero_shot_prompt_flant5 = """You are given an instruction and some relevant wikipedia information related to the instruction. The instruction is to respond to the QUESTION given the PASSAGE.
QUESTION: {}
PASSAGE:
{}

ANSWER:
"""



ds = load_dataset("hotpot_qa", "distractor")
use_finetune = False
if use_finetune:
    model = PeftModel.from_pretrained(model, "flan_t5_xl_question_answering/checkpoint-600")

def main():

    gold_dataset = get_gold_passages(ds)
    for data in tqdm(gold_dataset):
   
        input_prompt = zero_shot_prompt_flant5.format(data["question"], data["passage"])
        inputs = tokenizer(input_prompt,return_tensors='pt')

        with torch.no_grad():
            input_ids = inputs['input_ids'].to("cuda")
            
            output = response_gen(model, tokenizer, input_ids, generation_config)

            extracted_ans = extract_ans_t5(output)
            print("Answer responded:", extracted_ans)
            print(extracted_ans,'Ground truth',ds['train'][i]['label'])
            correct += int(extract_ans_t5(output, data["answer"]))
            print(f"Current accuracy: {correct/(i+1)}")
    


if __name__ == '__main__':
    main()
    
        
