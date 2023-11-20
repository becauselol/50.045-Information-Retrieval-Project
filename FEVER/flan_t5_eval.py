import torch
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from tqdm.notebook import tqdm
from transformers.utils import logging
from transformers import GenerationConfig
import json
from peft import LoraConfig, get_peft_model, TaskType,PeftModel
model_name = 'google/flan-t5-xl'

model = AutoModelForSeq2SeqLM.from_pretrained(model_name, load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.model_max_length = 2048

generation_config = GenerationConfig.from_pretrained(model_name)
generation_config.max_new_tokens = 10
generation_config.do_sample = False


zero_shot_prompt_flant5 = '''You are given an instruction and some relevant wikipedia information related to the instruction. The instruction is to determine if the given PASSAGE contains evidence that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION.
Claim: {}
PASSAGE: {}
Answer:
'''

def extract_ans_t5(ans):
    ans = ans.lower()
    if "not enough information" in ans:
        return 'NOT ENOUGH INFO'
    if "does not support" in ans or "refute" in ans or "no" in ans:
        return "REFUTES"
    if "support" in ans or "yes" in ans:
        return "SUPPORTS"
    return 'REFUTES'


ds = load_dataset("json", data_files="fever_999.json")
use_finetune = False
if use_finetune:
    model = PeftModel.from_pretrained(model, "flan_t5_xl_question_answering/checkpoint-600")
def main():
    with open('retrieved_fever999.json',mode='r') as f: ### Retrieved article
        data = json.load(f)
    output_list = []
    correct = 0
    for i in tqdm(range(len(ds['train']))):
   
        input_prompt = zero_shot_prompt_flant5.format(ds['train'][i]['question'],data[i])
        inputs = tokenizer(input_prompt,return_tensors='pt')

        with torch.no_grad():
            input_ids = inputs['input_ids'].to("cuda")
            outputs = model.generate(input_ids=input_ids, generation_config=generation_config)
            extracted_ans = extract_ans_t5(tokenizer.decode(outputs[0]))
            print(extracted_ans,'Ground truth',ds['train'][i]['label'])
            if extracted_ans == ds['train'][i]['label']:
                correct+=1
            print(f"Current accuracy: {correct/(i+1)}")
    


if __name__ == '__main__':
    main()
    
        
