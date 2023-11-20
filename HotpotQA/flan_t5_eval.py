import torch
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from tqdm.notebook import tqdm
from transformers.utils import logging
from transformers import GenerationConfig
import json
from peft import LoraConfig, get_peft_model, TaskType,PeftModel
from .utils import get_gold_passages, response_gen, extract_ans_t5, write_json

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


output_folder = "data/"
output_file = f"./{output_folder}hotpotqa_generated_result.json"
ds = load_dataset("hotpot_qa", "distractor")
split = "validation"

use_finetune = False
if use_finetune:
    model = PeftModel.from_pretrained(model, "flan_t5_xl_question_answering/checkpoint-600")


def main():

    correct = 0
    
    done_ids = set()
    
    # create output file if it doesnt exist
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    # open file where we are saving data
    if os.path.exists(output_file):
    
        json_data = []
        with open(output_file, 'r') as f:
            for line in f:
                sample = json.loads(line)
                done_ids.add(sample["id"])
                json_data.append(sample)
                if sample["flag"]==True:
                    correct += 1
        
        start_point = len(done_ids)
        print(f"The generated samples reloaded, the number of sample is {start_point}. The accuracy is {correct/start_point}.")
    else:
        json_data = []
    
    for i, data in enumerate(tqdm(ds[split])):

        if data["id"] in done_ids:
            continue
   
        gold_passages = get_gold_passages(data)

        passage = "\n".join(gold_passages.values())
        input_prompt = zero_shot_prompt_flant5.format(data["question"], passage)

        inputs = tokenizer(input_prompt,return_tensors='pt')

        input_ids = inputs['input_ids'].to("cuda")
        
        output = response_gen(model, tokenizer, input_ids, generation_config)

        print("Answer responded:", output)
        print("Ground truth:", data["answer"])

        check_response = extract_ans_t5(output, data["answer"]) 
        correct += int(check_response)
        print(f"Current accuracy: {correct/(i+1)}")

        ans_data = {
            "id": data["id"],
            "response": output,
            "flag": check_response
        }

        # add ans_data to file
        write_json(ans_data, output_file)

        done_ids.add(data["id"])
    


if __name__ == '__main__':
    main()
