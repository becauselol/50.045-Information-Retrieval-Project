import torch
import re
import json

def get_gold_passages(data_dict):
    passages = dict(zip(data_dict["context"]["title"], data_dict["context"]["sentences"]))

    gold_passages = {}
    for gold_titles in set(data_dict["supporting_facts"]["title"]):
        gold_passages[gold_titles] = " ".join(passages[gold_titles])
    
    return gold_passages

def response_gen(model, tokenizer, input_ids, generation_config):
  with torch.no_grad():
      generation_output = model.generate(
          input_ids=input_ids,
          generation_config=generation_config
      )
  # s = generation_output.sequences[0][len(input_ids[0]):]
  output = tokenizer.decode(generation_output[0])
  return output

def extract_ans_t5(ans, ground_truth):
    ground_truth = ground_truth.lower()
    ans = ans.lower()

    # Remove special tokens and strip white space
    clean_ans = re.sub("([\<]).*?([\>])", "", ans).strip()

    return clean_ans == ground_truth

def write_json(data, path):
    f = open(path, mode='a', encoding='utf-8')
    json.dump(data, f, ensure_ascii=False)
    f.write('\n')
    f.close()
