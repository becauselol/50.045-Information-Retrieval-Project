import torch
import re

def get_gold_passages(dataset, split="validation"):
    dataset = dataset[split]

    return dataset

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
