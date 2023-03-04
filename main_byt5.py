from transformers import AutoTokenizer
from transformers import T5ForConditionalGeneration
import torch

model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")
num_special_tokens = 3
tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

input_ids_prompt = "the past tense of walk is "
input_ids = tokenizer(input_ids_prompt).input_ids
input_ids = torch.tensor([input_ids + [258]])

output_ids = model.generate(input_ids, max_length=100)[0].tolist()
output_ids_list = []
start_token = 0
sentinel_token = 258
while sentinel_token in output_ids:
    end_token = output_ids.index(sentinel_token)
    output_ids_list.append(output_ids[start_token:end_token])
    start_token = end_token
    sentinel_token -= 1

output_ids_list.append(output_ids[start_token:])
output_string = tokenizer.batch_decode(output_ids_list)
print(output_string)

