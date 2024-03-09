from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch

model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")

mps_device = torch.device('mps')
model.to(mps_device)
input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("mps")
outputs = model.generate(input_ids,max_new_tokens=50)
#list of unusable tokens
list_tokens = ['<pad>','<unk>','<s>','</s>']
outputs = tokenizer.decode(outputs[0])
#remove the unusable tokens
for token in list_tokens:
    outputs = outputs.replace(token,'')

print(outputs)
