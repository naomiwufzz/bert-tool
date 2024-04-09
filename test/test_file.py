import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer
from transformers import AutoTokenizer
model_file = "./hugging_face_model/bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_file)
tokenizer = AutoTokenizer.from_pretrained(model_file)
bert_model = BertModel.from_pretrained(model_file)
# tokenize
tokenizer.encode("Hello, my dog is cute", return_tensors="pt")
# tokenize for bert input
encode_res = tokenizer.encode_plus("今天我很开心", return_tensors="pt", add_special_tokens=True)
input_ids, attention_mask, token_type_ids = encode_res.get("input_ids"), encode_res.get("attention_mask"), encode_res.get("token_type_ids")
res = bert_model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
print(res)