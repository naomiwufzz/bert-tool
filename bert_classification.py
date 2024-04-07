import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer
from transformers import AutoTokenizer
class ClassificationModel(nn.Module):
    def __init__(self, bert_base_model_dir, label_size, loss):
        super(ClassificationModel, self).__init__()
        self.label_size = label_size
        self.loss = loss
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_base_model_dir)
        self.bert_model = BertModel.from_pretrained(bert_base_model_dir)
        cls_layer_input_size = self.bert_model.config.hidden_size
        self.cls_layer = nn.Linear(cls_layer_input_size, self.label_size)


    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_out = self.bert_model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
