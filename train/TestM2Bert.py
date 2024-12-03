import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from M2Bert import M2BertReranker


model = M2BertReranker()
model.load_state_dict(torch.load("./trained_model/M2BertStateDict.pth"))
print(model)