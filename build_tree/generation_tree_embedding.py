import torch
import json
import clip
import torch.nn.functional as F
from collections import OrderedDict


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16" , device=device, jit=False, download_root='./')
model.eval()
model.requires_grad_(False)

with open('/workspace/hico_obj_tree_32.json', 'r') as f:
    tree_dict = json.load(f)

tokenizer=clip.tokenize
description_encodings = OrderedDict()
for k, _ in tree_dict.items():
    description_encodings[k] = [] 
for k, v in tree_dict.items():
    for x in v:
        tokens = tokenizer(x).to(device)
        description_encodings[k].append(F.normalize(model.encode_text(tokens)).tolist())

with open('/workspace/hico_obj_decriptions_emb_16.json', 'w') as f:
    json.dump(description_encodings, f)