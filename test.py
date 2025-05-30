# test_model.py

import torch
from model import GPT, GPTConfig
from utils import get_encode_decode

VOCAB_SIZE    = len("0123456789/=*\n")
BLOCK_SIZE    = 128
N_LAYER       = 6
N_HEAD        = 6
N_EMBD        = 384
DROPOUT       = 0.1
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

encode, decode = get_encode_decode()

cfg   = GPTConfig(VOCAB_SIZE, BLOCK_SIZE, N_LAYER, N_HEAD, N_EMBD, DROPOUT)
model = GPT(cfg).to(DEVICE)
ckpt  = torch.load("ckpt_25000.pt", map_location=DEVICE)   # 修改为你的模型文件名
model.load_state_dict(ckpt)
model.eval()

test_prompts = [
    "436/342=",
    "399/457=",
    "245/331=",
    "691/390=",
    "96/16=",
    "43/7=",
    "75/67=",
    "35/45=",
    "8/4=",
    "9/6=",
    "9/3=",
    "5/6=",
]

print("=== result ===")
with torch.no_grad():
    for p in test_prompts:
        ids = torch.tensor([encode(p)], dtype=torch.long, device=DEVICE)
        out = model.generate(ids, max_new_tokens=10, temperature=1.0, top_k=1)
        result = decode(out[0].tolist())[len(p):].split("\n", 1)[0]
        print(f"{p}{result}")