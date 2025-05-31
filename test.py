# test_model.py

import torch
from model import GPT, GPTConfig
from utils import get_encode_decode

CHARS        = "0123456789+-*/=." + "\n"
VOCAB_SIZE    = len(CHARS)
BLOCK_SIZE    = 128
N_LAYER       = 6
N_HEAD        = 6
N_EMBD        = 384
DROPOUT       = 0.1
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

encode, decode = get_encode_decode(CHARS)

cfg   = GPTConfig(VOCAB_SIZE, BLOCK_SIZE, N_LAYER, N_HEAD, N_EMBD, DROPOUT)
model = GPT(cfg).to(DEVICE)
ckpt  = torch.load("model_best.pt", map_location=DEVICE)   # 修改为你的模型文件名
model.load_state_dict(ckpt)
model.eval()

test_prompts = [
    "756/174==",
    "756*174==",
    "756+174==",
    "756-174==",
    "96/16=",
    "96*16=",
    "96+16=",
    "96-16=",
    "43/7=",
    "43*7=",
    "43+7=",
    "43-7=",
    "5967/2556=",
    "5967*2556=",
    "5967+2556=",
    "5967-2556=",
    "8/4=",
    "8*4=",
    "8+4=",
    "8-4=",
]

print("=== result ===")
with torch.no_grad():
    for p in test_prompts:
        ids = torch.tensor([encode(p)], dtype=torch.long, device=DEVICE)
        out = model.generate(ids, max_new_tokens=10, temperature=0.1, top_k=1)
        result = decode(out[0].tolist())[len(p):].split("\n", 1)[0]
        print(f"{p}{result}")