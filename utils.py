import random
import torch
from torch import inference_mode

def get_encode_decode(CHARS):
    stoi = { ch:i for i,ch in enumerate(CHARS) }
    itos = { i:ch for i,ch in enumerate(CHARS) }

    def encode(s: str) -> list[int]:
        return [stoi[c] for c in s if c in stoi]

    def decode(l: list[int]) -> str:
        return ''.join(itos[i] for i in l)

    return encode, decode

def generate_division_data(N: int, max_digits: int) -> list[str]:
    data = []
    for _ in range(N):
        a = random.randint(10**(max_digits-1), 10**max_digits - 1)
        b = random.randint(10**(max_digits-1), 10**max_digits - 1)
        q = a // b
        r = a % b
        data.append(f"{a}/{b}={q}.{r}/{b}\n")
    return data

def generate_multi_data(N: int, max_digits: int) -> list[str]:
    data = []
    for _ in range(N):
        a = random.randint(10**(max_digits-1), 10**max_digits - 1)
        b = random.randint(10**(max_digits-1), 10**max_digits - 1)
        c=a*b
        data.append(f"{a}*{b}={c}\n")
    return data

def generate_add_data(N: int, max_digits: int) -> list[str]:
    data = []
    for _ in range(N):
        a = random.randint(10**(max_digits-1), 10**max_digits - 1)
        b = random.randint(10**(max_digits-1), 10**max_digits - 1)
        c=a+b
        data.append(f"{a}+{b}={c}\n")
    return data

def generate_minus_data(N: int, max_digits: int) -> list[str]:
    data = []
    for _ in range(N):
        a = random.randint(10**(max_digits-1), 10**max_digits - 1)
        b = random.randint(10**(max_digits-1), 10**max_digits - 1)
        c=a-b
        data.append(f"{a}-{b}={c}\n")
    return data

def generate_all_data1(N: int, max_digits: int) -> list[str]:
    data = []
    for _ in range(N):
        a = random.randint(10**(max_digits-1), 10**max_digits - 1)
        b = random.randint(10**(max_digits-1), 10**max_digits - 1)
        q = a // b
        r = a % b
        data.append(f"{a}/{b}={q}.{r}/{b}\n")
        a_mu = a*b
        data.append(f"{a}*{b}={a_mu}\n")
        a_a = a+b
        data.append(f"{a}+{b}={a_a}\n")
        a_m = a-b
        data.append(f"{a}-{b}={a_m}\n")
    return data

def generate_all_data2(N: int, max_digits: int) -> list[str]:
    data = []
    for _ in range(N):
        a = random.randint(1, 10**max_digits - 1)
        b = random.randint(1, 10**max_digits - 1)
        q = a // b
        r = a % b
        data.append(f"{a}/{b}={q}.{r}/{b}\n")
        a_mu = a*b
        data.append(f"{a}*{b}={a_mu}\n")
        a_a = a+b
        data.append(f"{a}+{b}={a_a}\n")
        a_m = a-b
        data.append(f"{a}-{b}={a_m}\n")
    return data

def generate_prompts(data: list[str]) -> tuple[list[str], list[str]]:
    prompts = []
    answers = []
    for line in data:
        expr, frac = line.strip().split('=')
        prompts.append(expr + '=')
        answers.append(frac)
    return prompts, answers

@inference_mode()
def evaluate(model, encode, decode, prompts, answers, device='cuda'):
    model.eval()
    correct = 0
    for prompt, true_ans in zip(prompts, answers):
        ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        out = model.generate(
            ids,
            max_new_tokens=len(true_ans) + 1,
            temperature=0.1,
            top_k=1
        )
        pred = decode(out[0].tolist())[len(prompt):].split('\n',1)[0]
        if pred == true_ans:
            correct += 1
    return correct / len(prompts)
