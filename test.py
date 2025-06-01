import torch
import random
from model import GPT, GPTConfig
from utils import (
    get_encode_decode,
    generate_add_data,
    generate_minus_data,
    generate_multi_data,
    generate_division_data,
    generate_prompts
)
CHARS     = "0123456789+-*/=." + "\n"
VOCAB_SIZE= len(CHARS)
BLOCK_SIZE= 128
N_LAYER   = 6
N_HEAD    = 6
N_EMBD    = 384
DROPOUT   = 0.1

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLES_PER_OP = 100

encode, decode = get_encode_decode(CHARS)

cfg   = GPTConfig(VOCAB_SIZE, BLOCK_SIZE, N_LAYER, N_HEAD, N_EMBD, DROPOUT)
model = GPT(cfg).to(DEVICE)
ckpt  = torch.load("model_best.pt", map_location=DEVICE)
model.load_state_dict(ckpt)
model.eval()

@torch.no_grad()
def count_correct(pred_model, encode_fn, decode_fn, prompts, answers, device="cuda"):
    correct = 0
    for prompt, true_ans in zip(prompts, answers):
        ids = torch.tensor([encode_fn(prompt)], dtype=torch.long, device=device)
        out = pred_model.generate(
            ids,
            max_new_tokens = len(true_ans) + 1,
            temperature=0.1,
            top_k=1
        )
        pred_str = decode_fn(out[0].tolist())[len(prompt):].split("\n", 1)[0]
        if pred_str == true_ans:
            correct += 1
    return correct

stats = {}

for max_digits in [1, 2, 3, 4]:
    stats[max_digits] = {}
    add_data = generate_add_data(SAMPLES_PER_OP, max_digits)
    add_prompts, add_answers = generate_prompts(add_data)
    num_correct_add = count_correct(model, encode, decode, add_prompts, add_answers, device=DEVICE)
    stats[max_digits]["加"] = (num_correct_add, SAMPLES_PER_OP)

    minus_data = generate_minus_data(SAMPLES_PER_OP, max_digits)
    minus_prompts, minus_answers = generate_prompts(minus_data)
    num_correct_minus = count_correct(model, encode, decode, minus_prompts, minus_answers, device=DEVICE)
    stats[max_digits]["减"] = (num_correct_minus, SAMPLES_PER_OP)

    multi_data = generate_multi_data(SAMPLES_PER_OP, max_digits)
    multi_prompts, multi_answers = generate_prompts(multi_data)
    num_correct_multi = count_correct(model, encode, decode, multi_prompts, multi_answers, device=DEVICE)
    stats[max_digits]["乘"] = (num_correct_multi, SAMPLES_PER_OP)

    div_data = generate_division_data(SAMPLES_PER_OP, max_digits)
    div_prompts, div_answers = generate_prompts(div_data)
    num_correct_div = count_correct(model, encode, decode, div_prompts, div_answers, device=DEVICE)
    stats[max_digits]["除"] = (num_correct_div, SAMPLES_PER_OP)

print("\n==== 分位数 & 分运算 错误统计 ====\n")
error_count_by_digits = {}
error_count_by_op     = {"加": 0, "减": 0, "乘": 0, "除": 0}

for digits, op_dict in stats.items():
    total_errors_this_digit = 0
    print(f"--- {digits} 位数 测试 ({SAMPLES_PER_OP} 条/运算) ---")
    for op_name, (correct_cnt, total_cnt) in op_dict.items():
        errors = total_cnt - correct_cnt
        error_rate = errors / total_cnt * 100
        total_errors_this_digit += errors
        error_count_by_op[op_name] += errors
        print(f"  {op_name} 法 ➔ 正确 {correct_cnt}/{total_cnt}，错误 {errors}/{total_cnt} （错误率 {error_rate:.2f}%）")
    error_count_by_digits[digits] = total_errors_this_digit
    print(f"  → {digits} 位总错误: {total_errors_this_digit}/{4* SAMPLES_PER_OP} （错误率 {(total_errors_this_digit/(4*SAMPLES_PER_OP))*100:.2f}%）\n")

worst_digit = max(error_count_by_digits.items(), key=lambda x: x[1])
worst_op    = max(error_count_by_op.items(), key=lambda x: x[1])

print("==== 综合统计 ====")
print(f"错误最多的位数：{worst_digit[0]} 位，共错 {worst_digit[1]}/{4*SAMPLES_PER_OP} 次")
print(f"错误最多的运算：{worst_op[0]} 法，共错 {worst_op[1]}/{4*SAMPLES_PER_OP}*4 次\n")
