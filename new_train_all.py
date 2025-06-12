import os
import random
import torch
import math
import argparse
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from model import GPT, GPTConfig
from loguru import logger
from utils import get_encode_decode, generate_prompts, evaluate

def load_lines(filepath: str) -> list[str]:
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    return [line + '\n' for line in lines]

def build_loader_from_lines(lines: list[str], block_size: int, batch_size: int):
    full_text = "".join(lines)
    encode, _ = get_encode_decode(CHARS)
    data_ids = encode(full_text)
    data = torch.tensor(data_ids, dtype=torch.long)

    stride = block_size + 1
    num_seq = data.size(0) // stride
    seqs = data[: num_seq * stride].view(num_seq, stride)
    x_all = seqs[:, :block_size]
    y_all = seqs[:, 1:]

    ds = TensorDataset(x_all, y_all)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=3,
        pin_memory=True
    )

def parse_args():
    p = argparse.ArgumentParser(description="Mini GPT Arithmetic with external data files")
    p.add_argument("--max-iters",    type=int,   default=100000)
    p.add_argument("--eval-interval", type=int,   default=100)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--block-size",   type=int,   default=128)
    p.add_argument("--n-layer",      type=int,   default=6)
    p.add_argument("--n-head",       type=int,   default=8)
    p.add_argument("--n-embd",       type=int,   default=512)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--resume",       action="store_true")
    p.add_argument("--resume-add",   type=str,   default=None)
    p.add_argument("--save-interval",type=int,   default=10000)
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    logger.add("all_Train.log")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(DEVICE)

    CHARS = "0123456789+-*/=.\n"
    VOCAB_SIZE = len(CHARS)
    BLOCK_SIZE = args.block_size
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    MAX_ITERS = args.max_iters
    EVAL_INT = args.eval_interval
    SAVE_INT = args.save_interval
    best_model = 93

    # 文件映射
    train_files = {1: 'train_data_1.txt', 2: 'train_data_2.txt', 3: 'train_data_3.txt'}
    valid_files = {1: 'valid_data_1.txt', 2: 'valid_data_2.txt', 3: 'valid_data_3.txt'}
    test_files  = {1: 'test_data_1.txt',  2: 'test_data_2.txt',  3: 'test_data_3.txt'}

    # 准备模型和优化器
    encode, decode = get_encode_decode(CHARS)
    config = GPTConfig(VOCAB_SIZE, BLOCK_SIZE, args.n_layer, args.n_head, args.n_embd, args.dropout)
    model = GPT(config).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()
    warmup_steps = max(1, MAX_ITERS // 10)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (MAX_ITERS - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = LambdaLR(opt, lr_lambda, last_epoch=-1)

    # 恢复训练
    if args.resume:
        ckpt = torch.load(args.resume_add, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        opt.load_state_dict(ckpt['optim_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_iter = ckpt.get('iter', 1) + 1
    else:
        start_iter = 1

    one_third  = MAX_ITERS / 30
    two_thirds = 2 * MAX_ITERS / 30

    STAGE1_IT = int(one_third // 100) * 100
    STAGE2_IT = int(two_thirds // 100) * 100
    STAGE3_IT = MAX_ITERS

    if STAGE1_IT == 0:
        STAGE1_IT = 100
    if STAGE2_IT <= STAGE1_IT:
        STAGE2_IT = STAGE1_IT + 100
    if STAGE2_IT > MAX_ITERS:
        STAGE2_IT = STAGE1_IT + 100
    if STAGE3_IT <= STAGE2_IT:
        STAGE3_IT = STAGE2_IT + 100


    # --- Stage 1 ---
    stage = 1
    train_lines_1 = load_lines(f"data/{train_files[stage]}")
    val_lines_1 = load_lines(f"data/{valid_files[stage]}")
    test_lines_1 = load_lines(f"data/{test_files[stage]}")
    val_prompts, val_answers = generate_prompts(val_lines_1)
    test_prompts, test_answers = generate_prompts(test_lines_1)

    loader1 = build_loader_from_lines(train_lines_1, BLOCK_SIZE, BATCH_SIZE)
    model.train()
    it = start_iter
    loader1_iter = iter(loader1)
    print(f"\n======== Stage 1 (1-digit): step to {STAGE1_IT} ========\n")
    while it <= STAGE1_IT:
        try:
            x, y = next(loader1_iter)
        except StopIteration:
            loader1_iter = iter(loader1)
            x, y = next(loader1_iter)
        x, y = x.to(DEVICE), y.to(DEVICE)
        with autocast(DEVICE):
            logits, loss = model(x, y)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        if it % EVAL_INT == 0:
            model.eval()
            val_acc = evaluate(model, encode, decode, val_prompts, val_answers, device=DEVICE)
            logger.info(f" → val_acc={val_acc*100:.2f}%")
            model.train()
        it += 1

    # --- Stage 2 ---
    stage = 2
    stage2_main = load_lines(f"data/{train_files[stage]}")
    train_lines_2 = stage2_main + train_lines_1
    random.shuffle(train_lines_2)

    val_lines_2 = load_lines(f"data/{valid_files[stage]}")+val_lines_1
    test_lines_2 = load_lines(f"data/{test_files[stage]}")+test_lines_1
    val_prompts, val_answers = generate_prompts(val_lines_2)
    test_prompts, test_answers = generate_prompts(test_lines_2)

    loader2 = build_loader_from_lines(train_lines_2, BLOCK_SIZE, BATCH_SIZE)
    loader2_iter = iter(loader2)
    print(f"\n======== Stage 2 (2-digit): step to {STAGE2_IT} ========\n")
    while it <= STAGE2_IT:
        try:
            x, y = next(loader2_iter)
        except StopIteration:
            loader2_iter = iter(loader2)
            x, y = next(loader2_iter)
        x, y = x.to(DEVICE), y.to(DEVICE)
        with autocast(DEVICE):
            logits, loss = model(x, y)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        if it % EVAL_INT == 0:
            model.eval()
            val_acc = evaluate(model, encode, decode, val_prompts, val_answers, device=DEVICE)
            logger.info(f" → val_acc={val_acc*100:.2f}%")
            model.train()
        it += 1

    # --- Stage 3 ---
    stage = 3
    stage3_main = load_lines(f"data/{train_files[stage]}")
    mix_data = load_lines(f"data/mixed_cross_digit.txt")
    stage3_main  = random.sample(stage3_main,  min(60000, len(stage3_main)))
    train_lines_3 = stage3_main + stage2_main + train_lines_1
    # train_lines_3 = mix_data
    random.shuffle(train_lines_3)

    val_lines_3 = load_lines(f"data/{valid_files[stage]}")+val_lines_2
    test_lines_3 = load_lines(f"data/{test_files[stage]}")+test_lines_2
    
    n = 100
    val_lines_3  = random.sample(val_lines_3,  min(n, len(val_lines_3)))
    test_lines_3 = random.sample(test_lines_3, min(n, len(test_lines_3)))

    val_prompts, val_answers = generate_prompts(val_lines_3)
    test_prompts, test_answers = generate_prompts(test_lines_3)


    loader3 = build_loader_from_lines(train_lines_3, BLOCK_SIZE, BATCH_SIZE)
    loader3_iter = iter(loader3)
    print(f"\n======== Stage 3 (3-digit): step to {STAGE3_IT} ========\n")
    while it <= STAGE3_IT:
        try:
            x, y = next(loader3_iter)
        except StopIteration:
            loader3_iter = iter(loader3)
            x, y = next(loader3_iter)
        x, y = x.to(DEVICE), y.to(DEVICE)
        with autocast(DEVICE):
            logits, loss = model(x, y)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        if it % 100 == 0:
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"[Stage3 iter {it}/{STAGE3_IT}] loss = {loss.item():.4f}  lr={current_lr:.2e}")

        if it % EVAL_INT == 0:
            model.eval()
            val_acc = evaluate(model, encode, decode, val_prompts, val_answers, device=DEVICE)
            val_acc_3 = round(val_acc * 100, 2)
            logger.info(f" → val_acc={val_acc_3:.2f}% ")
            model.train()
            
            if val_acc_3 > best_model:
                torch.save({
                    "iter": it,
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": opt.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict()
                }, f"best_{int(val_acc_3)}.pt")
                best_model = val_acc_3


        if it % SAVE_INT == 0:
            torch.save({
                "iter": it,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": opt.state_dict(),
                "scheduler_state_dict": scheduler.state_dict()
            }, f"all_{it}_{val_acc}.pt")

        it += 1

    # 最终评估
    model.eval()
    final_acc = evaluate(model, encode, decode, test_prompts, test_answers, device=DEVICE)
    print(f"\nFinal eval accuracy = {final_acc*100:.2f}%")

