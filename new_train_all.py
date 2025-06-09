import torch
import random
import math
import argparse
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from model import GPT, GPTConfig
from loguru import logger
from utils import *

def parse_args():
    p = argparse.ArgumentParser(description="Mini GPT Arithmetic with Curriculum, Replay, Warmup+Cosine, and Grad Clipping")
    p.add_argument("--max-iters",    type=int,   default=60000)
    p.add_argument("--eval-interval", type=int,   default=5000)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--block-size",   type=int,   default=128)
    p.add_argument("--n-layer",      type=int,   default=8)
    p.add_argument("--n-head",       type=int,   default=8)
    p.add_argument("--n-embd",       type=int,   default=512)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--resume",       action="store_true")
    p.add_argument("--resume-add",   type=str,   default=None)
    p.add_argument("--save-interval",type=int,   default=5000)
    return p.parse_args()

def build_lines_for_stage(stage: int):
    lines = []
    if stage >= 1:
        d = 1
        lines += generate_add_data(30000, d)
        lines += generate_minus_data(30000, d)
        lines += generate_multi_data(40000, d)
        lines += generate_division_data(40000, d)
    if stage >= 2:
        d = 2
        lines += generate_add_data(15000, d)
        lines += generate_minus_data(15000, d)
        lines += generate_multi_data(100000, d)
        lines += generate_division_data(100000, d)
    if stage >= 3:
        d = 3
        lines += generate_add_data(15000, d)
        lines += generate_minus_data(15000, d)
        lines += generate_multi_data(200000, d)
        lines += generate_division_data(200000, d)
    random.shuffle(lines)
    return lines

def build_loader_from_lines(lines: list[str], block_size: int, batch_size: int):
    full_text = "".join(lines)
    encode, _ = get_encode_decode(CHARS)
    data_ids  = encode(full_text)
    data      = torch.tensor(data_ids, dtype=torch.long)

    stride  = block_size + 1
    num_seq = data.size(0) // stride
    seqs    = data[: num_seq * stride].view(num_seq, stride)
    x_all   = seqs[:, :block_size]
    y_all   = seqs[:, 1:]

    ds = TensorDataset(x_all, y_all)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=3,
        pin_memory=True
    )

if __name__ == "__main__":
    args = parse_args()

    logger.add("all_Train.log")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(DEVICE)
    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    CHARS         = "0123456789+-*/=." + "\n"
    VOCAB_SIZE    = len(CHARS)
    BLOCK_SIZE    = args.block_size
    BATCH_SIZE    = args.batch_size
    LEARNING_RATE = args.lr
    MAX_ITERS     = args.max_iters
    EVAL_INT      = args.eval_interval
    SAVE_INT      = args.save_interval

    encode, decode = get_encode_decode(CHARS)

    config = GPTConfig(VOCAB_SIZE, BLOCK_SIZE, args.n_layer, args.n_head, args.n_embd, args.dropout)
    model  = GPT(config).to(DEVICE)
    opt    = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()

    warmup_steps = max(1, MAX_ITERS // 10)
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(warmup_steps)
        progress = float(current_step - warmup_steps) / float(MAX_ITERS - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = LambdaLR(opt, lr_lambda)

    if args.resume:
        ckpt = torch.load(args.resume_add, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        opt.load_state_dict(ckpt["optim_state_dict"])
        start_iter = ckpt.get("iter", 1) + 1
    else:
        start_iter = 1

    test_lines = (
        generate_add_data(1000, 3) +
        generate_minus_data(1000, 3) +
        generate_multi_data(1000, 3) +
        generate_division_data(1000, 3) +
        generate_add_data(100, 2) +
        generate_minus_data(100, 2) +
        generate_multi_data(100, 2) +
        generate_division_data(100, 2) +
        generate_add_data(30, 1) +
        generate_minus_data(30, 1) +
        generate_multi_data(30, 1) +
        generate_division_data(30, 1)
    )
    random.shuffle(test_lines)
    test_prompts, test_answers = generate_prompts(test_lines)

    small_test_idx     = random.sample(range(len(test_prompts)), 200)
    small_test_prompts = [test_prompts[i] for i in small_test_idx]
    small_test_answers = [test_answers[i] for i in small_test_idx]

    stage1_lines = build_lines_for_stage(stage=1)
    val_lines    = random.sample(stage1_lines, 600)
    val_prompts, val_answers = generate_prompts(val_lines)

    small_val_idx     = random.sample(range(len(val_prompts)), 100)
    small_val_prompts = [val_prompts[i] for i in small_val_idx]
    small_val_answers = [val_answers[i] for i in small_val_idx]

    one_third  = MAX_ITERS / 6
    two_thirds = 2 * MAX_ITERS / 6

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

    loader1 = build_loader_from_lines(stage1_lines, BLOCK_SIZE, BATCH_SIZE)
    model.train()
    it = start_iter
    loader1_iter = iter(loader1)

    print(f"\n======== Stage 1: step to {STAGE1_IT} , only 1 digit ========\n")
    while it <= STAGE1_IT:
        try:
            x, y = next(loader1_iter)
        except StopIteration:
            loader1_iter = iter(loader1)
            x, y = next(loader1_iter)

        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        with autocast(DEVICE):
            logits, loss = model(x, y)

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        if it % 100 == 0:
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"[Stage1 iter {it}/{STAGE1_IT}] loss = {loss.item():.4f}  lr={current_lr:.2e}")

        if it % EVAL_INT == 0:
            model.eval()
            train_acc = evaluate(model, encode, decode, small_val_prompts, small_val_answers, device=DEVICE)
            valid_acc = evaluate(model, encode, decode, small_test_prompts, small_test_answers, device=DEVICE)
            logger.info(f" → train_acc={train_acc*100:.2f}%  valid_acc={valid_acc*100:.2f}%")
            model.train()

        if it % SAVE_INT == 0:
            torch.save({
                "iter": it,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": opt.state_dict()
            }, f"all_{it}.pt")

        it += 1

    print(f"\n======== Stage 2: step to {STAGE2_IT} , Replay 10% Stage1 ========\n")
    stage2_main = build_lines_for_stage(stage=2)
    replay_cnt  = int(len(stage2_main) * 0.10)
    replay_idx  = random.sample(range(len(stage1_lines)), replay_cnt)
    replay_data = [stage1_lines[i] for i in replay_idx]

    combined_stage2 = stage2_main + replay_data
    random.shuffle(combined_stage2)
    loader2        = build_loader_from_lines(combined_stage2, BLOCK_SIZE, BATCH_SIZE)
    loader2_iter   = iter(loader2)

    while it <= STAGE2_IT:
        try:
            x, y = next(loader2_iter)
        except StopIteration:
            loader2_iter = iter(loader2)
            x, y = next(loader2_iter)

        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        with autocast(DEVICE):
            logits, loss = model(x, y)

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        if it % 100 == 0:
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"[Stage2 iter {it}/{STAGE2_IT}] loss = {loss.item():.4f}  lr={current_lr:.2e}")

        if it % EVAL_INT == 0:
            model.eval()
            train_acc = evaluate(model, encode, decode, small_val_prompts, small_val_answers, device=DEVICE)
            valid_acc = evaluate(model, encode, decode, small_test_prompts, small_test_answers, device=DEVICE)
            logger.info(f" → train_acc={train_acc*100:.2f}%  valid_acc={valid_acc*100:.2f}%")
            model.train()

        if it % SAVE_INT == 0:
            torch.save({
                "iter": it,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": opt.state_dict()
            }, f"all_{it}.pt")

        it += 1

    print(f"\n======== Stage 3: step to {STAGE3_IT}, Replay 10%(Stage1+Stage2) ========\n")
    stage3_main = build_lines_for_stage(stage=3)
    stage2_lines = stage2_main
    replay_pool_3 = stage1_lines + stage2_lines
    replay_cnt_3  = int(len(stage3_main) * 0.10)
    replay_idx_3  = random.sample(range(len(replay_pool_3)), replay_cnt_3)
    replay_data_3 = [replay_pool_3[i] for i in replay_idx_3]

    combined_stage3 = stage3_main + replay_data_3
    random.shuffle(combined_stage3)
    loader3      = build_loader_from_lines(combined_stage3, BLOCK_SIZE, BATCH_SIZE)
    loader3_iter = iter(loader3)

    while it <= STAGE3_IT:
        try:
            x, y = next(loader3_iter)
        except StopIteration:
            loader3_iter = iter(loader3)
            x, y = next(loader3_iter)

        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        with autocast(DEVICE):
            logits, loss = model(x, y)

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        if it % 100 == 0:
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"[Stage3 iter {it}/{STAGE3_IT}] loss = {loss.item():.4f}  lr={current_lr:.2e}")

        if it % EVAL_INT == 0:
            model.eval()
            train_acc = evaluate(model, encode, decode, small_val_prompts, small_val_answers, device=DEVICE)
            valid_acc = evaluate(model, encode, decode, small_test_prompts, small_test_answers, device=DEVICE)
            logger.info(f" → train_acc={train_acc*100:.2f}%  valid_acc={valid_acc*100:.2f}%")
            model.train()

        if it % SAVE_INT == 0:
            torch.save({
                "iter": it,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": opt.state_dict()
            }, f"all_{it}.pt")

        it += 1


    model.eval()
    acc_final = evaluate(model, encode, decode, test_prompts, test_answers, device=DEVICE)
    print(f"\nFinal eval accuracy = {acc_final*100:.2f}%")
