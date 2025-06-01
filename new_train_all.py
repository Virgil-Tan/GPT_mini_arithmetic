import torch, random, argparse
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import autocast, GradScaler
from model import GPT, GPTConfig
from loguru import logger
from utils import *

def parse_args():
    p = argparse.ArgumentParser(description="Mini GPT Arithmetic")
    p.add_argument("--max-iters",   type=int,   default=10000)
    p.add_argument("--eval-interval",type=int,  default=1000)
    p.add_argument("--batch-size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--block-size",  type=int,   default=128)
    p.add_argument("--n-layer",     type=int,   default=6)
    p.add_argument("--n-head",      type=int,   default=6)
    p.add_argument("--n-embd",      type=int,   default=384)
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--resume",      action="store_true")
    p.add_argument("--resume-add",  type=str,   default=None)
    p.add_argument("--save-interval",type=int,  default=5000)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    logger.add("all_Train.log")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark       = True
    torch.backends.cuda.matmul.allow_tf32= True
    torch.backends.cudnn.allow_tf32      = True

    CHARS        = "0123456789+-*/=." + "\n"
    VOCAB_SIZE   = len(CHARS)
    BLOCK_SIZE   = args.block_size
    BATCH_SIZE   = args.batch_size
    LEARNING_RATE= args.lr
    MAX_ITERS    = args.max_iters
    EVAL_INT     = args.eval_interval
    SAVE_INT     = args.save_interval

    encode, decode = get_encode_decode(CHARS)
    lines = (
        generate_all_data2(100000,4)
      + generate_division_data(10000,4)
      + generate_multi_data(10000,4)

      + generate_all_data2(60000,3)
      + generate_division_data(10000,3)
      + generate_multi_data(10000,3)

      + generate_all_data2(20000,2)
      + generate_all_data2(20000,1)
    )
    random.shuffle(lines)
    full_text = "".join(lines)
    data_ids   = encode(full_text)
    data       = torch.tensor(data_ids, dtype=torch.long)

    stride   = BLOCK_SIZE + 1
    num_seq  = data.size(0) // stride
    seqs     = data[: num_seq*stride].view(num_seq, stride)
    x_all    = seqs[:, :BLOCK_SIZE]
    y_all    = seqs[:, 1:]

    dataset  = TensorDataset(x_all, y_all)
    loader   = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    test_lines = (
        generate_all_data2(1400,4)
    + generate_all_data2(1000,3)
    + generate_all_data2(100,2)
    + generate_all_data2(20,1)
    )
    random.shuffle(test_lines)
    test_prompts, test_answers = generate_prompts(test_lines)

    small_test_idx     = random.sample(range(len(test_prompts)), 200)
    small_test_prompts = [test_prompts[i] for i in small_test_idx]
    small_test_answers = [test_answers[i] for i in small_test_idx]

    val_lines = random.sample(lines, 600)
    val_prompts, val_answers = generate_prompts(val_lines)

    small_val_idx     = random.sample(range(len(val_prompts)), 100)
    small_val_prompts = [val_prompts[i] for i in small_val_idx]
    small_val_answers = [val_answers[i] for i in small_val_idx]

    config = GPTConfig(VOCAB_SIZE, BLOCK_SIZE, args.n_layer, args.n_head, args.n_embd, args.dropout)
    model  = GPT(config).to(DEVICE)
    opt    = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()

    if args.resume:
        ckpt = torch.load(args.resume_add, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        opt.load_state_dict(ckpt["optim_state_dict"])
        start_iter = ckpt.get("iter", 1) + 1
    else:
        start_iter = 1


    model.train()
    it = start_iter
    while it <= MAX_ITERS:
        for x, y in loader:
            if it > MAX_ITERS:
                break
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            with autocast(DEVICE):
                logits, loss = model(x, y)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if it % 100 == 0:
                logger.info(f"[iter {it:5d}/{MAX_ITERS}] loss = {loss.item():.4f}")

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
            if it > MAX_ITERS:
                break

    acc_final = evaluate(model, encode, decode, test_prompts, test_answers, device=DEVICE)
    print(f"Final eval accuracy = {acc_final*100:.2f}%")
