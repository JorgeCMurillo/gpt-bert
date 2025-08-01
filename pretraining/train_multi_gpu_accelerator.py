#!/usr/bin/env python

import argparse
import copy
import json
import math
from itertools import count
from pathlib import Path
from statistics import mean

import torch
import torch.nn as nn
import wandb
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed
from dataset import CausalDataset, MaskedDataset, ValidationDataset
from lamb import Lamb
from model_extra import Bert
from model_logging import ModelLogger
from utils import cosine_schedule_with_warmup_cooldown


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", default="../data/babycosmofine_10M_tokenized.bin", type=Path, help="Path to the training data.")
    parser.add_argument("--valid_path", default="../data/babycosmofine_10M_tokenized.bin", type=Path, help="Path to the validation data.")
    parser.add_argument("--name", default="hybrid_100M", type=str, help="Name of the run.")
    parser.add_argument("--wandb_project", default="YOUR_WANDB_PROJECT_NAME", type=str, help="Name of the WandB project to log into.")
    parser.add_argument("--wandb_entity", default="YOUR_WANDB_ENTITY", type=str, help="The entity to log to on WandB (typically your wandb username).")
    parser.add_argument("--config_file", default="../configs/base.json", type=Path, help="The BERT model config")
    parser.add_argument("--tokenizer_path", default="../tokenizers/tokenizer_10M.json", type=Path, help="Path to the tokenizer.")
    parser.add_argument("--output_dir", default="../model_checkpoints", type=Path, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--resume_from_checkpoint", default=None, type=str, help="The checkpoint directory to resume training from.")
    parser.add_argument("--optimizer", default="lamb", type=str, help="The optimizer to use.")
    parser.add_argument("--hybrid_numerator", default=15, type=int, help="The numerator of the hybrid ratio.")
    parser.add_argument("--hybrid_denominator", default=16, type=int, help="The denominator of the hybrid ratio (the number of GPUs should be divisible by this number).")
    parser.add_argument("--seq_length", default=128, type=int, help="Sequence length for training.")
    parser.add_argument("--local_batch_size", default=256//8, type=int, help="Batch size for training per GPU.")
    parser.add_argument("--global_batch_size", default= 16384, type=int, help="Total batch size for training per GPUs and per grad accumulation step.")
    parser.add_argument("--batch_reduction", default=4, type=int, help="The initial batch size reduction factor.")
    parser.add_argument("--learning_rate", default=1e-2, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--max_steps", default=9_914 // 2, type=int, help="Total number of training steps to perform.")
    parser.add_argument("--ema_decay", default=0.999, type=float, help="Exponential moving average decay.")
    parser.add_argument("--validate_every", default=1_000, type=int, help="Run validation after every X training shards.")
    parser.add_argument("--validation_steps", default=1, type=int, help="Number of validation steps.")
    parser.add_argument("--log_stats_every", default=100, type=int, help="Log stats every X steps.")
    parser.add_argument("--warmup_proportion", default=0.016, type=float, help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument("--cooldown_proportion", default=0.016, type=float, help="Proportion of training to perform linear learning rate cooldown for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--save_every', type=int, default=1_000, help="save every X steps")
    parser.add_argument("--mask_p_start", default=0.3, type=float, help="Initial masking probability.")
    parser.add_argument("--mask_p_end", default=0.15, type=float, help="Final masking probability.")
    parser.add_argument("--mask_random_p", default=0.1, type=float, help="Probability of replacing the masked token with a random token.")
    parser.add_argument("--mask_keep_p", default=0.1, type=float, help="Probability of keeping the masked token.")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--optimizer_eps", default=1e-8, type=float, help="Optimizer epsilon.")
    parser.add_argument("--optimizer_beta1", default=0.9, type=float, help="Optimizer beta1.")
    parser.add_argument("--optimizer_beta2", default=0.98, type=float, help="Optimizer beta2.")
    parser.add_argument("--max_gradient", default=2.0, type=float, help="Max value for gradient clipping.")
    parser.add_argument('--mixed_precision', default='fp16', type=str, choices=['no', 'fp16', 'bf16'], help="Mixed precision training.")
    parser.add_argument('--n_special_tokens', default=16, type=int, help="Number of special tokens.")
    parser.add_argument('--z_loss_weight', default=1e-4, type=float, help="Weight for the z loss.")
    parser.add_argument('--token_weighted_loss', default=False, action=argparse.BooleanOptionalAction, help="Use token weighted loss.")
    args = parser.parse_args()

    args.name = "_".join([args.name, str(args.hybrid_numerator), str(args.hybrid_denominator)])
    args.output_path = (args.output_dir / args.name).with_suffix(".bin")

    return args


def load_config(args):
    with args.config_file.open("r") as f:
        config = json.load(f)
    for k, v in config.items():
        setattr(args, k, v)
    return args


def prepare_model_and_optimizer(args, accelerator):
    args = load_config(args)
    model = Bert(args)

    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(model)
        print(f"NUMBER OF PARAMETERS: {n_params}\n", flush=True)

    no_decay = ['bias', 'layer_norm']
    decay_params = [(n, p) for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
    no_decay_params = [(n, p) for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
    optimizer_grouped_parameters = [
        {'params': [p for _, p in decay_params], 'weight_decay': args.weight_decay},
        {'params': [p for _, p in no_decay_params], 'weight_decay': 0.0}
    ]

    if accelerator.is_main_process:
        print("Parameters without weight decay:")
        for n, _ in no_decay_params:
            print(n)
        print("\nParameters with weight decay:")
        for n, _ in decay_params:
            print(n)
        print(flush=True)

    if args.optimizer.lower() in ["adam", "adamw"]:
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            betas=(args.optimizer_beta1, args.optimizer_beta2),
            eps=args.optimizer_eps,
        )
    elif args.optimizer.lower() == "lamb":
        optimizer = Lamb(
            optimizer_grouped_parameters,
            args.learning_rate,
            betas=(args.optimizer_beta1, args.optimizer_beta2),
            eps=args.optimizer_eps,
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    scheduler = cosine_schedule_with_warmup_cooldown(
        optimizer,
        int(args.max_steps * args.warmup_proportion),
        int(args.max_steps * args.cooldown_proportion),
        args.max_steps,
        0.1
    )

    ema_model: nn.Module = copy.deepcopy(model)
    for param in ema_model.parameters():
        param.requires_grad = False

    return model, ema_model, optimizer, scheduler


# def get_batch(dataloader_iter, global_step):
#     dataloader_iter._dataset.set_global_step(global_step)
#     # Data is automatically moved to the correct device by the prepared dataloader
#     input_ids, target_ids, attention_mask, mask_p = next(dataloader_iter)
#     input_ids, target_ids = input_ids.t(), target_ids.t()
#     mask_p = mask_p.mean()

    # return input_ids, attention_mask, target_ids, mask_p

# Change the function signature to accept `dataloader`
def get_batch(dataloader, dataloader_iter, global_step):
    # Now, access .dataset from the main dataloader object
    dataloader.dataset.set_global_step(global_step)
    
    # The rest of the function remains the same
    input_ids, target_ids, attention_mask, mask_p = next(dataloader_iter)
    input_ids, target_ids = input_ids.t(), target_ids.t()
    mask_p = mask_p.mean()

    return input_ids, attention_mask, target_ids, mask_p
def training_epoch(accelerator, model, ema_model, train_dataloader, valid_dataloader, optimizer, scheduler, global_step, epoch, args):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    num_steps_this_epoch = min(len(train_dataloader) // accelerator.gradient_accumulation_steps, args.max_steps - global_step)
    train_dataloader_iter = iter(train_dataloader)
    
    pbar = tqdm(range(num_steps_this_epoch), desc="Train iteration", disable=not accelerator.is_main_process)

    for step in range(num_steps_this_epoch):
        total_loss, total_accuracy, total_z_loss, total_mask_p, total_grad_norm = 0.0, 0.0, 0.0, 0.0, 0.0
        
        with accelerator.accumulate(model):
            # input_ids, attention_mask, target_ids, mask_p = get_batch(train_dataloader_iter, global_step)
            input_ids, attention_mask, target_ids, mask_p = get_batch(train_dataloader, train_dataloader_iter, global_step)

            with ModelLogger(enable=global_step % 100 == 0, module=model):
                loss, accuracy, z_loss, num_tokens = model(input_ids, attention_mask, target_ids)

            if args.token_weighted_loss:
                total_tokens = torch.tensor(num_tokens, device=accelerator.device, dtype=torch.long)
                total_tokens = accelerator.reduce(total_tokens, reduction="sum")
                weight = accelerator.num_processes * num_tokens / total_tokens
            else:
                weight = 1.0
            
            loss_to_backward = (loss + args.z_loss_weight * z_loss) * weight
            accelerator.backward(loss_to_backward)

            if accelerator.sync_gradients:
                total_grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_gradient)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            with torch.no_grad():
                unwrapped_model = accelerator.unwrap_model(model)
                for param_q, param_k in zip(unwrapped_model.parameters(), ema_model.parameters()):
                    param_k.data.mul_(args.ema_decay).add_((1.0 - args.ema_decay) * param_q.detach().data)

            # Gather metrics across all processes for logging
            avg_loss = accelerator.gather_for_metrics(loss.detach()).mean()
            avg_accuracy = accelerator.gather_for_metrics(accuracy).mean()
            avg_z_loss = accelerator.gather_for_metrics(z_loss.detach()).mean()
            avg_mask_p = accelerator.gather_for_metrics(mask_p).mean()
            
            # Note: The original script's hybrid MLM/CLM loss logging was complex and
            # depended on the Slurm rank. This version logs the average loss across all GPUs.
            # This can be extended if specific per-type loss is required.
            if accelerator.is_main_process:
                accelerator.log({
                    "epoch": epoch,
                    "train/loss": avg_loss.item(),
                    "train/z_loss": avg_z_loss.item(),
                    "train/perplexity": math.exp(avg_loss.item()),
                    "train/accuracy": avg_accuracy.item() * 100.0,
                    "stats/learning_rate": optimizer.param_groups[0]['lr'],
                    "stats/grad_norm": total_grad_norm.item() if torch.is_tensor(total_grad_norm) else total_grad_norm,
                    "stats/seq_length": train_dataloader.dataset.seq_length,
                    "stats/global_batch_size": args.current_global_batch_size,
                    "stats/local_batch_size": args.current_local_batch_size,
                    "stats/accumulate_steps": accelerator.gradient_accumulation_steps,
                    "stats/mask_p": avg_mask_p.item(),
                }, step=global_step)

            global_step += 1
            pbar.update(1)

            if (global_step % args.save_every == 0):
                save(accelerator, ema_model, args)

            if (global_step % args.validate_every == 0):
                validation_epoch(accelerator, model, valid_dataloader, epoch, args)

            if global_step >= args.max_steps:
                return global_step
    
    return global_step


@torch.no_grad()
def validation_epoch(accelerator, model, valid_dataloader, epoch, args, commit=False):
    model.eval()
    losses, accuracies = [], []
    valid_dataloader_iter = iter(valid_dataloader)

    for _ in tqdm(range(args.validation_steps), desc="Valid iteration", disable=not accelerator.is_main_process):
        # input_ids, attention_mask, target_ids, _ = get_batch(valid_dataloader_iter, 0)
        input_ids, attention_mask, target_ids, mask_p = get_batch(valid_dataloader, valid_dataloader_iter, 0)
        loss, accuracy, _, num_tokens = model(input_ids, attention_mask, target_ids)

        gathered_losses = accelerator.gather_for_metrics(loss)
        gathered_accuracies = accelerator.gather_for_metrics(accuracy)

        losses.append(gathered_losses.mean().item())
        accuracies.append(gathered_accuracies.mean().item())

    if accelerator.is_main_process:
        val_loss = mean(losses)
        val_acc = mean(accuracies)
        accelerator.log({
            "epoch": epoch,
            "validation/loss": val_loss,
            "validation/accuracy": val_acc * 100.0,
            "validation/perplexity": math.exp(val_loss)
        }, commit=commit)
    
    model.train()


def save(accelerator, ema_model, args):
    if accelerator.is_main_process:
        output_dir = Path(args.output_dir)
        run_dir = output_dir / args.name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the full training state for resumption
        accelerator.save_state(run_dir)

        # Save just the unwrapped model weights for inference
        unwrapped_model = accelerator.unwrap_model(accelerator.model)
        torch.save(unwrapped_model.state_dict(), run_dir / "pytorch_model.bin")
        
        # Also save the EMA model
        torch.save(ema_model.state_dict(), run_dir / "pytorch_model_ema.bin")
        print(f"Checkpoint saved to {run_dir}")


def load_datasets(accelerator, args, tokenizer, epoch, global_step, train_dataloader, valid_dataloader):
    # Determine dataset type based on process index (rank)
    if accelerator.process_index * args.hybrid_denominator < args.hybrid_numerator * accelerator.num_processes:
        args.dataset_type = "masked"
    else:
        args.dataset_type = "causal"

    if accelerator.is_main_process:
        print(f"Process {accelerator.process_index} is of type {args.dataset_type}")

    train_seed = args.seed + accelerator.process_index + epoch * accelerator.num_processes

    if (global_step + 1) / args.max_steps >= 0.9:
        seq_length = args.seq_length * 4
        global_batch_size = args.global_batch_size // 4
    elif (global_step + 1) / args.max_steps >= 0.7:
        seq_length = args.seq_length * 2
        global_batch_size = args.global_batch_size // 2
    else:
        seq_length = args.seq_length
        global_batch_size = args.global_batch_size

    if train_dataloader is None or train_dataloader.dataset.seq_length!= seq_length:
        if args.dataset_type == "masked":
            num_masked_processes = accelerator.num_processes * args.hybrid_numerator // args.hybrid_denominator
            rank = accelerator.process_index
            world_size = num_masked_processes
            train_data = MaskedDataset(args.train_path, tokenizer, args, seq_length, rank, world_size)
        else:
            num_masked_processes = accelerator.num_processes * args.hybrid_numerator // args.hybrid_denominator
            num_causal_processes = accelerator.num_processes - num_masked_processes
            rank = accelerator.process_index - num_masked_processes
            world_size = num_causal_processes
            train_data = CausalDataset(args.train_path, tokenizer, args, seq_length, rank, world_size)

        # if accelerator.is_main_process:
        #     train_data.show_random_item(tokenizer)
    else:
        train_data = train_dataloader.dataset

    args.current_global_batch_size = int(global_batch_size / args.batch_reduction * (1 - global_step / args.max_steps) + global_batch_size * (global_step / args.max_steps) + 0.5)
    total_local_batch_size = int(args.current_global_batch_size / accelerator.num_processes + 0.5)
    args.accumulate_steps = int(math.ceil(total_local_batch_size / args.local_batch_size))
    args.current_local_batch_size = total_local_batch_size // args.accumulate_steps if args.accumulate_steps > 0 else total_local_batch_size

    train_dataloader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=args.current_local_batch_size,
        num_workers=0,
        generator=torch.Generator().manual_seed(train_seed),
        drop_last=True,
        pin_memory=True,
    )

    if valid_dataloader is None:
        valid_data = ValidationDataset(args.valid_path, tokenizer, args)
        valid_dataloader = DataLoader(
            valid_data,
            shuffle=False,
            batch_size=args.local_batch_size,
            num_workers=0,
            generator=torch.Generator().manual_seed(42),
            drop_last=True,
            pin_memory=True,
        )

    return train_dataloader, valid_dataloader


def main():
    args = parse_arguments()

    accelerator = Accelerator(
        log_with="wandb"
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
            init_kwargs={"wandb": {"entity": args.wandb_entity, "name": args.name}}
        )
        print(f"Training with {accelerator.num_processes} GPUs")
        print(f"In total, the model will be trained on 'steps'({args.max_steps:,}) x 'GPUs'({accelerator.num_processes}) x 'batch_size'({args.local_batch_size:,}) x 'seq_len'({args.seq_length:,}) = {args.max_steps * accelerator.num_processes * args.local_batch_size * args.seq_length:,} subword instances")


    set_seed(args.seed)

    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    args.vocab_size = tokenizer.get_vocab_size()

    model, ema_model, optimizer, scheduler = prepare_model_and_optimizer(args, accelerator)
    ema_model.to(accelerator.device)

    train_dataloader, valid_dataloader = None, None
    
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    global_step = 0
    start_epoch = 0

    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
        # The step and epoch need to be extracted from the saved state.
        # This part might need custom logic depending on how you save/load state.
        # For simplicity, we assume we can manually track or infer them.
        # A more robust solution would save step/epoch in a separate file.
        print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")


    for epoch in count(start=start_epoch):
        train_dataloader, valid_dataloader = load_datasets(accelerator, args, tokenizer, epoch, global_step, train_dataloader, valid_dataloader)
        
        # Prepare dataloaders for each epoch as they might be recreated
        prepared_train_dl, prepared_valid_dl = accelerator.prepare(train_dataloader, valid_dataloader)
        
        # Update gradient accumulation steps on the accelerator
        accelerator.gradient_accumulation_steps = args.accumulate_steps

        global_step = training_epoch(accelerator, model, ema_model, prepared_train_dl, prepared_valid_dl, optimizer, scheduler, global_step, epoch, args)

        if global_step >= args.max_steps:
            break

    save(accelerator, ema_model, args)
    validation_epoch(accelerator, model, prepared_valid_dl, epoch, args, commit=True)

    if accelerator.is_main_process:
        accelerator.end_training()


if __name__ == "__main__":
    main()
