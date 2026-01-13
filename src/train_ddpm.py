import os
import torch
import torch.nn as nn
import logging
import numpy as np
import argparse

from tqdm import tqdm
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from utils import setup_logging
from model_vector import get_model
from dataset import get_data
from diffusion import Diffusion
from config import get_config_by_name
import shutil


logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)


def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved to {path}")


def train(args, config_name, resume=False, reset=False):
    run_name = os.path.join(args.dataset_name, config_name)

    models_dir = os.path.join("models", run_name)
    results_dir = os.path.join("results", run_name)
    runs_dir = os.path.join("runs", run_name)

    if reset:
        for path in [models_dir, results_dir, runs_dir]:
            if os.path.exists(path):
                shutil.rmtree(path)
        resume = False

    setup_logging(run_name)
    device = args.device

    dataloader = get_data(args)
    model = get_model(args).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    diffusion = Diffusion(
        noise_steps=args.T,
        img_size=args.img_size,
        img_channels=args.image_channels,
        device=device,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    )

    writer = SummaryWriter(os.path.join("runs", run_name))

    start_epoch = 0
    global_step = 0

    if resume:
        ckpt_path = os.path.join(models_dir, "ckpt_final.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1

    l = len(dataloader)
    global_step = start_epoch * l

    # ===========================
    # DDPM TRAINING (PHASE 1)
    # ===========================
    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs}")

        for images, _ in pbar:
            images = images.to(device, non_blocking=True)

            t = diffusion.sample_timesteps(images.size(0)).to(device)
            x_t, noise = diffusion.noise_images(images, t)

            # ZERO concept vector (important)
            concept_vector = torch.zeros(
                images.size(0),
                model.concept_dim,
                device=device,
            )

            predicted_noise = model(
                x_t,
                t,
                concept_vector=concept_vector,
            )

            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            writer.add_scalar("Train/MSE", loss.item(), global_step)
            global_step += 1

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                samples = diffusion.sample(
                    model,
                    n=4,
                )
            save_path = os.path.join(results_dir, f"sample_epoch_{epoch}.png")

        if epoch % 20 == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                os.path.join(models_dir, f"ckpt_{epoch}.pt"),
            )

    save_checkpoint(
        model,
        optimizer,
        epoch,
        os.path.join(models_dir, "ckpt_final.pt"),
    )
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config1_cryptopunks",
        help="Configuration name (e.g., config1, config2)",
    )
    cli_args = parser.parse_args()

    # Load the specific configuration
    try:
        config_args = get_config_by_name(cli_args.config)
        print(f"Loading configuration: {cli_args.config}")
        train(config_args, cli_args.config)
    except ValueError as e:
        print(e)
    