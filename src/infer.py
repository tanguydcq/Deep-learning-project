import os
import torch
from torchvision.utils import save_image
import argparse

from model_vector import get_model
from diffusion import Diffusion
from config import get_config_by_name


def infer_base_model(args):
    device = args.device

    # -------------------------
    # Load model
    # -------------------------
    model = get_model(args).to(device)
    model.eval()

    ckpt = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    print(f"[OK] Loaded checkpoint: {args.checkpoint_path}")

    # -------------------------
    # Diffusion
    # -------------------------
    diffusion = Diffusion(
        noise_steps=args.T,
        img_size=args.img_size,
        img_channels=args.image_channels,
        device=device,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    )

    # -------------------------
    # Output directory
    # -------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------
    # Sampling
    # -------------------------
    with torch.no_grad():
        samples = diffusion.sample(
            model,
            n=args.n_samples,
        )

    # samples in [-1,1] → [0,1]
    samples = (samples + 1) / 2
    samples.clamp_(0, 1)

    save_path = os.path.join(args.output_dir, "base_samples.png")
    save_image(samples, save_path, nrow=int(args.n_samples ** 0.5))

    print(f"[DONE] Saved samples to {save_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="cryptopunks1",
        help="Configuration name (e.g., cryptopunks1, mnist)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the checkpoint file to load",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/inference",
        help="Directory to save generated samples",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=16,
        help="Number of samples to generate",
    )
    cli_args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration: {cli_args.config}")
    config_args = get_config_by_name(cli_args.config)
    
    # Add CLI arguments to config
    config_args.checkpoint_path = cli_args.checkpoint_path
    config_args.output_dir = cli_args.output_dir
    config_args.n_samples = cli_args.n_samples
    
    infer_base_model(config_args)
