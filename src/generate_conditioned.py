"""
Génération avec le modèle conditionné.

Fonctionnalités:
- Générer avec des accessoires spécifiques
- Classifier-Free Guidance (CFG) pour améliorer la qualité
- Combinaison d'accessoires
"""

import torch
import argparse
import os
import json
from PIL import Image
import torchvision
from tqdm import tqdm

from model_conditioned import UNetConditioned
from diffusion import Diffusion
from config import get_config_by_name


def load_model_and_mapping(checkpoint_path, device):
    """Charger le modèle et le mapping des accessoires."""
    # Charger checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Charger mapping
    model_dir = os.path.dirname(checkpoint_path)
    mapping_path = os.path.join(model_dir, 'accessory_mapping.json')
    
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    
    # Récupérer config du checkpoint ou utiliser défauts
    ckpt_config = checkpoint.get('config', {})
    
    # Créer modèle
    model = UNetConditioned(
        c_in=3,
        c_out=3,
        time_dim=ckpt_config.get('time_dim', 64),
        num_accessories=checkpoint['num_accessories'],
        concept_dim=ckpt_config.get('concept_dim', 512),
        concept_scale=ckpt_config.get('concept_scale', 1.0),
        cfg_dropout=0.0,  # Pas de dropout à l'inférence
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, mapping


def create_accessory_vector(accessory_names, mapping, device):
    """
    Créer un vecteur multi-hot à partir des noms d'accessoires.
    """
    accessory_to_idx = mapping['accessory_to_idx']
    num_accessories = mapping['num_accessories']
    
    vector = torch.zeros(num_accessories, device=device)
    
    matched = []
    for name in accessory_names:
        # Essayer correspondance exacte
        if name in accessory_to_idx:
            vector[accessory_to_idx[name]] = 1.0
            matched.append(name)
        else:
            # Essayer correspondance insensible à la casse
            for key in accessory_to_idx:
                if key.lower() == name.lower():
                    vector[accessory_to_idx[key]] = 1.0
                    matched.append(key)
                    break
            else:
                print(f"⚠️  Accessoire inconnu: '{name}'")
    
    return vector, matched


def sample_ddpm(model, diffusion, accessory_vector, n_samples, device):
    """
    Sampling DDPM standard (sans CFG).
    """
    model.eval()
    
    with torch.no_grad():
        x = torch.randn(n_samples, 3, diffusion.img_size, diffusion.img_size).to(device)
        
        # Broadcast accessory vector
        if accessory_vector is not None:
            labels = accessory_vector.unsqueeze(0).expand(n_samples, -1)
        else:
            labels = None
        
        for t in tqdm(reversed(range(diffusion.noise_steps)), desc="Sampling"):
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
            
            predicted_noise = model(x, t_batch, accessory_labels=labels)
            
            alpha = diffusion.alpha[t]
            alpha_hat = diffusion.alpha_hat[t]
            beta = diffusion.beta[t]
            
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_hat)) * predicted_noise)
            x = x + torch.sqrt(beta) * noise
    
    x = (x.clamp(-1, 1) + 1) / 2
    return x


def sample_cfg(model, diffusion, accessory_vector, n_samples, cfg_scale, device):
    """
    Sampling avec Classifier-Free Guidance.
    
    ε_guided = ε_uncond + cfg_scale * (ε_cond - ε_uncond)
    """
    model.eval()
    
    with torch.no_grad():
        x = torch.randn(n_samples, 3, diffusion.img_size, diffusion.img_size).to(device)
        
        # Prepare conditional and unconditional labels
        if accessory_vector is not None:
            cond_labels = accessory_vector.unsqueeze(0).expand(n_samples, -1)
        else:
            cond_labels = torch.zeros(n_samples, model.num_accessories, device=device)
        
        uncond_labels = torch.zeros(n_samples, model.num_accessories, device=device)
        
        for t in tqdm(reversed(range(diffusion.noise_steps)), desc="Sampling (CFG)"):
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
            
            # Conditional prediction
            noise_cond = model(x, t_batch, accessory_labels=cond_labels)
            
            # Unconditional prediction
            noise_uncond = model(x, t_batch, accessory_labels=uncond_labels)
            
            # CFG: guided = uncond + scale * (cond - uncond)
            predicted_noise = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
            
            alpha = diffusion.alpha[t]
            alpha_hat = diffusion.alpha_hat[t]
            beta = diffusion.beta[t]
            
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_hat)) * predicted_noise)
            x = x + torch.sqrt(beta) * noise
    
    x = (x.clamp(-1, 1) + 1) / 2
    return x


def main():
    parser = argparse.ArgumentParser(description="Generate with conditioned DDPM")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--accessories', type=str, nargs='+', default=[],
                        help='Accessories to condition on (e.g., Cap Pipe Shades)')
    parser.add_argument('--n', type=int, default=4,
                        help='Number of images to generate')
    parser.add_argument('--cfg_scale', type=float, default=3.0,
                        help='CFG guidance scale (1.0 = no guidance)')
    parser.add_argument('--output', type=str, default='output/generated.png',
                        help='Output path')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--list', action='store_true',
                        help='List available accessories')
    parser.add_argument('--no_cfg', action='store_true',
                        help='Disable CFG (use standard sampling)')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Device: {device}")
    
    # Seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        print(f"🎲 Seed: {args.seed}")
    
    # Charger modèle
    print(f"📦 Loading model from {args.checkpoint}")
    model, mapping = load_model_and_mapping(args.checkpoint, device)
    print(f"✓ Model loaded ({mapping['num_accessories']} accessories)")
    
    # Lister accessoires
    if args.list:
        print("\n📋 Available accessories:")
        for i, acc in enumerate(mapping['accessory_list']):
            print(f"  {i:2d}. {acc}")
        return
    
    # Diffusion
    diffusion = Diffusion(
        noise_steps=1000,
        img_size=32,
        img_channels=3,
        device=device,
    )
    
    # Créer vecteur accessoires
    if args.accessories:
        accessory_vector, matched = create_accessory_vector(args.accessories, mapping, device)
        print(f"\n🎨 Conditioning on: {matched}")
        num_active = accessory_vector.sum().item()
        print(f"   Active accessories: {int(num_active)}")
    else:
        accessory_vector = None
        print("\n🎨 Unconditional generation")
    
    # Génération
    print(f"\n🖼️  Generating {args.n} images...")
    
    if args.no_cfg or args.cfg_scale == 1.0:
        print("   Mode: Standard DDPM")
        images = sample_ddpm(model, diffusion, accessory_vector, args.n, device)
    else:
        print(f"   Mode: CFG (scale={args.cfg_scale})")
        images = sample_cfg(model, diffusion, accessory_vector, args.n, args.cfg_scale, device)
    
    # Sauvegarder
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    nrow = int(args.n ** 0.5)
    if nrow * nrow < args.n:
        nrow += 1
    
    grid = torchvision.utils.make_grid(images, nrow=nrow, padding=2)
    grid_pil = torchvision.transforms.ToPILImage()(grid)
    grid_pil.save(args.output)
    
    print(f"\n✅ Saved to {args.output}")


if __name__ == "__main__":
    main()
