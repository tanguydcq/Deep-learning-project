"""
ÉTAPE E - Génération contrôlée

Méthodologie:
- Phrase → concepts → vecteurs
- Combinaison linéaire: c = Σ_k β_k * c_k
- Sampling DDPM standard avec injection de c
"""

import torch
import argparse
import os
from PIL import Image
import torchvision
from tqdm import tqdm

from model_vector import get_model
from diffusion import Diffusion
from config import get_config_by_name


def load_concept_vector(concept_path, device):
    """Charger un vecteur concept depuis un fichier .pt"""
    data = torch.load(concept_path, map_location=device)
    vector = data['vector'].to(device)
    return vector, data.get('concept', os.path.basename(concept_path))


def sample_with_concept(model, diffusion, concept_vector, n_samples, device):
    """
    Sampling DDPM avec injection du concept vector.
    
    À chaque step: ε_θ(x_t, t, c)
    """
    model.eval()
    
    with torch.no_grad():
        # x_T ~ N(0, I)
        x = torch.randn(n_samples, 3, diffusion.img_size, diffusion.img_size).to(device)
        
        # Broadcast concept vector
        if concept_vector is not None:
            c = concept_vector.unsqueeze(0).expand(n_samples, -1)
        else:
            c = None
        
        # Reverse process
        for t in tqdm(reversed(range(diffusion.noise_steps)), desc="Sampling"):
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
            
            # Prédire le bruit avec le concept
            predicted_noise = model(x, t_batch, concept_vector=c)
            
            # Denoising step
            alpha = diffusion.alpha[t]
            alpha_hat = diffusion.alpha_hat[t]
            beta = diffusion.beta[t]
            
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            # x_{t-1} = (1/√α_t) * (x_t - (β_t/√(1-α̂_t)) * ε_θ) + √β_t * z
            x = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_hat)) * predicted_noise)
            x = x + torch.sqrt(beta) * noise
    
    # Normalize to [0, 1]
    x = (x.clamp(-1, 1) + 1) / 2
    return x


def main():
    parser = argparse.ArgumentParser(description="ÉTAPE E - Génération contrôlée")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Chemin vers le checkpoint du modèle DDPM')
    parser.add_argument('--config', type=str, default='cryptopunks1',
                        help='Nom de la configuration')
    parser.add_argument('--concepts', type=str, nargs='+', default=[],
                        help='Concepts à utiliser (e.g., cap pipe shades)')
    parser.add_argument('--concepts_dir', type=str, default='concepts',
                        help='Dossier contenant les vecteurs concepts')
    parser.add_argument('--weights', type=float, nargs='+', default=None,
                        help='Poids β_k pour chaque concept (default: 1.0 pour tous)')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Échelle globale pour le vecteur combiné')
    parser.add_argument('--n', type=int, default=4,
                        help='Nombre d\'images à générer')
    parser.add_argument('--output', type=str, default='output/generated.png',
                        help='Chemin de sortie')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--list', action='store_true',
                        help='Lister les concepts disponibles')
    
    args = parser.parse_args()
    
    # ========== Seed ==========
    if args.seed is not None:
        torch.manual_seed(args.seed)
        print(f"🎲 Seed: {args.seed}")
    
    # ========== Liste des concepts ==========
    if args.list:
        print("\n📋 Concepts disponibles:")
        for f in sorted(os.listdir(args.concepts_dir)):
            if f.endswith('.pt'):
                path = os.path.join(args.concepts_dir, f)
                data = torch.load(path, map_location='cpu')
                norm = data['vector'].norm().item()
                print(f"  • {f.replace('.pt', ''):<25} |c| = {norm:.4f}")
        return
    
    # ========== Config & Device ==========
    config = get_config_by_name(args.config)
    device = config.device
    print(f"🔧 Device: {device}")
    
    # ========== Charger modèle ==========
    print(f"📦 Loading model from {args.checkpoint}")
    model = get_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Model loaded (α = {model.concept_scale})")
    
    # ========== Diffusion ==========
    diffusion = Diffusion(
        noise_steps=config.T,
        img_size=config.img_size,
        img_channels=config.image_channels,
        device=device,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
    )
    
    # ========== Charger et combiner les concepts ==========
    if args.concepts:
        # Poids par défaut: 1.0 pour tous
        weights = args.weights if args.weights else [1.0] * len(args.concepts)
        
        if len(weights) != len(args.concepts):
            print(f"⚠️  Nombre de poids ({len(weights)}) != nombre de concepts ({len(args.concepts)})")
            weights = [1.0] * len(args.concepts)
        
        print(f"\n🎨 Combinaison linéaire: c = Σ β_k * c_k")
        
        # c = Σ_k β_k * c_k
        combined_vector = torch.zeros(config.concept_dim, device=device)
        
        for concept, weight in zip(args.concepts, weights):
            # Trouver le fichier
            concept_file = f"acc_{concept.lower().replace(' ', '_')}.pt"
            concept_path = os.path.join(args.concepts_dir, concept_file)
            
            if not os.path.exists(concept_path):
                print(f"⚠️  Concept not found: {concept_path}")
                continue
            
            vector, name = load_concept_vector(concept_path, device)
            combined_vector = combined_vector + weight * vector
            print(f"   + {weight:.2f} × {concept:<15} (|c_k| = {vector.norm().item():.4f})")
        
        # Appliquer l'échelle globale
        combined_vector = args.scale * combined_vector
        print(f"\n   Scale: {args.scale}")
        print(f"   |c_final| = {combined_vector.norm().item():.4f}")
    else:
        print("🎨 Génération sans concept (c = 0)")
        combined_vector = None
    
    # ========== Génération ==========
    print(f"\n🖼️  Generating {args.n} images...")
    images = sample_with_concept(
        model=model,
        diffusion=diffusion,
        concept_vector=combined_vector,
        n_samples=args.n,
        device=device,
    )
    
    # ========== Sauvegarder ==========
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    # Grille d'images
    nrow = int(args.n ** 0.5)
    if nrow * nrow < args.n:
        nrow += 1
    
    grid = torchvision.utils.make_grid(images, nrow=nrow, padding=2)
    grid_pil = torchvision.transforms.ToPILImage()(grid)
    grid_pil.save(args.output)
    
    print(f"\n✅ Saved to {args.output}")


if __name__ == "__main__":
    main()
