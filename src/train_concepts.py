"""
ÉTAPE D - Apprentissage des concepts

Méthodologie:
- Modèle DDPM pré-entraîné avec c=0 (freeze total)
- Pour chaque concept k: optimiser c_k sur le dataset filtré
- Loss: min_{c_k} E||ε - ε_θ(x_t, t, c_k)||²

Pas de régularisation, pas de contrastive, juste la loss DDPM standard.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import json
from PIL import Image
import torchvision.transforms as transforms

from model_vector import get_model
from diffusion import Diffusion
from config import get_config_by_name
import argparse


class ConceptDataset(Dataset):
    """
    Dataset filtré pour un concept spécifique.
    """
    def __init__(self, images_dir, subdataset_path, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        
        with open(subdataset_path, 'r') as f:
            subdataset = json.load(f)
        
        self.concept_name = subdataset['accessory_name']
        self.image_ids = subdataset['punk_ids']
        
        print(f"📁 Concept '{self.concept_name}': {len(self.image_ids)} images")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.images_dir, f"{img_id}.png")
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image


def train_concept(
    model,
    diffusion,
    dataset,
    concept_name,
    device,
    epochs=100,
    lr=1e-3,
    batch_size=16,
    save_path="concepts",
    patience=30,
    init_std=0.01,
):
    """
    ÉTAPE D: Apprentissage d'un vecteur concept.
    
    min_{c_k} E||ε - ε_θ(x_t, t, c_k)||²
    
    Args:
        model: UNet pré-entraîné (FREEZE)
        diffusion: Diffusion process
        dataset: Images du concept uniquement
        concept_name: Nom du concept
        init_std: Écart-type pour initialisation c ~ N(0, σ²)
    """
    os.makedirs(save_path, exist_ok=True)

    # ========== ÉTAPE C: Freeze total ==========
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    # ========== Initialisation c_k ~ N(0, σ²) ==========
    c = nn.Parameter(
        torch.randn(model.concept_dim, device=device) * init_std
    )
    
    print(f"🔧 Concept dim: {model.concept_dim}")
    print(f"🔧 Init std: {init_std}, |c| initial = {c.norm().item():.4f}")
    print(f"🔧 Concept scale (α): {model.concept_scale}")

    # Optimiseur UNIQUEMENT sur c_k
    optimizer = optim.Adam([c], lr=lr)
    mse = nn.MSELoss()
    
    # Early stopping
    best_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_vector = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    print(f"\n{'='*60}")
    print(f"Training concept: {concept_name}")
    print(f"{'='*60}")

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        epoch_losses = []

        for images in pbar:
            images = images.to(device)
            B = images.size(0)

            # Sample timesteps
            t = diffusion.sample_timesteps(B).to(device)
            
            # Forward diffusion: x_t = √(α̂_t) * x_0 + √(1-α̂_t) * ε
            x_t, noise = diffusion.noise_images(images, t)

            # Broadcast concept vector pour le batch
            c_batch = c.unsqueeze(0).expand(B, -1)

            # Prédiction du bruit avec injection du concept
            # ε_θ(x_t, t, c_k)
            predicted_noise = model(x_t, t, concept_vector=c_batch)

            # Loss DDPM: ||ε - ε_θ(x_t, t, c_k)||²
            loss = mse(noise, predicted_noise)
            epoch_losses.append(loss.item())

            # Backprop sur c uniquement
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(
                MSE=f"{loss.item():.5f}",
                norm=f"{c.norm().item():.3f}"
            )

        # Métriques epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        c_norm = c.norm().item()
        
        print(f"[Epoch {epoch:3d}] MSE={avg_loss:.6f} | |c|={c_norm:.4f}")
        
        # Early stopping
        if avg_loss < best_loss - 1e-6:  # Amélioration significative
            best_loss = avg_loss
            best_vector = c.detach().clone()
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping at epoch {epoch}")
            print(f"📌 Best MSE: {best_loss:.6f} at epoch {best_epoch}")
            c.data = best_vector
            break

    # Sauvegarder le vecteur
    final_norm = c.norm().item()
    torch.save(
        {
            "concept": concept_name,
            "vector": c.detach().cpu(),
            "norm": final_norm,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
        },
        os.path.join(save_path, f"{concept_name}.pt"),
    )

    print(f"\n✅ Saved: {concept_name}.pt")
    print(f"   |c| = {final_norm:.4f}")
    print(f"   Best MSE = {best_loss:.6f} (epoch {best_epoch})")

    return c.detach()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ÉTAPE D - Apprentissage des concepts")
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Chemin vers le checkpoint du modèle DDPM')
    parser.add_argument('--config', type=str, default='cryptopunks1',
                        help='Nom de la configuration')
    parser.add_argument('--subdatasets_dir', type=str, default='subdatasets',
                        help='Dossier contenant les fichiers JSON des subdatasets')
    parser.add_argument('--concepts', type=str, nargs='+', 
                        default=['Cap', 'Pipe', 'Cigarette', 'Hoodie', 'Shades'],
                        help='Liste des concepts à entraîner')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Nombre d\'epochs par concept')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience')
    parser.add_argument('--init_std', type=float, default=0.01,
                        help='Écart-type pour l\'initialisation c ~ N(0, σ²)')
    parser.add_argument('--save_path', type=str, default='concepts',
                        help='Dossier pour sauvegarder les vecteurs')
    
    args = parser.parse_args()
    
    # ========== Charger config ==========
    print(f"📋 Configuration: {args.config}")
    config = get_config_by_name(args.config)
    device = config.device
    
    # ========== Charger modèle (ÉTAPE C: Freeze) ==========
    print(f"📦 Loading model from {args.checkpoint}")
    model = get_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Model loaded (concept_scale α = {model.concept_scale})")
    
    # ========== Diffusion ==========
    diffusion = Diffusion(
        noise_steps=config.T,
        img_size=config.img_size,
        img_channels=config.image_channels,
        device=device,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
    )
    
    # ========== Transform ==========
    transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Images directory
    images_dir = config.dataset_path.replace('CRYPTOPUNKS', 'CRYPTOPUNKS_CLASSES') + '/images'
    
    # ========== ÉTAPE D: Entraîner chaque concept ==========
    results = []
    
    for concept in args.concepts:
        print(f"\n{'='*60}")
        
        # Construire le chemin du subdataset
        concept_filename = f"acc_{concept.lower().replace(' ', '_')}.json"
        subdataset_path = os.path.join(args.subdatasets_dir, concept_filename)
        
        if not os.path.exists(subdataset_path):
            print(f"⚠️  Subdataset not found: {subdataset_path}")
            results.append({'concept': concept, 'status': 'NOT_FOUND'})
            continue
        
        try:
            # Charger dataset filtré
            dataset = ConceptDataset(
                images_dir=images_dir,
                subdataset_path=subdataset_path,
                transform=transform,
            )
            
            if len(dataset) == 0:
                print(f"⚠️  Empty dataset for '{concept}'")
                results.append({'concept': concept, 'status': 'EMPTY'})
                continue
            
            # Entraîner le concept
            vector = train_concept(
                model=model,
                diffusion=diffusion,
                dataset=dataset,
                concept_name=f"acc_{concept.lower().replace(' ', '_')}",
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                save_path=args.save_path,
                patience=args.patience,
                init_std=args.init_std,
            )
            
            results.append({
                'concept': concept,
                'status': 'SUCCESS',
                'norm': vector.norm().item(),
                'images': len(dataset),
            })
            
        except Exception as e:
            print(f"❌ Error training '{concept}': {e}")
            import traceback
            traceback.print_exc()
            results.append({'concept': concept, 'status': 'ERROR'})
    
    # ========== Résumé ==========
    print(f"\n{'='*60}")
    print("RÉSUMÉ")
    print(f"{'='*60}")
    print(f"{'Concept':<20} {'Status':<12} {'|c|':<10} {'Images':<10}")
    print(f"{'-'*20} {'-'*12} {'-'*10} {'-'*10}")
    
    for r in results:
        status = r['status']
        emoji = {'SUCCESS': '✓', 'NOT_FOUND': '⚠️', 'EMPTY': '⚠️', 'ERROR': '❌'}.get(status, '?')
        norm = f"{r.get('norm', 0):.4f}" if 'norm' in r else '-'
        imgs = r.get('images', 0)
        print(f"{r['concept']:<20} {emoji} {status:<10} {norm:<10} {imgs:<10}")
    
    print(f"\n💾 Vecteurs sauvegardés dans: {args.save_path}/")
