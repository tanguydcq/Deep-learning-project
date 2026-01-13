"""
Entraînement DDPM avec conditioning sur les accessoires.

Le modèle apprend dès le départ à utiliser les vecteurs concepts.
Chaque image est associée à un vecteur multi-hot d'accessoires.
CFG dropout permet la génération guidée à l'inférence.
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
import argparse

from model_conditioned import UNetConditioned, get_model
from diffusion import Diffusion
from config import get_config_by_name


class CryptoPunksConditionedDataset(Dataset):
    """
    Dataset CryptoPunks avec labels d'accessoires.
    Retourne (image, accessory_multi_hot).
    """
    def __init__(self, images_dir, metadata_path, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        
        # Charger metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Le format est: metadata['labels'][punk_id] = {type, accessories, accessory_vector}
        self.labels = metadata['labels']
        self.accessory_list = list(metadata['accessory_to_idx'].keys())
        self.accessory_to_idx = metadata['accessory_to_idx']
        self.num_accessories = metadata['num_accessories']
        
        # Filtrer aux images existantes
        self.valid_ids = []
        for punk_id in self.labels.keys():
            img_path = os.path.join(images_dir, f"{punk_id}.png")
            if os.path.exists(img_path):
                self.valid_ids.append(punk_id)
        
        print(f"📊 Dataset: {len(self.valid_ids)} images, {self.num_accessories} accessoires uniques")
    
    def __len__(self):
        return len(self.valid_ids)
    
    def __getitem__(self, idx):
        punk_id = self.valid_ids[idx]
        punk_data = self.labels[punk_id]
        
        # Charger image
        img_path = os.path.join(self.images_dir, f"{punk_id}.png")
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Accessory vector déjà présent dans le metadata
        accessory_vec = torch.tensor(punk_data['accessory_vector'], dtype=torch.float32)
        
        return image, accessory_vec


def train(config, args):
    device = config.device
    
    # ========== Dataset ==========
    transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = CryptoPunksConditionedDataset(
        images_dir=args.images_dir,
        metadata_path=args.metadata_path,
        transform=transform,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    # ========== Sauvegarder mapping accessoires ==========
    os.makedirs(args.save_dir, exist_ok=True)
    mapping = {
        'accessory_list': dataset.accessory_list,
        'accessory_to_idx': dataset.accessory_to_idx,
        'num_accessories': dataset.num_accessories,
    }
    with open(os.path.join(args.save_dir, 'accessory_mapping.json'), 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"💾 Accessory mapping saved ({dataset.num_accessories} accessories)")
    
    # ========== Model ==========
    model = get_model(dataset.num_accessories, config).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"🔧 Model parameters: {num_params:,}")
    print(f"🔧 Concept dim: {model.concept_dim}")
    print(f"🔧 Concept scale (α): {model.concept_scale}")
    print(f"🔧 CFG dropout: {model.cfg_dropout}")
    
    # ========== Diffusion ==========
    diffusion = Diffusion(
        noise_steps=config.T,
        img_size=config.img_size,
        img_channels=config.image_channels,
        device=device,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
    )
    
    # ========== Optimizer ==========
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    mse = nn.MSELoss()
    
    # ========== Training Loop ==========
    best_loss = float('inf')
    
    print(f"\n{'='*60}")
    print(f"Training DDPM with accessory conditioning")
    print(f"Epochs: {config.epochs}, Batch size: {config.batch_size}, LR: {config.lr}")
    print(f"{'='*60}\n")
    
    for epoch in range(config.epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        epoch_losses = []
        
        for images, accessory_labels in pbar:
            images = images.to(device)
            accessory_labels = accessory_labels.to(device)
            
            # Sample timesteps
            t = diffusion.sample_timesteps(images.size(0)).to(device)
            
            # Forward diffusion
            x_t, noise = diffusion.noise_images(images, t)
            
            # Predict noise with accessory conditioning
            # CFG dropout is applied inside the model during training
            predicted_noise = model(x_t, t, accessory_labels=accessory_labels)
            
            # DDPM Loss
            loss = mse(noise, predicted_noise)
            epoch_losses.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix(MSE=f"{loss.item():.5f}")
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"[Epoch {epoch:3d}] Avg MSE: {avg_loss:.6f}")
        
        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'num_accessories': dataset.num_accessories,
                'config': {
                    'concept_dim': model.concept_dim,
                    'concept_scale': model.concept_scale,
                    'time_dim': model.time_dim,
                    'cfg_dropout': model.cfg_dropout,
                }
            }, os.path.join(args.save_dir, 'ckpt_best.pt'))
            print(f"   💾 Saved best checkpoint (loss: {avg_loss:.6f})")
        
        # Regular checkpoint every N epochs
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'num_accessories': dataset.num_accessories,
            }, os.path.join(args.save_dir, f'ckpt_epoch{epoch:03d}.pt'))
    
    # Final checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': config.epochs - 1,
        'loss': avg_loss,
        'num_accessories': dataset.num_accessories,
        'config': {
            'concept_dim': model.concept_dim,
            'concept_scale': model.concept_scale,
            'time_dim': model.time_dim,
            'cfg_dropout': model.cfg_dropout,
        }
    }, os.path.join(args.save_dir, 'ckpt_final.pt'))
    
    print(f"\n{'='*60}")
    print(f"✅ Training complete!")
    print(f"   Best loss: {best_loss:.6f}")
    print(f"   Saved to: {args.save_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DDPM with accessory conditioning")
    parser.add_argument('--config', type=str, default='cryptopunks1',
                        help='Configuration name')
    parser.add_argument('--images_dir', type=str, 
                        default='./data/CRYPTOPUNKS_CLASSES/images',
                        help='Directory containing punk images')
    parser.add_argument('--metadata_path', type=str,
                        default='./data/CRYPTOPUNKS_CLASSES/metadata.json',
                        help='Path to metadata JSON')
    parser.add_argument('--save_dir', type=str,
                        default='./models/CRYPTOPUNKS_CONDITIONED',
                        help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Load config
    config = get_config_by_name(args.config)
    
    # Override if specified
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.lr = args.lr
    
    print(f"📋 Config: {args.config}")
    print(f"   Device: {config.device}")
    print(f"   Image size: {config.img_size}")
    print(f"   T (noise steps): {config.T}")
    
    train(config, args)
