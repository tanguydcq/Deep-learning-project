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


class CryptoPunksAccessoryDataset(Dataset):
    """
    Dataset for CryptoPunks filtered by specific accessory using subdataset files.
    """
    def __init__(self, images_dir, subdataset_path, transform=None):
        """
        Args:
            images_dir: Directory containing punk images
            subdataset_path: Path to subdataset JSON file (e.g., subdatasets/acc_cap.json)
            transform: Transform to apply to images
        """
        self.images_dir = images_dir
        self.transform = transform
        
        # Load subdataset info
        with open(subdataset_path, 'r') as f:
            subdataset = json.load(f)
        
        self.accessory_name = subdataset['accessory_name']
        self.punk_ids = subdataset['punk_ids']
        self.punk_ids_set = set(self.punk_ids)  # For fast lookup
        
        print(f"Found {len(self.punk_ids)} punks with '{self.accessory_name}'")
    
    def __len__(self):
        return len(self.punk_ids)
    
    def __getitem__(self, idx):
        punk_id = self.punk_ids[idx]
        img_path = os.path.join(self.images_dir, f"{punk_id}.png")
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image


class CryptoPunksContrastiveDataset(Dataset):
    """
    Contrastive dataset: returns pairs of (positive, negative) images.
    Positive = has the accessory, Negative = doesn't have the accessory.
    """
    def __init__(self, images_dir, subdataset_path, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        
        # Load subdataset info
        with open(subdataset_path, 'r') as f:
            subdataset = json.load(f)
        
        self.accessory_name = subdataset['accessory_name']
        self.positive_ids = subdataset['punk_ids']
        self.positive_set = set(self.positive_ids)
        
        # Get all existing punk IDs from the images directory
        all_existing_ids = []
        for fname in os.listdir(images_dir):
            if fname.endswith('.png'):
                try:
                    punk_id = int(fname.replace('.png', ''))
                    all_existing_ids.append(punk_id)
                except ValueError:
                    continue
        
        # Negative = existing punks that DON'T have this accessory
        self.negative_ids = [i for i in all_existing_ids if i not in self.positive_set]
        
        print(f"Contrastive dataset: {len(self.positive_ids)} positive, {len(self.negative_ids)} negative")
    
    def __len__(self):
        return len(self.positive_ids)
    
    def __getitem__(self, idx):
        # Positive sample
        pos_id = self.positive_ids[idx]
        pos_path = os.path.join(self.images_dir, f"{pos_id}.png")
        pos_img = Image.open(pos_path).convert('RGB')
        
        # Random negative sample
        neg_idx = torch.randint(0, len(self.negative_ids), (1,)).item()
        neg_id = self.negative_ids[neg_idx]
        neg_path = os.path.join(self.images_dir, f"{neg_id}.png")
        neg_img = Image.open(neg_path).convert('RGB')
        
        if self.transform:
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)
        
        return pos_img, neg_img


def train_concept_vector(
    model,
    diffusion,
    dataset,
    concept_name,
    device,
    epochs=50,
    lr=1e-2,
    batch_size=16,
    save_path="concept_vectors",
    patience=20,
    margin=0.01,
    train_scale=10.0,  # Scale used during TRAINING (higher = stronger signal)
):
    """
    Train a concept vector using CONTRASTIVE learning.
    
    The idea: the concept vector should help the model predict noise BETTER
    on images WITH the accessory than on images WITHOUT.
    
    Loss = MSE_positive - lambda * MSE_negative
    
    We want to MINIMIZE MSE on positives while the vector shouldn't help on negatives.
    """
    os.makedirs(save_path, exist_ok=True)

    # Freeze model COMPLETELY
    model.eval()
    model.requires_grad_(False)
    
    # Store original concept_scale and use higher scale for training
    original_scale = model.concept_scale
    model.concept_scale = train_scale
    print(f"🔧 Training with concept_scale={train_scale} (model default: {original_scale})")

    # Trainable concept vector
    c = nn.Parameter(
        torch.randn(model.concept_dim, device=device) * 0.1
    )

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

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"[Concept {concept_name}] Epoch {epoch}")
        epoch_losses = []
        epoch_pos_losses = []
        epoch_neg_losses = []

        for pos_images, neg_images in pbar:
            pos_images = pos_images.to(device)
            neg_images = neg_images.to(device)
            batch_size_actual = pos_images.size(0)

            # Same timestep for fair comparison
            t = diffusion.sample_timesteps(batch_size_actual).to(device)
            
            # Noise positive images
            x_t_pos, noise_pos = diffusion.noise_images(pos_images, t)
            # Noise negative images  
            x_t_neg, noise_neg = diffusion.noise_images(neg_images, t)

            # Broadcast concept vector
            c_batch = c.unsqueeze(0).expand(batch_size_actual, -1)

            # Predict noise WITH concept vector on BOTH
            pred_pos = model(x_t_pos, t, concept_vector=c_batch)
            pred_neg = model(x_t_neg, t, concept_vector=c_batch)

            # MSE losses
            loss_pos = mse(noise_pos, pred_pos)
            loss_neg = mse(noise_neg, pred_neg)
            
            # CONTRASTIVE LOSS:
            # Minimize loss on positives, but add penalty if loss_neg is too low
            # (we don't want the vector to help on negatives)
            # 
            # Alternative 1: loss = loss_pos - margin * loss_neg (maximize neg loss)
            # Alternative 2: loss = loss_pos / (loss_neg + eps)  (ratio)
            # Alternative 3: loss = loss_pos + relu(margin - (loss_neg - loss_pos))
            
            # We use: minimize pos, but penalize if pos < neg by less than margin
            # This encourages a GAP between positive and negative performance
            contrastive_loss = loss_pos + torch.relu(margin - (loss_neg - loss_pos))
            
            epoch_losses.append(contrastive_loss.item())
            epoch_pos_losses.append(loss_pos.item())
            epoch_neg_losses.append(loss_neg.item())

            optimizer.zero_grad()
            contrastive_loss.backward()
            optimizer.step()
            
            # Normalize vector to unit norm
            with torch.no_grad():
                c.data = c.data / c.data.norm()

            pbar.set_postfix(
                pos=f"{loss_pos.item():.4f}",
                neg=f"{loss_neg.item():.4f}",
                gap=f"{(loss_neg - loss_pos).item():.4f}"
            )

        # Epoch metrics
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_pos = sum(epoch_pos_losses) / len(epoch_pos_losses)
        avg_neg = sum(epoch_neg_losses) / len(epoch_neg_losses)
        gap = avg_neg - avg_pos
        
        print(f"[Epoch {epoch}] Loss={avg_loss:.6f} | MSE_pos={avg_pos:.4f} | MSE_neg={avg_neg:.4f} | Gap={gap:.4f}")
        
        # Early stopping on CONTRASTIVE loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_vector = c.detach().clone()
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"⚠️  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            print(f"📌 Best loss: {best_loss:.6f} at epoch {best_epoch}")
            c.data = best_vector
            break

    # Save learned vector
    torch.save(
        {
            "concept": concept_name,
            "vector": c.detach().cpu(),
            "train_scale": train_scale,
        },
        os.path.join(save_path, f"{concept_name}.pt"),
    )
    
    # Restore original scale
    model.concept_scale = original_scale

    print(f"✅ [Saved] Concept vector '{concept_name}' with final gap: {gap:.4f}")
    print(f"💡 Use with --concept_scale {train_scale} for generation")

    return c.detach()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='cryptopunks1', help='Configuration name')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--subdatasets_dir', type=str, default='subdatasets',
                        help='Directory containing subdataset JSON files')
    parser.add_argument('--accessories', type=str, nargs='+', 
                        default=['Cap', 'Pipe', 'Cigarette', 'Hoodie', 'Shades'],
                        help='List of accessories to train')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs per concept')
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--margin', type=float, default=0.005, help='Contrastive margin (gap between pos and neg)')
    parser.add_argument('--train_scale', type=float, default=10.0, help='Concept scale during training (higher = stronger signal)')
    parser.add_argument('--save_path', type=str, default='concepts', help='Directory to save concept vectors')
    cli_args = parser.parse_args()
    
    # Load config
    print(f"Loading configuration: {cli_args.config}")
    args = get_config_by_name(cli_args.config)
    device = args.device
    
    # Load model
    print(f"Loading model from {cli_args.checkpoint}")
    model = get_model(args).to(device)
    checkpoint = torch.load(cli_args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully")
    
    # Create diffusion
    diffusion = Diffusion(
        noise_steps=args.T,
        img_size=args.img_size,
        img_channels=args.image_channels,
        device=device,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    )
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Get images directory
    images_dir = args.dataset_path.replace('CRYPTOPUNKS', 'CRYPTOPUNKS_CLASSES') + '/images'
    
    # Train concept vectors for each accessory
    training_summary = []
    
    for accessory in cli_args.accessories:
        print(f"\n{'='*60}")
        print(f"Training concept vector for: {accessory}")
        print(f"{'='*60}")
        
        # Build subdataset path
        acc_filename = f"acc_{accessory.lower().replace(' ', '_')}.json"
        subdataset_path = os.path.join(cli_args.subdatasets_dir, acc_filename)
        
        if not os.path.exists(subdataset_path):
            print(f"⚠️  Subdataset not found: {subdataset_path}")
            print(f"   Run: python src/create_subdatasets.py")
            training_summary.append({
                'accessory': accessory,
                'status': 'NOT_FOUND',
                'num_images': 0
            })
            continue
        
        try:
            # Use CONTRASTIVE dataset
            dataset = CryptoPunksContrastiveDataset(
                images_dir=images_dir,
                subdataset_path=subdataset_path,
                transform=transform,
            )
            
            num_images = len(dataset)
            print(f"📊 Dataset size: {num_images} positive images (+ negatives)")
            
            if num_images == 0:
                print(f"Warning: No images found for '{accessory}', skipping...")
                training_summary.append({
                    'accessory': accessory,
                    'status': 'EMPTY',
                    'num_images': 0
                })
                continue
            
            concept_vector = train_concept_vector(
                model=model,
                diffusion=diffusion,
                dataset=dataset,
                concept_name=f"acc_{accessory.lower().replace(' ', '_')}",
                device=device,
                epochs=cli_args.epochs,
                lr=cli_args.lr,
                batch_size=cli_args.batch_size,
                save_path=cli_args.save_path,
                patience=cli_args.patience,
                margin=cli_args.margin,
                train_scale=cli_args.train_scale,
            )
            
            training_summary.append({
                'accessory': accessory,
                'status': 'SUCCESS',
                'num_images': num_images
            })
            
        except Exception as e:
            print(f"Error training '{accessory}': {e}")
            import traceback
            traceback.print_exc()
            training_summary.append({
                'accessory': accessory,
                'status': 'ERROR',
                'num_images': num_images if 'num_images' in locals() else 0
            })
            continue
    
    # Print summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"{'Accessory':<30} {'Status':<12} {'Images':<10}")
    print(f"{'-'*30} {'-'*12} {'-'*10}")
    
    total_images = 0
    successful = 0
    
    for item in training_summary:
        status_emoji = {
            'SUCCESS': '✓',
            'ERROR': '✗',
            'NOT_FOUND': '⚠️',
            'EMPTY': '⚠️'
        }.get(item['status'], '?')
        
        print(f"{item['accessory']:<30} {status_emoji} {item['status']:<10} {item['num_images']:<10}")
        
        if item['status'] == 'SUCCESS':
            total_images += item['num_images']
            successful += 1
    
    print(f"{'-'*30} {'-'*12} {'-'*10}")
    print(f"\n📊 Total accessories trained: {successful}/{len(training_summary)}")
    print(f"📊 Total images used: {total_images}")
    print(f"\n💾 Saved to: {cli_args.save_path}/")
    print(f"{'='*60}")
