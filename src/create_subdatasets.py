import os
import json
import argparse
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch
import torchvision.transforms as T


def create_subdataset_info(metadata_path, output_dir):
    """
    Create subdataset information for each accessory.
    Saves filtered metadata for each accessory.
    """
    print(f"Loading metadata from {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Get accessory mappings
    accessory_to_idx = metadata['accessory_to_idx']
    labels = metadata['labels']
    
    # Count punks per accessory
    accessory_counts = Counter()
    accessory_punks = {acc: [] for acc in accessory_to_idx.keys()}
    
    for punk_id, label_data in labels.items():
        accessory_vector = label_data['accessory_vector']
        for acc_name, acc_idx in accessory_to_idx.items():
            if accessory_vector[acc_idx] == 1:
                accessory_counts[acc_name] += 1
                accessory_punks[acc_name].append(punk_id)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save subdataset info for each accessory
    subdatasets_info = {}
    
    print(f"\n{'='*60}")
    print(f"Creating subdatasets for {len(accessory_to_idx)} accessories")
    print(f"{'='*60}\n")
    
    for acc_name in sorted(accessory_to_idx.keys()):
        count = accessory_counts[acc_name]
        punk_ids = accessory_punks[acc_name]
        
        if count == 0:
            print(f"⚠️  {acc_name:30s} - No punks found, skipping...")
            continue
        
        # Create subdataset info
        subdataset = {
            'accessory_name': acc_name,
            'count': count,
            'punk_ids': sorted([int(pid) for pid in punk_ids]),
        }
        
        # Save to JSON
        acc_filename = f"acc_{acc_name.lower().replace(' ', '_')}.json"
        acc_path = os.path.join(output_dir, acc_filename)
        
        with open(acc_path, 'w') as f:
            json.dump(subdataset, f, indent=2)
        
        subdatasets_info[acc_name] = {
            'count': count,
            'file': acc_path
        }
        
        print(f"✓ {acc_name:30s} - {count:4d} punks - {acc_path}")
    
    # Save summary
    summary = {
        'total_accessories': len([a for a in accessory_to_idx.keys() if accessory_counts[a] > 0]),
        'total_punks': len(labels),
        'subdatasets': subdatasets_info
    }
    
    summary_path = os.path.join(output_dir, 'subdatasets_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ Summary saved to: {summary_path}")
    print(f"{'='*60}")
    
    return subdatasets_info


def visualize_subdataset(images_dir, subdataset_path, output_path, n_samples=16):
    """
    Visualize a sample of images from a subdataset.
    """
    with open(subdataset_path, 'r') as f:
        subdataset = json.load(f)
    
    accessory_name = subdataset['accessory_name']
    punk_ids = subdataset['punk_ids'][:n_samples]
    
    print(f"\nVisualizing '{accessory_name}' - {len(punk_ids)} samples")
    
    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
    ])
    
    images = []
    for punk_id in punk_ids:
        img_path = os.path.join(images_dir, f"{punk_id}.png")
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            images.append(transform(img))
    
    if len(images) == 0:
        print(f"⚠️  No images found for '{accessory_name}'")
        return
    
    # Create grid
    grid = make_grid(images, nrow=4, padding=2, normalize=True)
    
    # Save visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(f"{accessory_name} ({len(images)} samples)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create CryptoPunks subdatasets by accessory')
    parser.add_argument('--metadata_path', type=str, 
                        default='data/CRYPTOPUNKS_CLASSES/metadata.json',
                        help='Path to metadata.json')
    parser.add_argument('--images_dir', type=str,
                        default='data/CRYPTOPUNKS_CLASSES/images',
                        help='Directory containing punk images')
    parser.add_argument('--output_dir', type=str, default='subdatasets',
                        help='Directory to save subdataset info')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualizations for each subdataset')
    parser.add_argument('--visualize_accessories', type=str, nargs='+',
                        help='Specific accessories to visualize (default: all)')
    parser.add_argument('--n_samples', type=int, default=16,
                        help='Number of samples to show in visualization')
    args = parser.parse_args()
    
    # Create subdatasets
    subdatasets_info = create_subdataset_info(args.metadata_path, args.output_dir)
    
    # Visualize if requested
    if args.visualize:
        viz_dir = os.path.join(args.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("Creating visualizations...")
        print(f"{'='*60}")
        
        accessories_to_viz = args.visualize_accessories
        if accessories_to_viz is None:
            accessories_to_viz = list(subdatasets_info.keys())
        
        for acc_name in accessories_to_viz:
            if acc_name not in subdatasets_info:
                print(f"⚠️  Accessory '{acc_name}' not found, skipping...")
                continue
            
            acc_filename = f"acc_{acc_name.lower().replace(' ', '_')}.json"
            subdataset_path = os.path.join(args.output_dir, acc_filename)
            
            viz_filename = f"viz_{acc_name.lower().replace(' ', '_')}.png"
            viz_path = os.path.join(viz_dir, viz_filename)
            
            visualize_subdataset(
                args.images_dir,
                subdataset_path,
                viz_path,
                n_samples=args.n_samples
            )
        
        print(f"\n✓ All visualizations saved to: {viz_dir}")
    
    print(f"\n{'='*60}")
    print("Subdatasets creation complete!")
    print(f"Total subdatasets: {len(subdatasets_info)}")
    print(f"Location: {args.output_dir}")
    print(f"{'='*60}")
