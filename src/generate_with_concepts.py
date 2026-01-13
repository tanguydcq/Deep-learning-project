import torch
import argparse
import os
from torchvision.utils import save_image

from model_vector import get_model
from diffusion import Diffusion
from config import get_config_by_name


def load_concept_vector(concept_path, device):
    """Load a concept vector from file."""
    data = torch.load(concept_path, map_location=device)
    return data['vector'].to(device)


def generate_with_concepts(
    model,
    diffusion,
    concept_vectors,
    concept_scale,
    n_samples,
    device,
    output_path,
):
    """Generate images using concept vectors."""
    print(f"\nGenerating {n_samples} samples...")
    print(f"Using {len(concept_vectors)} concept(s) with scale {concept_scale}")
    
    model.eval()
    with torch.no_grad():
        # Combine concept vectors
        if len(concept_vectors) > 0:
            combined_vector = sum(concept_vectors)
            # Broadcast to batch size
            concept_batch = combined_vector.unsqueeze(0).expand(n_samples, -1)
            print(f"Combined concept vector norm: {combined_vector.norm().item():.4f}")
        else:
            # Zero vector (unconditional)
            concept_batch = torch.zeros(n_samples, model.concept_dim, device=device)
        
        # Scale the concept vector
        concept_batch = concept_batch * concept_scale
        
        # Sample from diffusion
        samples = diffusion.sample_with_concept(
            model,
            n=n_samples,
            concept_vector=concept_batch,
        )
    
    # Normalize to [0, 1]
    samples = (samples.clamp(-1, 1) + 1) / 2
    
    # Save images
    save_image(samples, output_path, nrow=int(n_samples ** 0.5))
    print(f"✓ Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate CryptoPunks with concept vectors')
    parser.add_argument('--config', type=str, default='cryptopunks1', 
                        help='Configuration name')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--concepts_dir', type=str, default='concepts',
                        help='Directory containing concept vectors')
    parser.add_argument('--accessories', type=str, nargs='+', default=[],
                        help='List of accessories to add (e.g., cap pipe cigarette)')
    parser.add_argument('--concept_scale', type=float, default=1.0,
                        help='Scale factor for concept vectors (0.5-2.0 recommended)')
    parser.add_argument('--n_samples', type=int, default=16,
                        help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='output/generated_concepts.png',
                        help='Output path for generated images')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print(f"Using seed: {args.seed}")
    
    # Load config
    print(f"Loading configuration: {args.config}")
    config = get_config_by_name(args.config)
    device = config.device
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = get_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Model loaded")
    
    # Create diffusion
    diffusion = Diffusion(
        noise_steps=config.T,
        img_size=config.img_size,
        img_channels=config.image_channels,
        device=device,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
    )
    
    # Load concept vectors
    concept_vectors = []
    if args.accessories:
        print(f"\nLoading concept vectors:")
        for acc in args.accessories:
            concept_name = f"acc_{acc.lower().replace(' ', '_')}"
            concept_path = os.path.join(args.concepts_dir, f"{concept_name}.pt")
            
            if os.path.exists(concept_path):
                vector = load_concept_vector(concept_path, device)
                concept_vectors.append(vector)
                print(f"  ✓ {acc}: {concept_path} (norm: {vector.norm().item():.4f})")
            else:
                print(f"  ✗ {acc}: Concept file not found at {concept_path}")
                print(f"    Available concepts:")
                if os.path.exists(args.concepts_dir):
                    available = [f.replace('.pt', '').replace('acc_', '') 
                                for f in os.listdir(args.concepts_dir) 
                                if f.endswith('.pt')]
                    print(f"    {', '.join(available)}")
    else:
        print("\nNo accessories specified, generating unconditional samples...")
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Generate
    generate_with_concepts(
        model=model,
        diffusion=diffusion,
        concept_vectors=concept_vectors,
        concept_scale=args.concept_scale,
        n_samples=args.n_samples,
        device=device,
        output_path=args.output,
    )
    
    print("\n" + "="*60)
    print("Generation complete!")
    print("="*60)
