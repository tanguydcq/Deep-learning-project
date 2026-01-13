import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
import imageio
import numpy as np

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray((ndarr * 255).astype('uint8'))
    im.save(path)

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()

def make_gif(frames, path):
    # Frames shape: (T, N, 3, H, W)
    formatted_frames = []
    for frame_batch in frames:
        # frame_batch is (N, 3, H, W)
        # Select first 16 or less
        n_imgs = min(16, frame_batch.shape[0])
        subset = frame_batch[:n_imgs] 
        # Make grid using torchvision for better layout (default is nrow=8, we can force nrow=4 for 16 images)
        grid = torchvision.utils.make_grid(subset, nrow=int(np.sqrt(n_imgs)), padding=2, normalize=False)
        
        # Grid is (3, H_grid, W_grid). Convert to numpy (H, W, 3)
        grid = grid.permute(1, 2, 0).cpu().numpy()
        
        # Scale to uint8 if not already
        if grid.dtype != np.uint8:
             if grid.max() <= 1.0:
                 grid = (grid * 255).astype(np.uint8)
             else:
                 grid = grid.astype(np.uint8)
        
        formatted_frames.append(grid)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, formatted_frames, fps=20, loop=0)
