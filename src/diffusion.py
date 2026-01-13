import torch
import logging
import torch.nn as nn
from tqdm import tqdm

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, img_channels=3, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.img_channels = img_channels
        self.device = device
        
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        ε = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * ε, ε
    
    def sample_timesteps(self, n):
        # Randomly sample timesteps
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    def sample(self, model, n, save_gif=False, labels=None, type_idx=None, accessory_vector=None):
        """
        Sample n images from the model.
        
        Args:
            model: The diffusion model
            n: Number of images to sample
            save_gif: Whether to save intermediate frames
            labels: Class labels for simple class conditioning (MNIST)
            type_idx: Type indices for multi-attribute conditioning (CryptoPunks)
            accessory_vector: Accessory vectors for multi-attribute conditioning (CryptoPunks)
        """
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, self.img_channels, self.img_size, self.img_size)).to(self.device)
            frames = []
            for i in tqdm(reversed(range(0, self.noise_steps)), position=0):
                # We generate z ~ N(0, I) if i > 1 else z = 0
                z = torch.randn_like(x) if i > 0 else torch.zeros_like(x)

                # create time batch for all images
                t = (torch.ones(n) * i).long().to(self.device)
                
                # Predict noise based on model type
                if type_idx is not None and accessory_vector is not None and hasattr(model, 'attr_embedding') and model.attr_embedding is not None:
                    # Multi-attribute conditioning (CryptoPunks)
                    predicted_noise = model(x, t, type_idx=type_idx, accessory_vector=accessory_vector)
                elif labels is not None and hasattr(model, 'num_classes') and model.num_classes is not None:
                    # Simple class conditioning (MNIST)
                    predicted_noise = model(x, t, labels)
                else:
                    # Unconditional
                    predicted_noise = model(x, t)

                # Reconstruction formula
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                coef1 = 1 / torch.sqrt(alpha)
                coef2 = beta / torch.sqrt(1 - alpha_hat)
                sigma = torch.sqrt(beta)

                x = coef1 * (x - coef2 * predicted_noise) + sigma * z
                
                if save_gif:
                    # Normalize to [0, 1] for visualization
                    frame = (x.clamp(-1, 1) + 1) / 2
                    frame = (frame * 255).type(torch.uint8).cpu()
                    frames.append(frame)

        model.train()
        # Normalize to [0, 1]
        x = (x.clamp(-1, 1) + 1) / 2
        
        if save_gif:
            return x, frames
        return x

    def sample_cfg(self, model, n, guidance_scale=3.0, save_gif=False, 
                   labels=None, type_idx=None, accessory_vector=None, concept_vector=None):
        """
        Sample n images from the model using Classifier-Free Guidance (CFG).
        
        CFG improves conditional generation by combining conditional and unconditional
        predictions: noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        
        Args:
            model: The diffusion model
            n: Number of images to sample
            guidance_scale: How strongly to follow conditioning (1.0 = no guidance, 3.0-7.0 = typical)
            save_gif: Whether to save intermediate frames
            labels: Class labels for simple class conditioning (MNIST)
            type_idx: Type indices for multi-attribute conditioning (CryptoPunks)
            accessory_vector: Accessory vectors for multi-attribute conditioning (CryptoPunks)
            concept_vector: Optional concept vector for steering generation
        """
        logging.info(f"Sampling {n} new images with CFG (guidance_scale={guidance_scale})....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, self.img_channels, self.img_size, self.img_size)).to(self.device)
            frames = []
            
            for i in tqdm(reversed(range(0, self.noise_steps)), position=0):
                # We generate z ~ N(0, I) if i > 1 else z = 0
                z = torch.randn_like(x) if i > 0 else torch.zeros_like(x)

                # Create time batch for all images
                t = (torch.ones(n) * i).long().to(self.device)
                
                # === CONDITIONAL PREDICTION ===
                if type_idx is not None and accessory_vector is not None and hasattr(model, 'attr_embedding') and model.attr_embedding is not None:
                    # Multi-attribute conditioning (CryptoPunks)
                    noise_cond = model(x, t, type_idx=type_idx, accessory_vector=accessory_vector, concept_vector=concept_vector)
                elif labels is not None and hasattr(model, 'num_classes') and model.num_classes is not None:
                    # Simple class conditioning (MNIST)
                    noise_cond = model(x, t, labels)
                else:
                    # No conditioning available, fall back to regular sampling
                    noise_cond = model(x, t)
                    noise_uncond = noise_cond  # No CFG possible
                    predicted_noise = noise_cond
                    # Skip CFG computation
                    alpha = self.alpha[t][:, None, None, None]
                    alpha_hat = self.alpha_hat[t][:, None, None, None]
                    beta = self.beta[t][:, None, None, None]
                    coef1 = 1 / torch.sqrt(alpha)
                    coef2 = beta / torch.sqrt(1 - alpha_hat)
                    sigma = torch.sqrt(beta)
                    x = coef1 * (x - coef2 * predicted_noise) + sigma * z
                    if save_gif:
                        frame = (x.clamp(-1, 1) + 1) / 2
                        frame = (frame * 255).type(torch.uint8).cpu()
                        frames.append(frame)
                    continue
                
                # === UNCONDITIONAL PREDICTION ===
                if hasattr(model, 'attr_embedding') and model.attr_embedding is not None:
                    # Multi-attribute model: pass None for unconditional
                    noise_uncond = model(x, t, type_idx=None, accessory_vector=None, concept_vector=concept_vector)
                elif hasattr(model, 'num_classes') and model.num_classes is not None:
                    # Class-conditioned model: pass None for unconditional
                    noise_uncond = model(x, t, y=None)
                else:
                    noise_uncond = model(x, t)
                
                # === CFG COMBINATION ===
                # noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                predicted_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

                # Reconstruction formula
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                coef1 = 1 / torch.sqrt(alpha)
                coef2 = beta / torch.sqrt(1 - alpha_hat)
                sigma = torch.sqrt(beta)

                x = coef1 * (x - coef2 * predicted_noise) + sigma * z
                
                if save_gif:
                    # Normalize to [0, 1] for visualization
                    frame = (x.clamp(-1, 1) + 1) / 2
                    frame = (frame * 255).type(torch.uint8).cpu()
                    frames.append(frame)

        model.train()
        # Normalize to [0, 1]
        x = (x.clamp(-1, 1) + 1) / 2
        
        if save_gif:
            return x, frames
        return x


    def sample_with_concept(self, model, n, concept_vector=None):
        """
        Sample n images from the model using concept vectors.
        
        Args:
            model: The diffusion model
            n: Number of images to sample
            concept_vector: (n, concept_dim) tensor for concept-based generation
        """
        logging.info(f"Sampling {n} new images with concept vectors...")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, self.img_channels, self.img_size, self.img_size)).to(self.device)
            
            for i in tqdm(reversed(range(0, self.noise_steps)), position=0, desc="Sampling"):
                z = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
                t = (torch.ones(n) * i).long().to(self.device)
                
                predicted_noise = model(x, t, concept_vector=concept_vector)
                
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                coef1 = 1 / torch.sqrt(alpha)
                coef2 = beta / torch.sqrt(1 - alpha_hat)
                sigma = torch.sqrt(beta)
                
                x = coef1 * (x - coef2 * predicted_noise) + sigma * z
        
        model.train()
        return x
