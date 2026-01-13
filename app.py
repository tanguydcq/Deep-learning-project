"""
Interface Streamlit pour générer des CryptoPunks avec accessoires.
"""

import streamlit as st
import torch
import os
import json
from PIL import Image
import torchvision
from tqdm import tqdm
import sys

sys.path.insert(0, 'src')
from model_conditioned import UNetConditioned
from diffusion import Diffusion


@st.cache_resource
def load_model(checkpoint_path):
    """Charger le modèle (cached)."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model_dir = os.path.dirname(checkpoint_path)
    mapping_path = os.path.join(model_dir, 'accessory_mapping.json')
    
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    
    ckpt_config = checkpoint.get('config', {})
    
    model = UNetConditioned(
        c_in=3,
        c_out=3,
        time_dim=ckpt_config.get('time_dim', 64),
        num_accessories=checkpoint['num_accessories'],
        concept_dim=ckpt_config.get('concept_dim', 512),
        concept_scale=ckpt_config.get('concept_scale', 1.0),
        cfg_dropout=0.0,
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, mapping, device


def create_accessory_vector(selected_accessories, mapping, device):
    """Créer vecteur multi-hot."""
    accessory_to_idx = mapping['accessory_to_idx']
    num_accessories = mapping['num_accessories']
    
    vector = torch.zeros(num_accessories, device=device)
    
    for name in selected_accessories:
        if name in accessory_to_idx:
            vector[accessory_to_idx[name]] = 1.0
    
    return vector


def sample_cfg(model, diffusion, accessory_vector, n_samples, cfg_scale, device, progress_bar):
    """Sampling avec CFG et progress bar Streamlit."""
    model.eval()
    
    with torch.no_grad():
        x = torch.randn(n_samples, 3, diffusion.img_size, diffusion.img_size).to(device)
        
        if accessory_vector is not None:
            cond_labels = accessory_vector.unsqueeze(0).expand(n_samples, -1)
        else:
            cond_labels = torch.zeros(n_samples, model.num_accessories, device=device)
        
        uncond_labels = torch.zeros(n_samples, model.num_accessories, device=device)
        
        for i, t in enumerate(reversed(range(diffusion.noise_steps))):
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
            
            noise_cond = model(x, t_batch, accessory_labels=cond_labels)
            noise_uncond = model(x, t_batch, accessory_labels=uncond_labels)
            
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
            
            # Update progress
            progress_bar.progress((i + 1) / diffusion.noise_steps)
    
    x = (x.clamp(-1, 1) + 1) / 2
    return x


def main():
    st.set_page_config(
        page_title="CryptoPunks Generator",
        page_icon="🎨",
        layout="wide"
    )
    
    st.title("🎨 CryptoPunks Generator")
    st.markdown("Génère des CryptoPunks avec les accessoires de ton choix!")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Checkpoint path
    checkpoint_path = st.sidebar.text_input(
        "Checkpoint",
        value="models/CRYPTOPUNKS_CONDITIONED/ckpt_best.pt"
    )
    
    if not os.path.exists(checkpoint_path):
        st.error(f"❌ Checkpoint non trouvé: {checkpoint_path}")
        st.info("Lance d'abord l'entraînement avec: `python src/train_ddpm_conditioned.py --epochs 50`")
        return
    
    # Load model
    with st.spinner("Chargement du modèle..."):
        model, mapping, device = load_model(checkpoint_path)
    
    st.sidebar.success(f"✅ Modèle chargé ({device})")
    st.sidebar.info(f"📊 {mapping['num_accessories']} accessoires disponibles")
    
    # Parameters
    st.sidebar.header("🎛️ Paramètres")
    
    cfg_scale = st.sidebar.slider(
        "CFG Scale",
        min_value=1.0,
        max_value=10.0,
        value=3.0,
        step=0.5,
        help="Plus élevé = plus fidèle aux accessoires, mais moins de diversité"
    )
    
    n_images = st.sidebar.selectbox(
        "Nombre d'images",
        options=[1, 4, 9, 16],
        index=1
    )
    
    seed = st.sidebar.number_input(
        "Seed (optionnel)",
        min_value=-1,
        max_value=999999,
        value=-1,
        help="-1 = aléatoire"
    )
    
    # Main area - Accessory selection
    st.header("👕 Sélection des accessoires")
    
    # Group accessories by category
    accessory_list = mapping['accessory_list']
    
    # Create columns for selection
    col1, col2, col3 = st.columns(3)
    
    # Split accessories into 3 columns
    n = len(accessory_list)
    third = n // 3
    
    selected = []
    
    with col1:
        st.subheader("A-H")
        for acc in sorted([a for a in accessory_list if a[0].upper() <= 'H']):
            if st.checkbox(acc, key=f"acc_{acc}"):
                selected.append(acc)
    
    with col2:
        st.subheader("I-P")
        for acc in sorted([a for a in accessory_list if 'H' < a[0].upper() <= 'P']):
            if st.checkbox(acc, key=f"acc_{acc}"):
                selected.append(acc)
    
    with col3:
        st.subheader("Q-Z")
        for acc in sorted([a for a in accessory_list if a[0].upper() > 'P']):
            if st.checkbox(acc, key=f"acc_{acc}"):
                selected.append(acc)
    
    # Show selected
    st.markdown("---")
    if selected:
        st.info(f"🎯 **Accessoires sélectionnés:** {', '.join(selected)}")
    else:
        st.warning("⚠️ Aucun accessoire sélectionné (génération unconditionnelle)")
    
    # Generate button
    st.markdown("---")
    
    if st.button("🚀 Générer!", type="primary", use_container_width=True):
        # Set seed
        if seed >= 0:
            torch.manual_seed(seed)
            st.info(f"🎲 Seed: {seed}")
        
        # Create diffusion
        diffusion = Diffusion(
            noise_steps=1000,
            img_size=32,
            img_channels=3,
            device=device,
        )
        
        # Create accessory vector
        if selected:
            accessory_vector = create_accessory_vector(selected, mapping, device)
        else:
            accessory_vector = None
        
        # Generate with progress bar
        st.subheader("🖼️ Génération en cours...")
        progress_bar = st.progress(0)
        
        images = sample_cfg(
            model, diffusion, accessory_vector,
            n_images, cfg_scale, device, progress_bar
        )
        
        progress_bar.empty()
        
        # Display results
        st.subheader("✨ Résultats")
        
        # Convert to grid
        nrow = int(n_images ** 0.5)
        if nrow * nrow < n_images:
            nrow += 1
        
        grid = torchvision.utils.make_grid(images, nrow=nrow, padding=2)
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        
        # Display
        st.image(grid_np, use_container_width=True)
        
        # Download button
        # Convert to PIL for download
        grid_pil = torchvision.transforms.ToPILImage()(grid)
        
        import io
        buf = io.BytesIO()
        grid_pil.save(buf, format='PNG')
        
        st.download_button(
            label="💾 Télécharger",
            data=buf.getvalue(),
            file_name="cryptopunks_generated.png",
            mime="image/png"
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        🧠 Powered by DDPM with Classifier-Free Guidance
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
