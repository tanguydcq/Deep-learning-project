# Diffusion Model - DDPM Implementation

Une implémentation de **Denoising Diffusion Probabilistic Models (DDPM)** en PyTorch pour la génération d'images sur MNIST et CryptoPunks.

## 📋 Description

Ce projet implémente un modèle de diffusion capable de générer des images à partir de bruit aléatoire. Il inclut :

- Un modèle simple (SimpleUNet) pour MNIST
- Un modèle complexe (UNet avec attention) pour CryptoPunks
- Multiple configurations d'entraînement
- Scripts de training et d'inférence
- Génération de GIFs montrant le processus de diffusion/débruitage

## 🚀 Installation

### Prérequis

- Python 3.8+
- CUDA (optionnel, recommandé pour l'entraînement)

### Installation des dépendances

```bash
pip install torch torchvision tqdm tensorboard imageio numpy pillow matplotlib
```

## 📁 Structure du projet

```
Diffusion/
├── config.py              # Configurations d'entraînement
├── dataset.py             # Chargement des datasets
├── diffusion.py           # Processus de diffusion/débruitage
├── model.py               # Architectures UNet
├── train.py               # Script d'entraînement
├── infer.py               # Script d'inférence
├── utils.py               # Fonctions utilitaires
├── download_cryptopunks.py # Script pour télécharger CryptoPunks
├── training.ipynb         # Notebook Jupyter pour expérimentation
└── README.md              # Ce fichier
```

## 🎯 Utilisation

### Entraînement

Pour entraîner un modèle avec une configuration spécifique :

```bash
# Configuration 1 (baseline) pour MNIST
python train.py --config config1_mnist

# Configuration 2 (fast prototyping) pour MNIST
python train.py --config config2_mnist

# Configuration 3 (high precision) pour MNIST
python train.py --config config3_mnist

# Pour CryptoPunks
python train.py --config config1_cryptopunks
```

Les modèles sont sauvegardés dans `models/DATASET/CONFIG_NAME/`.

### Inférence

Pour générer des images et des GIFs avec un modèle entraîné :

```bash
# Génération avec MNIST
python infer.py --config config1_mnist

# Génération avec CryptoPunks
python infer.py --config config1_cryptopunks
```

Les résultats sont sauvegardés dans `results/DATASET/CONFIG_NAME/` :

- `noise.gif` : Visualisation du processus de bruitage
- `sampling.gif` : Visualisation du processus de débruitage
- `sampling.jpg` : Images générées finales

### Interface Web (Streamlit)

Une interface Streamlit permet de lancer rapidement des entraînements, visualiser les checkpoints, et générer des images (y compris le générateur CryptoPunk intégré).

Pour lancer le dashboard (Windows PowerShell) :

```powershell
# Activer l'environnement virtuel
.venv\Scripts\Activate.ps1

# Lancer Streamlit
streamlit run streamlit_dashboard.py
```

Ou depuis l'invite de commandes (cmd.exe) :

```bat
:: Activer l'environnement virtuel
.venv\Scripts\activate.bat

:: Lancer Streamlit
streamlit run streamlit_dashboard.py
```

Après démarrage Streamlit, l'application est disponible localement (par défaut) sur http://localhost:8501. Utilise l'onglet "Training" pour démarrer un entraînement et "Inference" pour générer des images.

### Télécharger CryptoPunks

```bash
python download_cryptopunks.py
```

## ⚙️ Configurations disponibles

### MNIST

| Config        | T (steps) | Epochs | LR   | Beta Schedule | Description            |
| ------------- | --------- | ------ | ---- | ------------- | ---------------------- |
| config1_mnist | 1000      | 100    | 3e-4 | 1e-4 → 0.02   | Baseline standard DDPM |
| config2_mnist | 300       | 100    | 3e-4 | 1e-4 → 0.02   | Prototypage rapide     |
| config3_mnist | 1000      | 100    | 2e-4 | 1e-4 → 0.01   | Haute précision        |

### CryptoPunks

| Config              | T (steps) | Epochs | LR   | Beta Schedule | Description            |
| ------------------- | --------- | ------ | ---- | ------------- | ---------------------- |
| config1_cryptopunks | 1000      | 100    | 3e-4 | 1e-4 → 0.02   | Baseline standard DDPM |

## 🏗️ Architecture

### SimpleUNet (MNIST)

- Architecture légère pour images 16x16 en niveaux de gris
- Encodeur-décodeur avec skip connections
- Time embedding simple

### UNet (CryptoPunks)

- Architecture complète avec self-attention
- Encodeur-décodeur avec skip connections
- Time embedding positionnel
- Modules d'attention multi-têtes

## 📊 Monitoring

L'entraînement est monitoré avec TensorBoard :

```bash
tensorboard --logdir runs/
```

Métriques suivies :

- MSE Loss
- Gradient Norm
- Learning Rate
- Images générées à chaque époque

## 📝 Notes

### Paramètres importants

- **T (noise_steps)** : Nombre de pas de diffusion. Plus élevé = meilleure qualité mais plus lent
- **Beta schedule** : Contrôle la vitesse d'ajout de bruit (beta_start → beta_end)
- **Batch size** : 128 pour MNIST, 64 pour CryptoPunks (ajuster selon la VRAM)
- **Image size** : 16x16 pour MNIST, 32x32 pour CryptoPunks

### Résultats attendus

- **MNIST** : Génération de chiffres réalistes après ~50 époques
- **CryptoPunks** : Génération de portraits pixelisés après ~100 époques

## 🔧 Personnalisation

Pour créer votre propre configuration, éditez `config.py` :

```python
custom_config = {
    "dataset_name": "MNIST",
    "epochs": 100,
    "lr": 3e-4,
    "T": 500,
    "batch_size": 128,
    "beta_start": 1e-4,
    "beta_end": 0.02,
}
```

## 📚 Références

- [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239)
- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)

## 📄 Licence

Voir le fichier [LICENSE](LICENSE) pour plus de détails.
