import torch
import os

# Force l'utilisation de CUDA même si l'architecture n'est pas officiellement supportée
# Pour RTX 5070 (sm_120 / Blackwell)
os.environ['TORCH_CUDA_ARCH_LIST'] = '5.0;6.0;6.1;7.0;7.5;8.0;8.6;9.0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class Config:
    def __init__(
        self,
        dataset_name="MNIST",
        epochs=50,
        lr=3e-4,
        T=500,
        batch_size=128,
        beta_start=1e-4,
        beta_end=0.02,
        num_classes=10,
        num_types=None,
        num_accessories=None,
        cfg_dropout=0.1,
        cfg_scale=3.0,
        concept_dim=512,
        concept_scale=1.0,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.num_types = num_types
        self.num_accessories = num_accessories
        
        # Classifier-Free Guidance parameters
        self.cfg_dropout = cfg_dropout  # Probability of dropping conditioning during training
        self.cfg_scale = cfg_scale      # Guidance scale at inference (1.0 = no guidance)
        
        # Concept vector parameters
        self.concept_dim = concept_dim      # Dimension of concept vectors
        self.concept_scale = concept_scale  # Scale factor for concept vectors

        if dataset_name == "MNIST":
            self.img_size = 16
            self.image_channels = 1
        elif dataset_name == "CRYPTOPUNKS":
            self.img_size = 32
            self.image_channels = 3
        elif dataset_name == "CRYPTOPUNKS_CLASSES":
            self.img_size = 32
            self.image_channels = 3

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Paths
        self.dataset_path = f"./data/{self.dataset_name}"
        self.models_path = f"./models/{self.dataset_name}"
        self.results_path = f"./results/{self.dataset_name}"

        # Model specific
        if dataset_name == "MNIST":
            self.time_emb_dim = 32
            self.start_channels = 128
        elif dataset_name in ("CRYPTOPUNKS", "CRYPTOPUNKS_CLASSES"):
            self.time_emb_dim = 64
            self.start_channels = 128


# Config 1: Standard DDPM (Baseline) - AVEC conditionnement par classes + CFG
# T=1000 is standard for good quality. Beta schedule 1e-4 to 0.02 is classic.
# num_classes=10 pour MNIST (chiffres 0-9)
# CFG: cfg_dropout=0.1 (10% unconditional training), cfg_scale=3.0 (inference guidance)
config1_mnist_classes = {
    "dataset_name": "MNIST",
    "epochs": 100,
    "lr": 6e-4,
    "T": 1000,
    "batch_size": 128,
    "beta_start": 1e-4,
    "beta_end": 0.02,
    "num_classes": 10,
    "cfg_dropout": 0.1,  # 10% unconditional training for CFG
    "cfg_scale": 3.0,    # Guidance scale at inference (1.0 = no guidance, 3.0-7.0 typical)
}

# Config 1 (No Classes): SANS conditionnement par classes
# Génération non-conditionnelle classique
config1_no_classes = {
    "dataset_name": "MNIST",
    "epochs": 100,
    "lr": 3e-4,
    "T": 1000,
    "batch_size": 128,
    "beta_start": 1e-4,
    "beta_end": 0.02,
    "num_classes": None,
}

# Config 2: Fast Prototyping (Faster training/sampling)
# T=300 is much faster but might be slightly noisier/less detailed.
# Useful for quick debugging.
config2_mnist = {
    "dataset_name": "MNIST",
    "epochs": 100,
    "lr": 3e-4,
    "T": 300,
    "batch_size": 128,
    "beta_start": 1e-4,
    "beta_end": 0.02,
    "num_classes": None,
}


# Config 3: High Precision / Smooth Schedule
# Lower beta_end (0.01) creates a "smoother" noise schedule,
# sometimes resulting in higher quality but might need more steps to converge perfectly.
# Config 3: High Precision / Smooth Schedule
# Lower beta_end (0.01) creates a "smoother" noise schedule,
# sometimes resulting in higher quality but might be slightly noisier/less detailed.
config3_mnist = {
    "dataset_name": "MNIST",
    "epochs": 100,
    "lr": 2e-4,  # Lower LR for more stable convergence
    "T": 1000,
    "batch_size": 128,
    "beta_start": 1e-4,
    "beta_end": 0.01,  # Slower noise accumulation
    "num_classes": None,
}


# Config 1: Standard DDPM (Baseline)
# T=1000 is standard for good quality. Beta schedule 1e-4 to 0.02 is classic.
config1_cryptopunks = {
    "dataset_name": "CRYPTOPUNKS",
    "epochs": 100,
    "lr": 3e-4,
    "T": 1000,
    "batch_size": 64,
    "beta_start": 1e-4,
    "beta_end": 0.02,
}

# Config 2: Fast Prototyping (Faster training/sampling)
# T=300 is much faster but might be slightly noisier/less detailed.
# Useful for quick debugging.
config2_cryptopunks = {
    "dataset_name": "CRYPTOPUNKS",
    "epochs": 100,
    "lr": 3e-4,
    "T": 300,
    "batch_size": 64,
    "beta_start": 1e-4,
    "beta_end": 0.02,
}

# Config 3: High Precision / Smooth Schedule
# Lower beta_end (0.01) creates a "smoother" noise schedule,
# sometimes resulting in higher quality but might need more steps to converge perfectly.
config3_cryptopunks = {
    "dataset_name": "CRYPTOPUNKS",
    "epochs": 100,
    "lr": 2e-4,  # Lower LR for more stable convergence
    "T": 1000,
    "batch_size": 64,
    "beta_start": 1e-4,
    "beta_end": 0.01,  # Slower noise accumulation
}

# ============================================================
# CryptoPunks with Multi-Attribute Conditioning (Type + Accessories)
# ============================================================
# Conditional generation based on:
# - Type: Male, Female, Zombie, Ape, Alien (5 categories)
# - Accessories: ~87 different accessories (multi-hot encoding)

config_cryptopunks_classes = {
    "dataset_name": "CRYPTOPUNKS_CLASSES",
    "epochs": 200,
    "lr": 3e-4,
    "T": 1000,
    "batch_size": 32,  # Smaller batch for more memory (conditional model is bigger)
    "beta_start": 1e-4,
    "beta_end": 0.02,
    "num_types": 5,  # Male, Female, Zombie, Ape, Alien
    "num_accessories": 87,  # All possible accessories
    "cfg_dropout": 0.1,  # 10% unconditional training for CFG
    "cfg_scale": 3.0,    # Guidance scale at inference
}

config_cryptopunks_classes_fast = {
    "dataset_name": "CRYPTOPUNKS_CLASSES",
    "epochs": 100,
    "lr": 3e-4,
    "T": 500,  # Faster sampling for prototyping
    "batch_size": 32,
    "beta_start": 1e-4,
    "beta_end": 0.02,
    "num_types": 5,
    "num_accessories": 87,
    "cfg_dropout": 0.1,  # 10% unconditional training for CFG
    "cfg_scale": 3.0,    # Guidance scale at inference
}

CONFIGS = {
    "mnist_classes": config1_mnist_classes,
    "mnist": config1_no_classes,
    "mnist2": config2_mnist,
    "mnist3": config3_mnist,
    "cryptopunks1": config1_cryptopunks,
    "cryptopunks2": config2_cryptopunks,
    "cryptopunks3": config3_cryptopunks,
    "cryptopunks_classes": config_cryptopunks_classes,
    "cryptopunks_classes_fast": config_cryptopunks_classes_fast,
}


def get_config_by_name(name):
    if name not in CONFIGS:
        raise ValueError(
            f"Config {name} not found. Available configs: {list(CONFIGS.keys())}"
        )
    return Config(**CONFIGS[name])
