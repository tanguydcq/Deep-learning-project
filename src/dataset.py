import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from utils import plot_images
import json
import os
from PIL import Image


class CryptoPunksWithAttributes(Dataset):
    """
    CryptoPunks dataset with multi-attribute labels (type + accessories).
    """
    def __init__(self, root_path, transform=None):
        self.root_path = root_path
        self.transform = transform
        
        # Load metadata
        metadata_path = os.path.join(root_path, "metadata.json")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.type_to_idx = self.metadata['type_to_idx']
        self.accessory_to_idx = self.metadata['accessory_to_idx']
        self.num_types = self.metadata['num_types']
        self.num_accessories = self.metadata['num_accessories']
        self.labels = self.metadata['labels']
        
        # Get list of images
        self.images_dir = os.path.join(root_path, "images")
        self.image_files = sorted([
            f for f in os.listdir(self.images_dir) 
            if f.endswith('.png')
        ], key=lambda x: int(x.split('.')[0]))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get label
        punk_id = str(int(img_name.split('.')[0]))
        if punk_id in self.labels:
            label_data = self.labels[punk_id]
            type_idx = label_data['type_idx']
            accessory_vector = torch.tensor(label_data['accessory_vector'], dtype=torch.float32)
        else:
            # Default values if label not found
            type_idx = 0
            accessory_vector = torch.zeros(self.num_accessories, dtype=torch.float32)
        
        return image, (type_idx, accessory_vector)


def get_mnist_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.img_size, args.img_size)), 
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)) 
    ])
    dataset = torchvision.datasets.MNIST(
        root=args.dataset_path,
        train=True,
        download=True,
        transform=transforms
    )
    return dataset


def get_custom_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.img_size, args.img_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = torchvision.datasets.ImageFolder(root=args.dataset_path, transform=transforms)
    return dataset


def get_cryptopunks_with_attributes(args):
    """Get CryptoPunks dataset with multi-attribute labels."""
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.img_size, args.img_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = CryptoPunksWithAttributes(
        root_path=args.dataset_path,
        transform=transforms
    )
    return dataset


def get_data(args):
    if args.dataset_name == "MNIST":
        dataset = get_mnist_data(args)
    elif args.dataset_name == "CRYPTOPUNKS":        
        dataset = get_custom_data(args)
    elif args.dataset_name == "CRYPTOPUNKS_CLASSES":
        dataset = get_cryptopunks_with_attributes(args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")
    
    # Adjust num_workers based on dataset and OS
    # Windows has issues with multiprocessing and custom datasets
    if args.dataset_name == "CRYPTOPUNKS_CLASSES":
        num_workers = 0  # Custom dataset needs 0 workers on Windows
        persistent_workers = False
    else:
        num_workers = 8
        persistent_workers = True
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers
    )
    return dataloader
