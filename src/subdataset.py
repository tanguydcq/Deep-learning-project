import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class CryptoPunksConceptDataset(Dataset):
    def __init__(
        self,
        image_dir,
        metadata_path,
        concept_name,
        img_size,
    ):
        """
        image_dir: folder with punk images
        metadata_path: json with attributes per image
        concept_name: string, e.g. "Smile"
        """
        self.image_dir = image_dir
        self.concept_name = concept_name

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Filter images containing the concept
        self.items = [
            item for item in metadata
            if concept_name in item["attributes"]
        ]

        if len(self.items) == 0:
            raise ValueError(f"No images found for concept '{concept_name}'")

        self.transform = T.Compose([
            T.Resize(img_size),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])

        print(f"[Dataset] Concept '{concept_name}': {len(self.items)} images")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = os.path.join(self.image_dir, item["image"])
        image = Image.open(img_path).convert("RGB")
        return self.transform(image)
