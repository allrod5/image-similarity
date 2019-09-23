from typing import Optional

import torchvision
from torch.utils.data import DataLoader


class TripletsDataLoader(DataLoader):
    DEFAULT_TRANSFORM = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomVerticalFlip(p=0.5),
            torchvision.transforms.ToTensor(),
        ]
    )

    def __init__(self, triplets_file: str, transform: Optional[object] = None):
        super().__init__()
        self.transform = transform or self.DEFAULT_TRANSFORM
