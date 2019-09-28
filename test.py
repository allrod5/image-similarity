import logging

from torch.utils.data import DataLoader

from image_similarity.data.datasets.triplets_dataset import TripletsDataset

logging.basicConfig(level=logging.ERROR)

dataset = TripletsDataset()

dataloader = DataLoader(dataset, batch_size=50, shuffle=True, num_workers=4)
for i, batch in enumerate(dataloader):
    print(i, batch)
