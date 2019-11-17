from torch import cat
from torch.nn import Module, Embedding, Sequential, Conv2d, MaxPool2d, Flatten, \
    TripletMarginLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.models import alexnet

from image_similarity.data.datasets.triplets_dataset import TripletsDataset


class DeepRanking(Module):
    def __init__(self, triplets_dataset: TripletsDataset, dimension: int):
        super().__init__()
        self.high_invariance_net = alexnet(pretrained=True)
        self.low_invariance_net_1 = Sequential(
            Conv2d(3, 3, 1, stride=4, padding=2),
            Conv2d(3, 96, 8, stride=4, padding=4),
            MaxPool2d(3, stride=4, padding=0),
            Flatten(),
        )
        self.low_invariance_net_2 = Sequential(
            Conv2d(3, 3, 1, stride=8, padding=2),
            Conv2d(3, 96, 8, stride=4, padding=4),
            MaxPool2d(7, stride=2, padding=3),
            Flatten(),
        )

    def forward(self, triplet):
        triplet_embeddings = []
        for sample in triplet:
            out_1 = self.high_invariance_net(sample)
            out_2 = self.low_invariance_net_1(sample)
            out_3 = self.low_invariance_net_2(sample)
            c = cat((out_2, out_3), dim=1)
            combined = cat((c, out_1), dim=1)
            triplet_embeddings.append(combined)
        return triplet_embeddings


if __name__ == '__main__':
    triplet_5033_dataset = TripletsDataset()
    dataloader = DataLoader(triplet_5033_dataset, batch_size=32, shuffle=True,
                            num_workers=10)
    model = DeepRanking(triplet_5033_dataset, 4096)
    model.float()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_tr = []
    big_l = []
    total_step = len(dataloader)
    epochs = range(6)
    for epoch in epochs:
        for i, batch in enumerate(dataloader):
            print(i, end='\r')
            for triplet in zip(*batch):
                optimizer.zero_grad()
                output = model(triplet)
                triplet_loss = TripletMarginLoss(margin=1.0, p=2)
                loss = triplet_loss(*output)
                loss.backward()
                loss_tr.append(loss.item())
                optimizer.step()
                print(
                    f"Epoch [{epoch + 1}/{len(epochs)}], Step [{i + 1}/{total_step}]"
                    f" Loss: {loss.item()}"
                )

