# %%

import torch
import torchvision
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from image_similarity.data.datasets.triplets_dataset import TripletsDataset

torch.manual_seed(1)

# %%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%

triplet_5033_dataset = TripletsDataset()
dataloader = DataLoader(triplet_5033_dataset, batch_size=16, shuffle=True,
                        num_workers=10)

# %%

epochs = range(6)
learning_rate = 0.0001

# %%

model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2000)
model = model.to(device)
model_path = None

# %%

try:
    model.load_state_dict(torch.load(model_path))
except:
    pass

# %%

optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)


def forward(x):
    x = x.type("torch.FloatTensor").to(device)
    return model(x)


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


loss_tr = []
big_l = []

# %%

total_step = len(dataloader)
current_lr = learning_rate

for epoch in epochs:
    for i, (triplet, l, idx) in enumerate(dataloader):
        print(i, end='\r')
        # forward pass
        positive = forward(triplet[0])
        query = forward(triplet[1])
        negative = forward(triplet[2])
        # compute loss
        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        loss = triplet_loss(positive, query, negative)
        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        loss_tr.append(loss.item())
        optimizer.step()
        if (i + 1) % 100 == 0:
            instant_loss = sum(loss_tr) / len(loss_tr)
            print(
                f"Epoch [{epoch + 1}/{len(epochs)}], Step [{i + 1}/{total_step}] Loss: {instant_loss}")
            big_l += loss_tr
            loss_tr = []

    # Decay learning rate
    if (epoch + 1) % 3 == 0:
        current_lr /= 1.5
        update_lr(optimizer, current_lr)

    torch.save(model.state_dict(), f"MRS{epoch}.ckpt")
    try:
        np.save("loss_file", big_l)
    except:
        pass
# %%
num_features


