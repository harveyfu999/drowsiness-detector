import torch
from torch import nn
import os
import numpy as np
from skimage import io, transform
import skimage
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
from torchvision.transforms import v2
from torchvision.io import read_image
import PIL

from model import Net

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EyeDataset(Dataset):
    def __init__(self, open_eye_path, closed_eye_path, transform=None):
        self.transform = transform

        self.open_eye_path = open_eye_path
        self.closed_eye_path = closed_eye_path

        self.open_eye_filenames = os.listdir(self.open_eye_path)
        self.closed_eye_filenames = os.listdir(self.closed_eye_path)

        self.num_open_eye = len(self.open_eye_filenames)
        self.num_closed_eye = len(self.closed_eye_filenames)

    def __len__(self):
        return self.num_open_eye + self.num_closed_eye

    def __getitem__(self, idx):
        img_path = None
        if idx < self.num_open_eye:
            img_path = os.path.join(self.open_eye_path, self.open_eye_filenames[idx])
        else:
            img_path = os.path.join(self.closed_eye_path, self.closed_eye_filenames[idx - self.num_open_eye])
        img = read_image(img_path)

        if self.transform:
            img = self.transform(img)
        return img, torch.tensor([0.80, 0.20], dtype=torch.float32).to(device) if idx < self.num_open_eye else torch.tensor([0, 1], dtype=torch.float32).to(device)


image_size = 128
eye_dataset = EyeDataset(
    "dataset/open_eye", "dataset/closed_eye",
    transform=v2.Compose([
        v2.Resize(image_size),

        v2.RandomHorizontalFlip(),
        v2.RandomRotation(10),

        v2.ToDtype(torch.float32),
        v2.Normalize([89.9], [28.6]) # normalize to -1 to 1
    ])
)
own_dataset = EyeDataset(
    "dataset/test-images-from-myself/open_eye", "dataset/test-images-from-myself/closed_eye",
    transform=v2.Compose([
        v2.Grayscale(),
        v2.Resize(image_size),
        v2.ToDtype(torch.float32),
        v2.Normalize([255/2], [50]) # normalize to -1 to 1
    ])
)

train_size = int(0.95 * len(eye_dataset))
test_size = len(eye_dataset) - train_size
print(f"total: {len(eye_dataset)} train_size: {train_size}, test_size: {test_size}")
eye_dataset_train, eye_dataset_test = torch.utils.data.random_split(eye_dataset, [train_size, test_size])

train_dataloader = DataLoader(eye_dataset_train, batch_size=128, shuffle=True, num_workers=0)
test_dataloader = DataLoader(eye_dataset_test, batch_size=64, shuffle=True, num_workers=0)
own_dataloader = DataLoader(own_dataset, batch_size=1)

model = Net()
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

def test_data(dl):
    model.eval()

    test_len = len(dl) * dl.batch_size
    num_correct = 0
    for i, data in enumerate(dl):
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        outputs = torch.max(outputs, dim=-1).indices
        labels = torch.max(labels, dim=-1).indices

        num_correct += torch.count_nonzero(outputs == labels)
    print("number of correct outputs {}/{} accuracy: {:5.1f}%".format(num_correct, test_len, num_correct/test_len * 100))

def train_one_epoch(epoch_index):
    model.train()
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    train_len = len(train_dataloader)
    for i, data in enumerate(train_dataloader):
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 10 == 0:
            last_loss = running_loss / 10 # loss per batch
            print('epoch {}  batch {}/{} loss: {}'.format(epoch_index, i, train_len, last_loss))
            running_loss = 0.

    return last_loss

print("start")

test_data(test_dataloader)
for i in range(20):
    train_one_epoch(i)
    test_data(test_dataloader)
    test_data(own_dataloader)

torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict()
}, "model.pt")
