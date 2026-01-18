import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='../data/mnist', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = datasets.MNIST(root='../data/mnist', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        # Flatten data from (n, 1, 28, 28) to (n, 784)
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)  # flatten
        x = self.fc(x)
        return x
# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.model = torch.nn.Sequential(
#             # The size of the picture is 28x28
#             torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2),
#
#             # The size of the picture is 14x14
#             torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2),
#
#             # The size of the picture is 7x7
#             torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#
#             torch.nn.Flatten(),
#             torch.nn.Linear(in_features=7 * 7 * 64, out_features=128),
#             torch.nn.ReLU(),
#             torch.nn.Linear(in_features=128, out_features=10),
#             torch.nn.Softmax(dim=1)
#         )
#
#     def forward(self, input):
#         output = self.model(input)
#         return output

model = Net()

device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
model.to(device)


criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)  # GPU
        optimizer.zero_grad()
        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %3d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 2000))
            running_loss = 0.0


accuracy = []


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)  # GPU
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print('Accuracy on test set: %d %% [%d/%d]' % (100 * correct / total, correct, total))
    accuracy.append(100 * correct / total)


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
    print(accuracy)

    plt.plot(range(10), accuracy)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), "mnist_cnn.pth")
    print("模型已保存为 mnist_cnn.pth")