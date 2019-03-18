from shufflenet import *
import torchvision as tv
from torchvision.transforms import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_epochs = 50
batch_size = 100
lr = 0.001
n_groups = 2

transform = transforms.Compose([transforms.Pad(96), transforms.ToTensor()])

train_set = tv.datasets.CIFAR10(root='../dataset/', train=True, download=False, transform=transform)
test_set = tv.datasets.CIFAR10(root='../dataset/', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

model = ShuffleNet(n_groups, 10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Testing
def testing():
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# Training
n_train_iters = len(train_loader)
for epoch in range(n_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch_{}, Iteration: [{}/{}], Loss: {:.4f}'.format(epoch+1, i+1, n_train_iters, loss.item()))

    print('-'*50)
    testing()
    print('-'*50)