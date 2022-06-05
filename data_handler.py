import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
torch.manual_seed(0)


train_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(size=124),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])


test_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.CenterCrop(124),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

])


root_dir = 'C:/Users/andre/Documents/dataset/Intel Image Classification'
# Load the training data
trainset = datasets.ImageFolder(
    root_dir+'/seg_train/seg_train', transform=train_transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)


# Load the test data
testset = datasets.ImageFolder(
    root_dir + '/seg_test/seg_test', transform=test_transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

n_classes = len(trainset.classes)

# print(trainloader.dataset, '\n')
# print(testloader.dataset)
