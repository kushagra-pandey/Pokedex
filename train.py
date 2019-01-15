import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms

transform = transforms.Compose(
                   [transforms.Scale((32,32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset  = dset.ImageFolder(root="dataset",transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True,num_workers=2)
testset = dset.ImageFolder(root='tests',transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=True,num_workers=2)
