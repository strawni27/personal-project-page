import torch
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import sys
import matplotlib.pyplot as plt

#Want to save the output to a .txt file
log_file = open('./frogs.txt', 'w')
sys.stdout = log_file

#Starting by providing the logic to transform images to be output as tensors with normalized
# ranges from -1 to 1
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

#Now, we subset the data and provide the data loaders
batch_size = 32

trainset = torchvision.datasets.CIFAR10(root='./data', train = True,
                                        download = True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
                                          shuffle = True, num_workers = 2)

testset = torchvision.datasets.CIFAR10(root='./data', train = False,
                                       download = True, transform = transform)

testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size,
                                         shuffle = False, num_workers = 2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

#Now we write a function to show some images like in the 60 minute blitz
def imshow(img):
    img = img/2 + 0.5 #unnormalizing them
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

#Get random training images
#dataiter = iter(trainloader)
#images, labels = next(dataiter)

#Displaying the images
#imshow(torchvision.utils.make_grid(images))
#Printing labels
#print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Giving this a similar ResNet structure as I gave my font classifier
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #gonna have a similar structure as the original net did, but with a resnet flavor instead of pooling layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x

#Defining the neural net
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #This layer takes the 3 channels, flattens them into 6 maps, and uses a 5x5 filter to slide over the images
        self.block1 = ResBlock(3, 25)
        #Pooling layer with 2x2 window and stride of 2
        self.pool = nn.MaxPool2d(2,2)
        #Convolution layer that takes the 6 maps, turns them into 16 maps, and uses a 5x5 filter again. Gets more fine details
        self.block2 = ResBlock(25, 60)
        self.block3 = ResBlock(60, 100) 
        #Creating 3 linear layers that take our maps and flattens them, then compresses them
        self.fc1 = nn.Linear(100*8*8, 4000)
        self.fc2 = nn.Linear(4000, 1000)
        self.fc3 = nn.Linear(1000, 10)
        
    def forward(self,x):
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        x = self.block3(x)
        x = torch.flatten(x, 1) #Flatten everything except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return(x)
    
net = Net()

#Defining loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

#Training loop
epochs = 12
total_batches = len(trainloader) * epochs  # total iterations across all epochs
batches_done = 0
batch_print = 500
for epoch in range(epochs): #This tells it to loop over the dataset multiple times
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        #Pulling the inputs, with the data being a list of [inputs, labels]
        inputs, labels = data
        
        #zero the parameter gradients
        optimizer.zero_grad()
        
        #forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        #Print statistics
        running_loss += loss.item()
        batches_done += 1
        if i % batch_print == (batch_print-1): 
            pct = 100 * batches_done / total_batches
            msg = f'[Epoch {epoch+1}, Batch {i+1}] loss: {running_loss/batch_print:.3f} | Progress: {pct:.1f}%\n'
            print(msg)
            running_loss = 0.0
            
print('Finished Training')

#Saving the model that we just made
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

correct = 0
total = 0 
#Without calculating gradients, we're checking the correctness of each of the test images
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    
#Checking how well it did category by category
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

#Still not using gradients
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # Collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

#Printing the accuracy
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    
log_file.close()