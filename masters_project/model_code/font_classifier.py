import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

#This loads in the data 
folder = './data/character_font_images/'
#This next line takes all of the CSV's and joins them together
all_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]
#Reads all of the CSV's, concatonates them all into their own data files
df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)

#Poking around with the data momentarily to get normalization statistics
pixel_cols = [c for c in df.columns if c.startswith('r')]
pixels = df[pixel_cols].values / 255.0

#Getting the mean and standard deviation
mean = pixels.mean()
std = pixels.std()

print(f"Mean: {mean:.4f}, Std: {std:.4f}")    

#To begin, I'm focusing on the FONT labels, not necessarily the characters. In order to check how those go, 
# we need to encode the font labels numerically rather than with the character labels

#Start by getting the unique fonts
unique_fonts = sorted(df['font'].unique())

font_to_idx = {font: idx for idx, font in enumerate(unique_fonts)}
df['font_label'] = df['font'].map(font_to_idx)

#Defining the data class that we'll be using
from torch.utils.data import Dataset, random_split
class FontDataset(Dataset):
    #Setting up the data set (different features that might be present here)
    def __init__(self, X, y, mean, std, transform=None):
        self.images = X[pixel_cols].values.astype(np.float32)
        self.labels = y['font_label'].values
        self.mean = np.float32(mean)
        self.std = np.float32(std)
        self.transform = transform
        
    #How long is the data set?
    def __len__(self):
        return len(self.images)
    
    #How do we pull an item from this dataset?
    def __getitem__(self, idx):
        #Making the rows look like 20x20 images
        image = self.images[idx].reshape(1, 20, 20).astype(np.float32) / 255.0
        #Standardizing
        image = (image - self.mean) / self.std
        #Keeping track of labels
        label = self.labels[idx]
        #Give output pls
        return torch.tensor(image), torch.tensor(label, dtype = torch.long)

pixel_cols = [c for c in df.columns if c.startswith('r')]
X = df[pixel_cols]
y = df[['font_label']]

#Getting the data loaders
#Data set first
dataset = FontDataset(X, y, mean, std)

#Getting batch sizes
batch_size = 32

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

trainset, valset, testset = random_split(dataset, [train_size, val_size, test_size])

#Defining the data loaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
                                          shuffle = True, num_workers = 2)

valloader = torch.utils.data.DataLoader(valset, batch_size = batch_size,
                                         shuffle = False, num_workers = 2) #Doesn't need to be shuffled cuz we're just validating on it

testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size,
                                         shuffle = False, num_workers = 2) #Doesn't need to be shuffled

import matplotlib.pyplot as plt

#Now we write a function to show some images like in the 60 minute blitz
def imshow(img):
    img = img*std + mean #unnormalizing them
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), cmap = 'gray')
    plt.show()

#Get random training images
dataiter = iter(trainloader)

images, labels = next(dataiter)

#Displaying the images
#n = 4
#imshow(torchvision.utils.make_grid(images[:n]))
#Printing font names
#idx_to_font = {v: k for k, v in font_to_idx.items()}
#print(' '.join(f'{idx_to_font[labels[j].item()]:5s}' for j in range(n)))

import torch.nn.functional as F
import torch.optim as optim

#Trying a new structure. I'm going to create a ResNet block so that the dimensions of my output of each of the block match the dimensions of the input; this is important
# because in the skip connection, we need the dimensions of the residuals and the input to match so that we can add them together in the relu step.
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

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #Going to have 2 repititions of the resnet blocks, a couple pooling layers, and then 3 linear layers.
        self.block1 = ResBlock(1, 30)
        self.pool = nn.MaxPool2d(2,2)
        self.block2 = ResBlock(30, 60)
        self.block3 = ResBlock(60, 100)
        self.fc1 = nn.Linear(100*5*5, 250)
        self.fc2 = nn.Linear(250, 200)
        self.fc3 = nn.Linear(200,153)
    
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

#Loss function, optimizer, and training loop
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum = 0.9)

#Attempting to shove my stuff into the GPU manually, just to be safe
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device)

#Training Loop
epochs = 12
total_batches = len(trainloader) * epochs
batches_done = 0
log_file = open('font_classifier.txt', 'w')

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        #Getting the inputs, assuming data is of structure [inputs, labels], and sending them to the GPU
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        #Zero parameter gradients
        optimizer.zero_grad()
        
        #Forward, backward, and optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        #Tell me the statistics
        running_loss += loss.item()
        batches_done += 1
        #Print the loss statistics every 3000 mini batches
        if i % 3000 == 2999: 
            pct = 100 * batches_done / total_batches
            msg = f'[Epoch {epoch+1}, Batch {i+1}] loss: {running_loss/3000:.3f} | Progress: {pct:.1f}%\n'
            print(msg)
            log_file.write(msg)
            log_file.flush()
    
    #Adding a validation step
    net.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in valloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted ==labels).sum().item()
    msg = f'Val Loss: {val_loss/len(valloader):.3f} | Accuracy: {100*correct/total:.1f}%\n'
    print(msg)
    log_file.write(msg)
    log_file.flush()
    
    #Saving a checkpoint at the end of each epoch in case bad shit happens
    torch.save(net.state_dict(), f'checkpoint_epoch_{epoch+1}.pth')
    net.train()

print('Training Complete')
    
#Saving the model as a path
PATH = './font_net.pth'
torch.save(net.state_dict(), PATH)

#Loading back in our model
net = Net()
net.load_state_dict(torch.load(PATH, weights_only=True))

#Checking the accuracy of the model
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

log_file.close()