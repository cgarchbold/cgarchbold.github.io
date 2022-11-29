---
id: 1
title: "Corgi Classification using CNNs"
subtitle: " UKY CS 460 Machine Learning "
date: "2020.12.04"
tags: "cnn, jupyter"
---

# Corgi Classification Using CNNs

##Bringing in the data


```
!git clone https://github.com/cgarchbold/460Assignment5.git
```

    Cloning into '460Assignment5'...
    remote: Enumerating objects: 4, done.[K
    remote: Counting objects: 100% (4/4), done.[K
    remote: Compressing objects: 100% (4/4), done.[K
    remote: Total 4 (delta 0), reused 0 (delta 0), pack-reused 0[K
    Unpacking objects: 100% (4/4), done.
    


```
!unzip -q ./460Assignment5/data
```


```
!mv ./460Assignment5/dataset.py ./
```

##Imports



```
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
```

## Dataset


```
import dataset as corgi_Data

# Use Dr.Harrisons code to import the data
images,labels,img_names,cls = corgi_Data.load_train('./data/training_data', 224, ['cardigan','pembroke'])
```

    Going to read training images
    Now going to read cardigan files (Index: 0)
    Now going to read pembroke files (Index: 1)
    


```
# Lets take a look at the first image
plt.imshow(images[0])
print(images[0].shape)
```

    (224, 224, 3)
    


    
![png](/images/Corgi_Classification_9_1.png)
    



```
# move the channel to the front
transformed_images = []
for image in images:
    x = np.moveaxis(image, -1, 0)
    transformed_images.append(x)
```


```
# converting to tensor shapes
images_tensor = torch.Tensor(transformed_images)
labels_tensor = torch.Tensor(labels)

# Make Pytorch dataset and dataloader
dataset = TensorDataset(images_tensor,labels_tensor)
print(dataset[1][0].shape)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2) # shuffle is important here
```

    torch.Size([3, 224, 224])
    

## CNN


```
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = ConvNet()
```


```
# Training Parameters
epochs = 10

# setting loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```


```
for epoch in range(epochs):

    running_loss = 0.0

    for i,data in enumerate(dataloader,0):

        inputs,labels = data

        #convert labels to single 0s or 1s
        labels = np.argmax(labels, axis=1)
        
        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs,labels)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 25 == 24:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 25))
            running_loss = 0.0

       
```

    [1,    25] loss: 0.739
    [1,    50] loss: 0.693
    [2,    25] loss: 0.680
    [2,    50] loss: 0.683
    [3,    25] loss: 0.657
    [3,    50] loss: 0.630
    [4,    25] loss: 0.773
    [4,    50] loss: 0.671
    [5,    25] loss: 0.595
    [5,    50] loss: 0.623
    [6,    25] loss: 0.386
    [6,    50] loss: 0.426
    [7,    25] loss: 0.194
    [7,    50] loss: 0.244
    [8,    25] loss: 0.104
    [8,    50] loss: 0.053
    [9,    25] loss: 0.028
    [9,    50] loss: 0.047
    [10,    25] loss: 0.147
    [10,    50] loss: 0.103
    


```
PATH = './corgi_net.pth'
torch.save(model.state_dict(), PATH)
```

## Testing Accuracy

Lets load in the test data the same way we did the training data


```
test_images,test_labels,test_img_names,test_cls = corgi_Data.load_train('./data/testing_data', 224, ['cardigan','pembroke'])

test_transformed_images = []
for image in test_images:
    x = np.moveaxis(image, -1, 0)
    test_transformed_images.append(x)

test_images_tensor = torch.Tensor(test_transformed_images)
test_labels_tensor = torch.Tensor(test_labels)

test_dataset = TensorDataset(test_images_tensor,test_labels_tensor)
print(dataset[1][0].shape)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)
```

    Going to read training images
    Now going to read cardigan files (Index: 0)
    Now going to read pembroke files (Index: 1)
    torch.Size([3, 224, 224])
    


```
correct = 0
total = 0
with torch.no_grad():
    for data in test_dataloader:
        
        images, labels = data

        #convert label to single classes
        labels = np.argmax(labels,axis=1)
        
        outputs = model(images)
        
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
```

    Accuracy of the network on the test images: 75 %
    
