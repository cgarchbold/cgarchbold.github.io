---
id: 5
title: "MNIST Classification using Pytorch"
subtitle: "  "
date: "2020.3.20"
tags: "pytorch, mnist, matplotlib"
---

# MNIST Recognition using Pytorch

In this notebook, we will be using pytorch to recognize digits using the MNIST DataSet.


```
# Gather necessary imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
```


```
n_epochs = 30
batch_size = 64
input_size = 784  # this is the length of a 28x28 image flattened
output_size = 10  # there are ten digits to classify to
learning_rate = 0.001


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)
```

    cuda:0
    

# Grabbing the DataLoaders from Pytorch Library


```
train_loader = torch.utils.data.DataLoader( datasets.MNIST('../data', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader( datasets.MNIST('../data', train=False, download=True, transform =transform), batch_size=batch_size, shuffle=True)
```

    0it [00:00, ?it/s]

    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to ../data/MNIST/raw/train-images-idx3-ubyte.gz
    

    9920512it [00:01, 7125387.89it/s]                            
    

    Extracting ../data/MNIST/raw/train-images-idx3-ubyte.gz to ../data/MNIST/raw
    

      0%|          | 0/28881 [00:00<?, ?it/s]

    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to ../data/MNIST/raw/train-labels-idx1-ubyte.gz
    

    32768it [00:00, 125831.96it/s]           
      0%|          | 0/1648877 [00:00<?, ?it/s]

    Extracting ../data/MNIST/raw/train-labels-idx1-ubyte.gz to ../data/MNIST/raw
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to ../data/MNIST/raw/t10k-images-idx3-ubyte.gz
    

    1654784it [00:00, 2006293.24it/s]                            
    0it [00:00, ?it/s]

    Extracting ../data/MNIST/raw/t10k-images-idx3-ubyte.gz to ../data/MNIST/raw
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to ../data/MNIST/raw/t10k-labels-idx1-ubyte.gz
    

    8192it [00:00, 47651.31it/s]            

    Extracting ../data/MNIST/raw/t10k-labels-idx1-ubyte.gz to ../data/MNIST/raw
    Processing...
    Done!
    

    
    


```
dataiter = iter(train_loader)
images, labels = dataiter.next()

print(images.shape)
print(labels.shape)

```

    torch.Size([64, 1, 28, 28])
    torch.Size([64])
    


```
fig, axs = plt.subplots(7, 7, figsize=(25,25))
for i, ax in enumerate(axs.flatten()):
    ax.text(0.5,0.5,labels[i].item())
    ax.imshow(images[i].numpy().squeeze(), cmap='gray_r')
    ax.axis('off')
```


    
![png](/images/MNIST_Recognition_6_0.png)
    


# Defining the Network


```
hidden_sizes = [128,64]

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))
print(model)
```

    Sequential(
      (0): Linear(in_features=784, out_features=128, bias=True)
      (1): ReLU()
      (2): Linear(in_features=128, out_features=64, bias=True)
      (3): ReLU()
      (4): Linear(in_features=64, out_features=10, bias=True)
      (5): LogSoftmax()
    )
    

##Defining the Loss Function and Optimizer


```
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate , momentum=0.9)
```

# Training


```
for epoch in range(n_epochs): 

    running_loss = 0.0
    for images, labels in train_loader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(images)
        loss = criterion(output, labels)
        
        #backpropigate and optimize weights
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    else:
       print("Epoch {} - Training loss: {}".format(epoch, running_loss/len(train_loader)))

print('Finished Training')
```

    Epoch 0 - Training loss: 1.0705403806939562
    Epoch 1 - Training loss: 0.3867814064915501
    Epoch 2 - Training loss: 0.3237278477183537
    Epoch 3 - Training loss: 0.29278225556158943
    Epoch 4 - Training loss: 0.2684654144288253
    Epoch 5 - Training loss: 0.24709439691481813
    Epoch 6 - Training loss: 0.22732517401626243
    Epoch 7 - Training loss: 0.20831199590442404
    Epoch 8 - Training loss: 0.19244041254144234
    Epoch 9 - Training loss: 0.17651523826028237
    Epoch 10 - Training loss: 0.16485402044425132
    Epoch 11 - Training loss: 0.15292192004057073
    Epoch 12 - Training loss: 0.14248154261338114
    Epoch 13 - Training loss: 0.13378774341958355
    Epoch 14 - Training loss: 0.12629079531981494
    Epoch 15 - Training loss: 0.11931361968734308
    Epoch 16 - Training loss: 0.11213078654841828
    Epoch 17 - Training loss: 0.106711111781098
    Epoch 18 - Training loss: 0.10172186182267758
    Epoch 19 - Training loss: 0.09737626568618804
    Epoch 20 - Training loss: 0.09174584891639158
    Epoch 21 - Training loss: 0.08835734114515534
    Epoch 22 - Training loss: 0.0851000149978567
    Epoch 23 - Training loss: 0.08132447421884359
    Epoch 24 - Training loss: 0.07750558923445404
    Epoch 25 - Training loss: 0.07428115591017613
    Epoch 26 - Training loss: 0.07198296943957459
    Epoch 27 - Training loss: 0.06840373125862577
    Epoch 28 - Training loss: 0.06641779836567083
    Epoch 29 - Training loss: 0.06408126084312701
    Finished Training
    

# Testing


```
correct_count, all_count = 0, 0
for images,labels in test_loader:
  for i in range(len(labels)):
    img = images[i].view(1, 784)
    with torch.no_grad():
        logps = model(img)

    
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))
```

    Number Of Images Tested = 10000
    
    Model Accuracy = 0.9732
    
