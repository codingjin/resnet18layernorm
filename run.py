from resnet18_layernorm import ResNet18

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import sys

if len(sys.argv)==1:
    BATCH_SIZE =1
else:
    BATCH_SIZE = int(sys.argv[1])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#set hyperparameter
EPOCH = 1000
#EPOCH = 20
#BATCH_SIZE = 128
LR = 0.001
#WEIGHT_DECAY=1e-4

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

#labels in CIFAR10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#define ResNet18
net = ResNet18(input_shape=[32, 32]).to(device)

#define loss funtion & optimizer
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
#optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
optimizer = optim.Adam(net.parameters(), lr=LR)

outputfilename = "_".join([str(BATCH_SIZE), "output.csv"])
fw = open(outputfilename, "w")
fw.write('Epoch\tLoss\tTrain_accuracy\tTest_accuracy\n')

#train
for epoch in range(0, EPOCH):
    print('\nEpoch: %d' % (epoch + 1))
    net.train()
    sum_loss = 0.0
    #correct = 0.0
    #total = 0.0
    train_correct = 0.0
    train_total = 0.0
    for i, data in enumerate(trainloader, 0):
        #prepare dataset
        length = len(trainloader)
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        #forward & backward
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        #print ac & loss in each batch
        sum_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels.data).cpu().sum()
#        print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% ' 
#              % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * train_correct / train_total))
    print('Epoch: %d | Loss: %.03f | Acc: %.3f%% ' % (epoch+1, sum_loss / (i + 1), 100. * train_correct / train_total))
        
    #get the ac with testdataset in each epoch
    print('Waiting Test...')
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        #correct = 0
        #total = 0
        for data in testloader:
            net.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum()
        print('Test\'s ac is: %.3f%%' % (100 * test_correct / test_total))

    fw.write('%d\t%.03f\t%.3f%%\t%.3f%%\n' % (epoch+1, sum_loss / (i + 1), 100. * train_correct / train_total, 100 * test_correct / test_total))

fw.close()

print('Train has finished, total epoch is %d' % EPOCH)

