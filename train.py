import argparse
import json
import torch
import numpy as np
from torchvision import datasets, transforms, models
from torch import nn, optim
from collections import OrderedDict

parser = argparse.ArgumentParser(description = 'Train a neural network on a dataset')

parser.add_argument('data_directory', help = 'Path to the dataset')

parser.add_argument('--arch', default = 'densenet121', choices=['vgg13', 'densenet121'], help = 'vgg13 or densenet121')

parser.add_argument('--save_dir', default = './',
                    help = 'Path to the directory where checkpoint will be saved')

parser.add_argument('--learning_rate', default = 0.001, help = 'Learning rate')

parser.add_argument('--hidden_units', default = 500, help = 'Hidden units')

parser.add_argument('--epochs', default = 3, help = 'Epochs')

parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

args = parser.parse_args();

if args.gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('***********************Arguments**************************')
    
data_dir = args.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30), transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
valid_datasets = datasets.ImageFolder(valid_dir, transform=test_transforms)

trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
testloaders = torch.utils.data.DataLoader(test_datasets, batch_size = 64, shuffle=True)
validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size = 64, shuffle=True)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

print('***********************Data Loading Done**************************')
    
hidden_units = int(args.hidden_units)
learning_rate = float(args.learning_rate)

if args.arch == 'densenet121':
    model = models.densenet121(pretrained = True)
    
    input_size = model.classifier.in_features
    
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    model.classifier = classifier

elif args.arch == 'vgg13':
    model = models.vgg13(pretrained = True)
    
    input_size = model.classifier[0].in_features
    
    for param in model.parameters():
        param.requires_grad = False
        
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    model.classifier = classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

model.to(device)

print('***********************Model Training Starting**************************')

epochs = int(args.epochs)
print_every = 10
steps = 0
running_loss = 0

for epoch in range(epochs):
     for inputs, labels in trainloaders:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            
            model.eval()
            
            with torch.no_grad():
                for inputs, labels in validloaders:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Train loss: {running_loss/print_every:.3f}, "
                  f"Valid loss: {valid_loss/len(validloaders):.3f}, "
                  f"Valid accuracy: {accuracy/len(validloaders):.3f}") 
            running_loss = 0
            model.train()
            
print('***********************Test Accuracy**************************')
            
test_loss = 0
test_accuracy = 0
model.eval()
with torch.no_grad():
    for inputs, labels in testloaders:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)
        
        test_loss += batch_loss.item()

        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

print(f"Test loss: {test_loss/len(testloaders):.3f}, "
      f"Test accuracy: {test_accuracy/len(testloaders):.3f}")

checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'criterion_state_dict': criterion.state_dict(),
    'class_to_idx': train_datasets.class_to_idx,
    'classifier': model.classifier,
    'arch': args.arch
}

torch.save(checkpoint, 'checkpoint.pth')