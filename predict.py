import argparse
import json
import torch
import numpy as np
from torchvision import datasets, transforms, models
from torch import nn, optim
from collections import OrderedDict
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument('input', default='flowers/test/15/image_06369.jpg', help = 'Path to the input image')

parser.add_argument('checkpoint', default='checkpoint.pth', help = 'Path to model checkpoint')

parser.add_argument('--category_names', default='cat_to_name.json', help = 'Use a mapping of categories to real names')

parser.add_argument('--top_k', default = 5, help = 'Return top K most likely classes')

parser.add_argument('--gpu', action='store_true', help = 'Use GPU for inference')

args = parser.parse_args()

if args.gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    if checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained = True)
    
    model.classifier = checkpoint['classifier']    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

checkpoint = args.checkpoint
model = load_checkpoint(checkpoint)
model.to(device)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return image_transform(Image.open(image))

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    image = process_image(image_path)
    image = image.float().unsqueeze(0).to(device)
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        output = model.forward(image)
        
    output_probabilities = torch.exp(output)
    probabilities, indices = output_probabilities.topk(topk)

    probabilities = probabilities.cpu().numpy().tolist()[0]
    indices = indices.cpu().numpy().tolist()[0]
    mapping = {val: key for key, val in model.class_to_idx.items()}
    classes = [mapping[item] for item in indices]
    
    return probabilities, classes

image_path = args.input
top_k = int(args.top_k)
probabilities, classes = predict(image_path, model, top_k)

category_names = args.category_names
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

classNames = [cat_to_name[c] for c in classes]

max_class_name_length = max(len(name) for name in classNames)

print(f"{'Class Name':<{max_class_name_length + 2}}Probability")
print("-" * (max_class_name_length + 14)) 

for name, prob in zip(classNames, probabilities):
    print(f"{name:<{max_class_name_length + 2}}{prob:.4f}")