import argparse
import torch
import json
import numpy as np
from PIL import Image
from torchvision import datasets, transforms, models

def process_image(image):
    
    new_image = Image.open(image)
    new_image.thumbnail((256,256))
    width, height = new_image.size
    #print(width, height)
    
    # crop out center 224*224 portion of image
    
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    
    #Crop image
    new_image = new_image.crop((left, top, right, bottom))
    
    # convert the cropped image into numpy array
    new_image = np.array(new_image)
    
    # Since encoded values are within 0-255, but the model expected floats 0-1 we will devide the numpy array of image by 255
    new_image = new_image/255
    
    # normalize with given mean and standard deviation
    mean = np.array([0.485,0.456,0.406])
    sd = np.array([0.229, 0.224, 0.225])
    new_image = (new_image - mean) / sd
    #print(new_image)
    
    new_image = new_image.transpose((2,0,1))
    
    return torch.from_numpy(new_image)

import sys
def load_checkpoint(filepath, gpu=True):
    checkpoint = torch.load(filepath)
    arch = checkpoint['architecture']
    class_idx = checkpoint['class_to_idx']
    
    if arch == 'vgg16':
        load_model = models.vgg16(pretrained=True)
    elif arch == 'densenet':
        load_model = models.densenet121(pretrained=True)
    else:
        print('no supported architecture')
        sys.exit()
        
    for param in load_model.parameters():
        param.requires_grad = False
    
    load_model.classifier = checkpoint['classifier']
    load_model.load_state_dict(checkpoint['state_dict'])
    load_model.class_to_idx = checkpoint['class_to_idx']
    
    if not gpu:
        return load_model.to('cpu')
    return load_model.cuda()

def predict(input_image, checkpoint, category_names, top_k, gpu=True):
    load_model = load_checkpoint(checkpoint, gpu)
    image = process_image(input_image)
    
    image = image.unsqueeze(0).float()
    if gpu:
        image = image.to('cuda')
    else:
        image = image.to('cpu')
                    
    with torch.no_grad():
        out = load_model.forward(image)
        results = torch.exp(out).data.topk(int(top_k))
    
    classes = np.array(results[1][0])
    prob = np.array(results[0][0])
    

    class_idx = {value:key for key,value in load_model.class_to_idx.items()}
    classes = [class_idx[x] for x in classes]
    print(classes)
    print(prob)

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    flowers_names = [cat_to_name[x] for x in classes]
    print(flowers_names)
    return prob, classes


def main():
    
    # Common Use case: 
    # python predict.py flowers/test/1/image_06743.jpg checkpoint.pth --top_k 3
    
    parser = argparse.ArgumentParser(description='Get NN arguments')
    #Define arguments
    parser.add_argument('input', type=str)
    parser.add_argument('checkpoint', default='')
    parser.add_argument('--category_names', default="cat_to_name.json" )
    parser.add_argument('--top_k', default=5)
    parser.add_argument('--gpu', default=True )
    
    args = parser.parse_args() 
    
    predict(args.input, args.checkpoint, args.category_names, args.top_k, args.gpu)
        
if __name__ == '__main__':
    main()