import argparse
import json
from PIL import Image
import torch
import numpy as np

from math import ceil
from train import check_gpu
from torchvision import models, transforms

def arg_parser():
    
    parser = argparse.ArgumentParser(description="Prediction settings")
    
    parser.add_argument('-i', '--image', type=str, help = 'path of the image to predict', required=True)
    parser.add_argument('-c', '--checkpoint', type=str, help = 'checkpoint path', required=True )
    parser.add_argument('-k', '--topk', type=int, help = 'choose number of matches')
    parser.add_argument('-cn', '--category_names', type=str, help = 'file pointing to category names', required=True)
    parser.add_argument('-g', '--gpu',  action='store_true',help = 'use gpu for prediction')
    
    return parser.parse_args()

def load_checkpoint(checkpoint_path):
    # load checlpoint for the path
    checkpoint = torch.load(checkpoint_path)
    
    model = models.vgg16(pretrained=True)
    model.name = "vgg16"
    ldic=locals()
    if checkpoint['architecture'] != "vgg16":
        exec("model = models.{}(pretrained=True)".format(checkpoint['architecture']), globals(), ldic)
        model.name = checkpoint['architecture']
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])
    
    img = transform(pil_image)
    return img

def predict(img_tensor, model, device, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    if topk is None:
        topk = 5
    model.to(device)
    model.eval()
    img_tensor.unsqueeze_(0)
    img_tensor = img_tensor.float()
    with torch.no_grad():
        if device != "cpu":
            logps = model.forward(img_tensor.cuda())
        else:
            logps = model.forward(img_tensor)
    ps = torch.exp(logps)
    return ps.topk(topk, dim = 1)


def print_prediction(probabilities, json_path):
    with open(json_path, 'r') as json_file:
        cat_to_name = json.load(json_file)



    labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
    probability = np.array(probabilities[0][0])

    final_prediction = ""
    max = 0
    i=0
    while i < len(probability):
        if probability[i] > max:
            max = probability[i]
            final_prediction = labels[i]
        print("{} with a probability of {}".format(labels[i], probability[i]))
        i += 1
    
    print("\n\nFinal prediction for the given image is {}".format(final_prediction))
    

def main():
    args = arg_parser()
    
    # load the model from the checkpoint
    model = load_checkpoint(args.checkpoint)
    
    # process the image used for prediction
    img_tensor = process_image(args.image)
    
    # check if the gpu is available
    device = check_gpu(use_gpu=args.gpu)
    
    # Get predicted probabilities for the input image
    probabilities = predict(img_tensor, model, device ,args.topk)
    
    # print the probabilities and predicted category.
    print_prediction(probabilities, args.category_names)
    
    
    
if __name__ == '__main__': main()