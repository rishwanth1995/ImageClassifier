import argparse
import torch
from collections import OrderedDict
import os
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import time

def arg_parser():
    # Define arguments for the application 
    argument_parser = argparse.ArgumentParser(description="Classifier settings")
    argument_parser.add_argument('-a', '--architecture', type=str,
                                 help = 'Choose pytorch architecture, please use string')
    argument_parser.add_argument('-s', '--save_dir', type=str, help='Directory to save the checkpoints for the model')
    argument_parser.add_argument('-l','--learning_rate', type=float, help='Learning rate for the model')
    argument_parser.add_argument('-hl','--hidden_layer_units', type=int, help = 'Please enter number of hidden units for the hidden layer')
    argument_parser.add_argument('-e','--epoch', type=int, help='Number of epochs for training use integer')
    argument_parser.add_argument('-g','--gpu', action='store_true', help='Is GPU enabled when training')
    
    args = argument_parser.parse_args()
    return args

def get_pretrained_model(architecture = "vgg16"):
    
    model = models.vgg16(pretrained=True)
    model.name = "vgg16"
    ldic=locals()
    if architecture is not None:
        exec("model = models.{}(pretrained=True)".format(architecture), globals(), ldic)
        model.name = architecture
    for param in model.parameters():
        param.requires_grad = False
    
    return model
    
    
    
def build_classifier(model, hidden_layer_units):
    
    if hidden_layer_units is None:
        hidden_layer_units = 4096
     
    input_features = model.classifier[0].in_features
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_features, hidden_layer_units, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.4)),
                          ('fc2', nn.Linear(hidden_layer_units, 102, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    return classifier
    
def check_gpu(use_gpu = True):
    
    if use_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    
    return device

def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    
    for ii, (inputs, labels) in enumerate(testloader):
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)
        test_loss += batch_loss.item()
        
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    return test_loss, accuracy
   
def train_network(model, train_dataloaders, validloader,
                                 device, criterion, optimizer, epochs,
                                 print_every, steps):
    
    if epochs is None:
        epochs = 10
    
    
    print("Training process started....\n")
    
    for e in range(epochs):
        running_loss = 0
    
    
        for ii, (inputs, labels) in enumerate(train_dataloaders):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion, device)

                print("Epoch: {}/{} | ".format(e+1, epochs),
                      "Training Loss: {:.4f} | ".format(running_loss/print_every),
                      "Validation Loss: {:.4f} | ".format(valid_loss/len(validloader)),
                      "Validation Accuracy: {:.4f}".format(accuracy/len(validloader)))

                running_loss = 0
                model.train()
    return model

def checkpoint_model( model, path, train_data):
    
    if os.path.isdir(path):
        model.class_to_idx = train_data.class_to_idx
        checkpoint = {
            'architecture' : model.name,
            'classifier' : model.classifier,
            'class_to_idx': model.class_to_idx,
            'state_dict': model.state_dict()
        }

        torch.save(checkpoint, os.path.join(path, model.name + "_" + str(int(time.time())) +"_checkpoint.pth"))
    else:
        print("Path not found, please provide valid path")
    
def main():
    args = arg_parser()
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #apply transform to the training data
    
    training_data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    test_data_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])


    train_data = datasets.ImageFolder(data_dir + '/train', transform=training_data_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_data_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=test_data_transforms) 


    train_dataloaders = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    
    model = get_pretrained_model(architecture=args.architecture)
    
    model.classifier = build_classifier(model, hidden_layer_units = args.hidden_layer_units)
    
    device = check_gpu(use_gpu = args.gpu)
    
    model.to(device)
    
    if type(args.learning_rate) == type(None):
        learning_rate = 0.001
    else: 
        learning_rate = args.learning_rate
        
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    
    print_every = 10
    steps = 0
    
    trained_model = train_network(model, train_dataloaders, validloader,
                                 device, criterion, optimizer, args.epoch,
                                 print_every, steps)
    with torch.no_grad():
        trained_model.eval()
        loss, accuracy = validation(trained_model, testloader, criterion, device)
        print(f"Test Accuracy: {accuracy/len(testloader):.3f}")

    if args.save_dir is not None:
        checkpoint_model(trained_model, args.save_dir, train_data)
        
        
if __name__ == '__main__': main()