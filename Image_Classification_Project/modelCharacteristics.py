from torchvision import models
from collections import OrderedDict
from torch import nn
from torch import optim

def build_model(architecture):
    print(architecture)
    if architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif architecture == 'densenet':
        model = models.densenet121(pretrained=True)
    else:
        print('architecture support are only vgg16 and densenet')
        sys.exit()
    return model

def new_classifier(model, input_size, output_size, hidden_layers, learning_rate):
    for param in model.parameters():
        param.requires_grad = False
    
    hidden_layers.append(output_size)
    
    deep_layers = nn.ModuleList([nn.Linear(input_size,hidden_layers[0])])
    layers = zip(hidden_layers[:-1], hidden_layers[1:])

    deep_layers.extend([nn.Linear(h1, h2) for h1, h2 in layers])   
    
    network = OrderedDict()
    
    for x in range(len(deep_layers)):
        if x == 0:
            network.update({'fc{}'.format(x):deep_layers[x]})
        else:
            network.update({'relu{}'.format(x):nn.ReLU()})
            network.update({'drop{}'.format(x):nn.Dropout(0.2)}) # we will hardcode the droupout value for now
            network.update({'fc{}'.format(x):deep_layers[x]})
        
    network.update({'output':nn.LogSoftmax(dim=1)})
 
    classifier = nn.Sequential(network)
    
    #Apply criterion and optimizer
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(learning_rate))     
    
    return model, criterion, optimizer