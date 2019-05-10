import sys
import argparse
import dataloader, modelCharacteristics, learning

def main():
    
    # Common Use cases: 
    # python train.py flowers --save_dir "checkpoint.pth" --hidden_units 512 --arch "vgg16" --learning_rate 0.0001 --epochs 10
    # python train.py flowers --save_dir "checkpoint.pth" --hidden_units 512,256 --arch "vgg16" --learning_rate 0.01 --epochs 10
    
    parser = argparse.ArgumentParser()
    #Define arguments
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--save_dir', default='checkpoint.pth')
    parser.add_argument('--learning_rate', default=0.0001 )
    parser.add_argument('--hidden_units', default='512', type=str)
    parser.add_argument('--arch', default='vgg16')
    parser.add_argument('--output_size', default=102, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--gpu', default=True)          
    
    args = parser.parse_args()                    
    # Create data loaders for training purpose 
    idx, trainloader, testloader, validationloader = dataloader.create_dataLoaders(args.data_dir)
    
    # build the model 
    model = modelCharacteristics.build_model(parser.parse_args().arch)
   
    hidden_layers = args.hidden_units.split(',')
    hidden_layers = [int(x) for x in hidden_layers]   
                        
    input_size = 25088  # since default architecture is vgg16    
    
    if parser.parse_args().arch == "vgg16":
        model, criterion, optimizer = modelCharacteristics.new_classifier(model, input_size, args.output_size, hidden_layers, args.learning_rate)
    elif parser.parse_args().arch == "densenet121":
       input_size = 1024
       model, criterion, optimizer = model-characteristics.new_classifier(model, input_size, args.output_size, hidden_layers, args.learning_rate)
    else: 
        print("Please Select between vgg16 or densenet121")
        sys.exit()
                        
    train_model = learning.train_network(model, criterion, optimizer, args.epochs, trainloader, validationloader, args.gpu)             
    learning.save_checkpoint(model, input_size, args.epochs, args.save_dir, args.arch, args.learning_rate, idx, optimizer, args.output_size)
    pass

if __name__ == '__main__':
    main()