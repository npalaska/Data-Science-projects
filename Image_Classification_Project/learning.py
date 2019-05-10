import torch

def train_network(model, criterion, optimizer, epochs, trainloader, validationloader, gpu):
    print_every = 40
    steps = 0

    # convert model to use cuda
    if gpu:          
        model.to('cuda')
    else:
        model.to('cpu') 

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            if gpu:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            else:
                inputs, labels = inputs.to('cpu'), labels.to('cpu')
        
            optimizer.zero_grad()
        
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
            
                valid_loss = 0
            
                with torch.no_grad():
                    for data in validationloader:
                        images, labels = data
                        if gpu:
                            inputs, labels = images.to('cuda'), labels.to('cuda')
                        else:
                            inputs, labels = images.to('cpu'), labels.to('cpu')
                    
                        out = model.forward(inputs)
                        valid_loss += criterion(out, labels).data.item()
                    
                print("Epoch: {}/{}... ".format(e+1, epochs),"Train Loss: {:.4f}".format(running_loss/print_every),"Valid Loss:      {:.4f}".format(valid_loss/print_every))
            
                running_loss = 0
            
        correct = 0
        total = 0
        with torch.no_grad():
            for data in validationloader:
                images, labels = data
                if gpu:
                    inputs, labels = images.to('cuda'), labels.to('cuda')
                else:
                    inputs, labels = images.to('cpu'), labels.to('cpu')
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the validation images: %d %%' % (100 * correct / total))

def save_checkpoint(model, input_size, epochs, save_dir, architecture, learning_rate, class_idx, optimizer, output_size):
    
    checkpoint = {
    'input_size':input_size,
    'epochs':epochs,
    'architecture':architecture,
    'hidden_units':[each.out_features for each in model.classifier if hasattr(each, 'out_features') == True],
    'output_size': output_size,
    'learning_rate': learning_rate,
    'class_to_idx': class_idx,
    'optimizer_dict': optimizer.state_dict(),
    'classifier': model.classifier,
    'state_dict': model.state_dict() 
    }
    torch.save(checkpoint, save_dir)