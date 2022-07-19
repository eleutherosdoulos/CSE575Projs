from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def train_model(no_epochs):

    #setup, batch is the number of rows supplied by each pass of the data loaders for gradient update
    batch_size = 16 #generally, smaller is better, but it takes longer
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()


    learning_rate = 0.0001 #fiddled with this, it started at .01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.zero_grad() 

    weight = torch.FloatTensor([1.7]) # overweights collisions by 30x
    loss_function = nn.BCEWithLogitsLoss(pos_weight=weight)

    losses = []
    min_loss = model.evaluate(model, data_loaders.test_loader, loss_function) 
    losses.append(min_loss)

    epochs = range(1,no_epochs+1)
    train_losses = []
    test_losses = []

    for epoch_i in range(no_epochs):
        model.train()
        for idx, sample in enumerate(data_loaders.train_loader): # sample['input'] and sample['label']
            
            network_output = torch.flatten(model.forward(sample['input'])).float()
            target_output = torch.flatten(sample['label']).float() 
            loss = loss_function(network_output, target_output)
            loss.backward()
            optimizer.step()


        #for loss tracking graph
        train_loss = model.evaluate(model, data_loaders.train_loader, loss_function)
        train_losses.append(train_loss)
        test_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
        test_losses.append(test_loss)

    torch.save(model.state_dict(), 'saved/saved_model.pkl', _use_new_zipfile_serialization=False)

    print(min(test_losses))
    plt.plot(epochs,train_losses, label = "Train Loss")
    plt.plot(epochs, test_losses, label = "Test Loss")
    plt.xlabel('Epoch')
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    

if __name__ == '__main__':
    no_epochs = 30 #there are diminishing returns to increasing this
    train_model(no_epochs)
