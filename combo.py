import torch
import torch.utils.data as data
import torch.nn as nn
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('saved/training_data.csv', delimiter=',')
# STUDENTS: it may be helpful for the final part to balance the distribution of your collected data
#note: slack chat implies this may not be necessary, that tweaking the loss function to hate missed collisions (false negatives) does the same
#note: I'm implying this to mean filtering the training_data so that it contains an equal number of collision and noncollision rows
#note: I'm currently using the training data that I didn't create from part 1. The other is available.

        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference

    def __len__(self):
# STUDENTS: __len__() returns the length of the dataset
        return self.data.shape[1] #I think this returns an int, but this is something to check when the file is running
        pass

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
# STUDENTS: for this example, __getitem__() must return a dict with entries {'input': x, 'label': y}
# x and y should both be of type float32. There are many other ways to do this, but to work with autograding
# please do not deviate from these specifications.
#note: ok, feel like I have a better handle on this, outputting a row of the dataframe, as a dict
        item = {
            'input' : self.normalized_data[idx][0:6],
            'label' : self.normalized_data[idx][6]
        }

        return item

class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
# STUDENTS: randomly split dataset into two data.DataLoaders, self.train_loader and self.test_loader
# make sure your split can handle an arbitrary number of samples in the dataset as this may vary
        
        train_set_size = int(len(self.nav_dataset)* 0.8) #eighty percent is prob good, but here's where it lives
        test_set_size = len(self.nav_dataset) - train_set_size
        train_set, test_set = data.random_split(self.nav_dataset, [train_set_size, test_set_size])
        
        self.train_loader = data.DataLoader(train_set)

        self.test_loader = data.DataLoader(test_set)

class Action_Conditioned_FF(nn.Module):
    def __init__(self):
# STUDENTS: __init__() must initiatize nn.Module and define your network's
# custom architecture
        input_size = 6 #warning!
        hidden_size = 4
        hidden2_size = 2
        output_size = 1
        super(Action_Conditioned_FF, self).__init__() 
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        self.nonlinear_activation = nn.Sigmoid()
        self.hidden_to_hidden2 = nn.Linear(hidden_size,hidden2_size)
        self.fc_layer = nn.Linear(hidden2_size, output_size)

    def forward(self, input):
# STUDENTS: forward() must complete a single forward pass through your network
# and return the output which should be a tensor
        hidden = self.input_to_hidden(input.float())
        hidden = self.nonlinear_activation(hidden)
        hidden = self.hidden_to_hidden2(hidden)
        hidden = self.nonlinear_activation(hidden)
        output = self.fc_layer(hidden) 
        return output
    
    def evaluate(self, model, test_loader, loss_function):
# STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
# mind that we do not need to keep track of any gradients while evaluating the
# model. loss_function will be a PyTorch loss function which takes as argument the model's
# output and the desired output.

        loss = 0 

        for batch in test_loader:
            output = torch.flatten(model.forward(batch['input']))
            target = torch.flatten(batch['label'])
            batch_loss = loss_function(output, target)
            loss += batch_loss
        return loss

def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    # STUDENTS : note this is how the dataloaders will be iterated over, and cannot be deviated from
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']

    
    model = Action_Conditioned_FF()

    for batch in data_loaders.test_loader:

        #print(torch.flatten(batch["input"]))
        target= torch.flatten(batch["label"]).float().item()
        output = torch.flatten(model.forward(batch["input"])).float().item()
        l = nn.MSELoss(output, target, reduction='none')
        print(target)
        print(output)
        print(l)

    input = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)

    mae_loss = nn.L1Loss()
    output = mae_loss(input, target)
    #output.backward()
    val = output.item()

    print('input: ', input)
    print('target: ', target)
    print('output: ', val)



if __name__ == '__main__':
    main()