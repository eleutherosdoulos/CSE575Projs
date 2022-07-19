import torch
import torch.utils.data as data
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
            'input' : self.normalized_data[idx][0:5],
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

def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    # STUDENTS : note this is how the dataloaders will be iterated over, and cannot be deviated from
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']

if __name__ == '__main__':
    main()
