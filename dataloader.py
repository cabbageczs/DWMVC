"""
load recovered data
"""

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch


class BDGP(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path)['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path)['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path)['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()


class CCV(Dataset):
    def __init__(self, path):
        self.data1 = np.load(path+'STIP.npy').astype(np.float32)
        scaler = MinMaxScaler()
        self.data1 = scaler.fit_transform(self.data1)
        self.data2 = np.load(path+'SIFT.npy').astype(np.float32)
        self.data3 = np.load(path+'MFCC.npy').astype(np.float32)
        self.labels = np.load(path+'label.npy')

    def __len__(self):
        return 6773

    def __getitem__(self, idx):
        x1 = self.data1[idx]
        x2 = self.data2[idx]
        x3 = self.data3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(
           x2), torch.from_numpy(x3)], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class MNIST_USPS(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32)

    def __len__(self):
        return 5000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Fashion(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Fashion_recover.mat')['Y'].astype(np.int32).reshape(10000,)
        self.V1 = scipy.io.loadmat(path + 'Fashion_recover.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Fashion_recover.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Fashion_recover.mat')['X3'].astype(np.float32)
        

        self.V1 = self.V1.reshape(10000, -1)
        self.V2 = self.V2.reshape(10000, -1)  
        self.V3 = self.V3.reshape(10000, -1) 

    def __len__(self):
        return 10000

    def __getitem__(self, idx):

        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Caltech(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view5[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(
                self.view4[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()

class HandWritten(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = StandardScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view
        
    def __len__(self):
        return 2000
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
            self.view2[idx]), torch.from_numpy(self.view3[idx]), torch.from_numpy(
            self.view4[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(np.array(self.labels[idx])), torch.from_numpy(np.array(idx)).long()

class Scene(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = StandardScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view
    
    def __len__(self):
        return 4485
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
            self.view2[idx]), torch.from_numpy(self.view3[idx]),],torch.from_numpy(np.array(self.labels[idx])), torch.from_numpy(np.array(idx)).long()
        
class aloideep3v(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = StandardScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view
    
    def __len__(self):
        return 10800
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
            self.view2[idx]), torch.from_numpy(self.view3[idx])],torch.from_numpy(np.array(self.labels[idx])), torch.from_numpy(np.array(idx)).long()
      
class Caltech101_7(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.view6 = scaler.fit_transform(data['X6'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view
    
    def __len__(self):
        return 1474
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
            self.view2[idx]), torch.from_numpy(self.view3[idx]), torch.from_numpy(self.view4[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(self.view6[idx])
            ],torch.from_numpy(np.array(self.labels[idx])), torch.from_numpy(np.array(idx)).long()
        
        
class CIFAR10(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = StandardScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view
    
    def __len__(self):
        return 30000
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
            self.view2[idx]), torch.from_numpy(self.view3[idx])],torch.from_numpy(np.array(self.labels[idx])), torch.from_numpy(np.array(idx)).long()
      
    
def load_data(data_path,dataset):
    if dataset == "HandWritten":
        dataset = HandWritten(data_path + '_recover.mat',view=5)
        dims = [240, 76, 216, 47, 64]
        view = 5 
        data_size = 2000
        class_num = 10  
    elif dataset == "Fashion":
        dataset = Fashion('./data/recover_data/')
        dims = [784, 784, 784]
        view = 3
        data_size = 10000
        class_num = 10
    elif dataset == "Scene-15":
        dataset = Scene(data_path + '_recover.mat',view=3)
        dims = [20, 59, 40]
        view = 3
        data_size = 4485
        class_num = 15
    elif dataset == "aloideep3v":
        dataset = aloideep3v(data_path + '_recover.mat',view=3)
        dims = [2048, 4096, 2048]
        view = 3
        data_size = 10800
        class_num = 100
    elif dataset =="Caltech101-7":
        dataset = Caltech101_7(data_path + '_recover.mat',view=6)
        dims = [48, 40, 254, 1984, 512, 928]
        view = 6
        data_size = 1474
        class_num = 7
    elif dataset == "BDGP":
        dataset = BDGP(data_path + '_recover.mat')
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "CIFAR10":
        dataset = CIFAR10(data_path + '_recover.mat', view=3)
        dims = [512, 2048, 1024]
        view = 3
        data_size = 30000
        class_num = 10
    elif dataset == "MNIST-USPS":
        dataset = MNIST_USPS('./data/')
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 5000
    elif dataset == "CCV":
        dataset = CCV('./data/')
        dims = [5000, 5000, 4000]
        view = 3
        data_size = 6773
        class_num = 20
    elif dataset == "Caltech-2V":
        dataset = Caltech('data/Caltech-5V.mat', view=2)
        dims = [40, 254]
        view = 2
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-3V":
        dataset = Caltech('data/Caltech-5V.mat', view=3)
        dims = [40, 254, 928]
        view = 3
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-4V":
        dataset = Caltech('data/Caltech-5V.mat', view=4)
        dims = [40, 254, 928, 512]
        view = 4
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-5V":
        dataset = Caltech('data/Caltech-5V.mat', view=5)
        dims = [40, 254, 928, 512, 1984]
        view = 5
        data_size = 1400
        class_num = 7  
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
