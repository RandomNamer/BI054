#A 1-D Deep Convlutional network implementation:


import numpy as np

from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader    

from cfg import PrintTrainShape

class MyDataset(Dataset): 
                                                                                                
    def __init__(self, data, label, cudalabel=-1):                                                                                   
        self.data = data                                                                                               
        self.label = label  
        self.cudalabel=cudalabel                                                                                           
                                                                                                                       
    def __getitem__(self, index):   
        if self.cudalabel==-1:                                                                            
            return (torch.tensor(self.data[index], dtype=torch.float), torch.tensor(self.label[index], dtype=torch.long))  
        else:
            return (torch.tensor(self.data[index], dtype=torch.float).cuda(self.cudalabel), torch.tensor(self.label[index], dtype=torch.long).cuda(self.cudalabel))  
                                                                                                                       
    def __len__(self):                                                                                                 
        return len(self.data)  
class 