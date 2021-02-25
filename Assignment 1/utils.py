import  matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import scipy.io as scio
from scipy.fftpack import fft,ifft
from cfg import PrintTrainShape, data_path,label_path,NoTestTrainSplit,val_data_path,val_label_path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

'''
data_path='./data/Training/'
label_path='./data/REFERENCE-v3-training.csv'
'''

#print(label.head)
def load_raw():
    label=pd.read_csv(label_path,names = ["filename", "class"])
    #print(label.head) 
    print("Loading files...")
    raw=[]
    for filename in tqdm(label['filename'].values):
        gt=label[label['filename']==filename]['class'].values[0]
        filepath=data_path+filename+'.mat'
        data=scio.loadmat(filepath)
        data=np.array(data['val'],dtype='int16')[0]
        raw.append({'data':data,'class':gt})
    return raw

def load_raw_val(path=val_label_path,path_data=val_data_path):
    label=pd.read_csv(path,names = ["filename", "class"])
    #print(label.head) 
    print("Loading files...")
    raw=[]
    for filename in tqdm(label['filename'].values):
        gt=label[label['filename']==filename]['class'].values[0]
        filepath=path_data+filename+'.mat'
        data=scio.loadmat(filepath)
        data=np.array(data['val'],dtype='int16')[0]
        raw.append({'data':data,'class':gt,'filename':filename})
    return raw

def generate_data_4classes_windowed(window_size, stride):
    data_raw=load_raw()
    x_out=[]
    y_out=[]
    print("Generating dataset...")
    for  i in range(len(data_raw)):
        x_raw=data_raw[i]['data']
        y_raw=data_raw[i]['class']
        if y_raw=='A': y_raw=0
        elif y_raw=='N': y_raw=1
        elif y_raw=='O': y_raw=2
        elif y_raw=='~': y_raw=3
        else:
            print('Error,label not recognized: ',y_raw)
        if len(x_raw)<=window_size:continue
        for j in range(0, len(x_raw)-window_size, stride):
            x_out.append(x_raw[j:j+window_size])
            y_out.append(y_raw)
    x_out=np.array(x_out);y_out=np.array(y_out)
    print("Orignial: ",len(data_raw)," After: ",x_out.shape,y_out.shape)
    if NoTestTrainSplit: 
        x_train=x_out;y_train=y_out
        x_test=[[]];y_test=[]
    else:
        x_train,x_test,y_train,y_test=train_test_split(x_out, y_out, test_size=0.1,random_state=0)
    #Shuffle Training Set:
    shuffle_id = np.random.permutation(y_train.shape[0])
    x_train = x_train[shuffle_id]
    y_train = y_train[shuffle_id]

    #Generate 2-D Tensor:
    x_train = np.expand_dims(x_train, 1)
    x_test = np.expand_dims(x_test, 1)

    return x_train, x_test, y_train, y_test

def generate_data_4classes_windowed_val(window_size,stride,path=val_label_path,path_data=val_data_path):
    data_raw=load_raw_val(path,path_data)
    x_out=[]
    y_out=[]
    z_out=[]
    print("Generating validation dataset...")
    for  i in range(len(data_raw)):
        x_raw=data_raw[i]['data']
        y_raw=data_raw[i]['class']
        z_raw=data_raw[i]['filename']
        if y_raw=='A': y_raw=0
        elif y_raw=='N': y_raw=1
        elif y_raw=='O': y_raw=2
        elif y_raw=='~': y_raw=3
        else:
            print('Error,label not recognized: ',y_raw)
        #if path_data==val_data_path: print(z_raw,y_raw)
        index_start=len(y_out)
        if len(x_raw)<=window_size: continue
        for j in range(0, len(x_raw)-window_size, stride):
            x_out.append(x_raw[j:j+window_size])
            y_out.append(y_raw)
        index_end=len(y_out)
        z_out.append({'sample_name':z_raw,'index_start':index_start,'index_end':index_end})
    x_out=np.array(x_out);y_out=np.array(y_out)
    print("Orignial: ",len(data_raw)," After: ",x_out.shape,y_out.shape)
    
    #Generate 2-D Tensor:
    x_out = np.expand_dims(x_out, 1)
    if PrintTrainShape: print("Generated Val Dataset shapes(x,y,z):",x_out.shape,y_out.shape,len(z_out))
    return x_out,y_out,z_out


def generate_data_4_padding(size):
    return


            

    

