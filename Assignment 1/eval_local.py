import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize

from cfg import batch_size,window_size,window_generate_stride,label_path
from cfg import lr,weight_decay,ReduceLRFactor,ReduceLRPatience,num_epochs
from cfg import out_channels,loss_weight
from cfg import print_freq
from cfg import checkpoint_path
from cfg import PrintTrainShape,PrintTestShape,PrintEvalIntermediates
from utils import *
from eval import EvalFromVal
from crnn import MyDataset,CRNN
from eval import ClsBackend


checkpoint='./checkpoint_3_0.32/checkpoint_39.pth'

x_train,y_train,z_train=generate_data_4classes_windowed_val(window_size=window_size, stride=window_generate_stride,path=label_path,path_data=data_path)
x,y,z= generate_data_4classes_windowed_val(window_size=window_size, stride=window_generate_stride)

if torch.cuda.is_available():
    #print(torch.cuda.current_device(),torch.cuda.device_count())
    cudalabel=torch.cuda.current_device()
    device=torch.device("cuda")
else:
    cudalabel=-1
    device=torch.device("cpu")

dataset_test=MyDataset(torch.tensor(x),torch.tensor(y),cudalabel)
dataset_train=MyDataset(torch.tensor(x_train),torch.tensor(y_train),cudalabel)
dataloader_test=DataLoader(dataset_test,batch_size=batch_size)
dataloader_train=DataLoader(dataset_train,batch_size=batch_size)
model=CRNN(
    in_channels=1,
    out_channels=out_channels,
    n_len_seg=50,
    n_classes=4,
    device=device
)
print(model)
model.load_state_dict(torch.load(checkpoint))
model.cuda()
model.to(device)
model.eval()
#Get Trainset outputs:
y_pred_all_prob=[]
with torch.no_grad():
    for idx,data in enumerate(dataloader_train):
        x_input,y_input=tuple(t.to(device) for t in data)
        y_pred=model(x_input)
        y_pred_all_prob.append(y_pred.cpu().data.numpy())
y_pred_all_prob=np.concatenate(y_pred_all_prob)

y_pred_all=np.argmax(y_pred_all_prob,axis=1)
'''
if PrintTestShape:
    print('pred and gt Shapes:',y_pred_all.shape,y_train.shape,len(z_train))
    print(y,y_pred_all,z_train)
'''
x_train_cls=[]
y_train_cls=[]
for i in range(len(z_train)):
    sample_name=z_train[i]['sample_name']
    y_train_sample=[]
    y_pred_sample=[]
    for j in range(z_train[i]['index_start'],z_train[i]['index_end']):
        y_train_sample.append(y_train[j])
        y_pred_sample.append(y_pred_all[j])
    if not len(y_train_sample): continue
    y_train_value=y_train_sample[0]
    y_train_cls.append(y_train_value)
    #if PrintEvalIntermediates: print(sample_name,y_train_cls[i],y_train_sample)
    counts=[y_pred_sample.count(j) for j in range(4)]
    sample_len=len(y_pred_sample)
    x_train_cls.append([counts[i]/sample_len for i in range(4)])
#y_train_cls=label_binarize(y_train_cls,np.arange(4))
if PrintEvalIntermediates: print("Train set cls:",np.shape(x_train_cls),np.shape(y_train_cls))
cls=ClsBackend(x_train_cls,y_train_cls)
cls.fit()
print("Score:",cls.score(x_train_cls,y_train_cls))


x_test_cls=[]
y_test_cls=[]
for i in range(len(z)):
    sample_name=z[i]['sample_name']
    y_test_sample=[]
    y_pred_sample=[]
    for j in range(z[i]['index_start'],z[i]['index_end']):
        y_test_sample.append(y[j])
        y_pred_sample.append(y_pred_all[j])
    if not len(y_test_sample): continue
    y_test_cls.append(y_test_sample[0])
    #if PrintEvalIntermediates: print(sample_name,y_test_cls[i],y_test_sample)
    counts=[y_pred_sample.count(j) for j in range(4)]
    sample_len=len(y_pred_sample)
    x_test_cls.append([counts[i]/sample_len for i in range(4)])

#y_test_cls=label_binarize(y_test_cls,np.arange(4))
if PrintEvalIntermediates: 
    print("Val set cls:",np.shape(x_test_cls),len(y_test_cls))
#    print(x_test_cls)
print("Score:",cls.score(x_test_cls,y_test_cls))
y_pred_cls=cls.pred(x_test_cls)
if PrintEvalIntermediates: print(y_pred_cls)
report=classification_report(y_test_cls, y_pred_cls,target_names=['AF', 'Normal', 'Others','Noisy'],output_dict=True)
print(report)





'''
F1,Acc,F1n,F1a,F1o,F1p=EvalFromVal(y,z,y_pred_all)
print('F1:',F1,'\nAcc: ',Acc,'N,A,O,P F1s: ',F1n,F1a,F1o,F1p)
'''