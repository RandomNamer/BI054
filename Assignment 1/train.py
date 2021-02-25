import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from sklearn.metrics import classification_report, confusion_matrix

from cfg import batch_size,window_size,window_generate_stride
from cfg import lr,weight_decay,ReduceLRFactor,ReduceLRPatience,num_epochs
from cfg import out_channels,loss_weight
from cfg import print_freq
from cfg import checkpoint_path
from cfg import PrintTrainShape,PrintTestShape
from utils import *
from crnn import MyDataset,CRNN








x_train, x_test, y_train, y_test = generate_data_4classes_windowed(window_size=window_size, stride=window_generate_stride)
print("Traing set shapes(x,y):",x_train.shape,y_train.shape)
if PrintTestShape: print("Test set shapes(x,y):",x_test.shape,y_test.shape)
if torch.cuda.is_available():
    #print(torch.cuda.current_device(),torch.cuda.device_count())
    cudalabel=torch.cuda.current_device()
    device=torch.device("cuda")
else:
    cudalabel=-1
    device=torch.device("cpu")

dataset_train=MyDataset(torch.tensor(x_train),torch.tensor(y_train),cudalabel)
dataset_test=MyDataset(torch.tensor(x_test),torch.tensor(y_test),cudalabel)

dataloader_train=DataLoader(dataset_train,batch_size=batch_size)
dataloader_test=DataLoader(dataset_test,batch_size=batch_size)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=CRNN(
    in_channels=1,
    out_channels=out_channels,
    n_len_seg=50,
    n_classes=4,
    device=device
)
model.cuda()
model.to(device)
#summary(model,(1,window_size))
#print(model)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=ReduceLRFactor, patience=ReduceLRPatience,verbose=True)
w=torch.FloatTensor(loss_weight).cuda(cudalabel)
loss_func = torch.nn.CrossEntropyLoss(w)

print("Training strat, begin with Lr=",lr)
for epoch in range(num_epochs):
    model.train()
    running_loss=0.0
    for idx,data in enumerate(dataloader_train):
        x_input,y_input=tuple(t.to(device) for t in data)
        y_pred=model(x_input)

        loss=loss_func(y_pred,y_input)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss+=loss.item()
        if idx%print_freq==(print_freq-1):
            print("Epoch ",epoch+1,", Iter ",idx+1,":  Loss=", running_loss/print_freq,"  Lr=",optimizer.state_dict()['param_groups'][0]['lr'])
            running_loss=0.0
    scheduler.step(epoch)
    if not os.path.exists(checkpoint_path): os.mkdir(checkpoint_path)
    torch.save(model.state_dict(),checkpoint_path+'/checkpoint_'+str(epoch+1)+'.pth')
    #Testing:
    model.eval()
    y_pred_all_prob=[]
    with torch.no_grad():
        for idx,data in enumerate(dataloader_test):
            x_input,y_input=tuple(t.to(device) for t in data)
            y_pred=model(x_input)
            y_pred_all_prob.append(y_pred.cpu().data.numpy())
    y_pred_all_prob=np.concatenate(y_pred_all_prob)
    if PrintTestShape: print(y_pred_all_prob)
    y_pred_all=np.argmax(y_pred_all_prob,axis=1)
    if PrintTestShape:
        print('pred and gt Shapes:',y_pred_all.shape,y_test.shape)
        print(y_test,y_pred_all)
    target_names = ['AF', 'Normal', 'Others','Noisy']
    tmp_report = classification_report(y_test, y_pred_all,target_names=target_names,output_dict=True)
    print(tmp_report)
    f1_score = (tmp_report['AF']['f1-score'] + tmp_report['Normal']['f1-score'] + tmp_report['Others']['f1-score'] + tmp_report['Noisy']['f1-score'])/4
    print('F-1 Score: ',f1_score)



    





        
print("Training ends.")
        




