import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import sklearn
from sklearn.svm import LinearSVC,SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


from cfg import batch_size,window_size,window_generate_stride
from cfg import lr,weight_decay,ReduceLRFactor,ReduceLRPatience,num_epochs
from cfg import print_freq
from cfg import checkpoint_path
from utils import *
from cfg import PrintEvalIntermediates
from crnn import MyDataset,CRNN


class ClsBackend():
    def __init__(self,x,y):
        self.x=x
        self.y=y

        self.model=SVC(C=1.3,class_weight='balanced')
    def fit(self):
        self.model.fit(self.x,self.y)
    def score(self,x_test,y_test):
        s=self.model.score(x_test,y_test)
        return s

    def pred(self,x_test):
        y_pred=self.model.predict(x_test)
        return y_pred
    




def EvalFromVal(y_test,z_test,y_pred):
    #Confusion matrix
    Normal={'Nn':0,'Na':0,'No':0,'Np':0}
    A_F={'An':0,'Aa':0,'Ao':0,'Ap':0}
    Other={'On':0,'Oa':0,'Oo':0,'Op':0}
    Noisy={'Pn':0,'Pa':0,'Po':0,'Pp':0}
    Total_gt={'N':0,'A':0,'O':0,'P':0}
    Total_pred={'n':0,'a':0,'o':0,'p':0}

    for i in range(len(z_test)):   
        sample_name=z_test[i]['sample_name']
        
        y_test_sample=[]
        y_pred_sample=[]
        for j in range(z_test[i]['index_start'],z_test[i]['index_end']):
            y_test_sample.append(y_test[j])
            y_pred_sample.append(y_pred[j])
        counts=[y_pred_sample.count(j) for j in range(4)]
        sample_result=counts.index(max(counts))
        sample_result_gt=y_test_sample[0]
        if PrintEvalIntermediates: print("Sample Name:",sample_name," Result:",sample_result)
        if sample_result==0:
            Total_pred['a']+=1
            if sample_result_gt==0: 
                A_F['Aa']+=1
        elif sample_result==1:
            Total_pred['n']+=1
            if sample_result_gt==1:
                Normal['Nn']+=1
        elif sample_result==2:
            Total_pred['o']+=2
            if sample_result_gt==2:
                Other['Oo']+=1
        elif sample_result==3:
            Total_pred['p']+=1
            if sample_result_gt==1:
                Normal['Pp']+=1
        else: print('Error! pred label not defined.')
        if sample_result_gt==0:
            Total_gt['A']+=1
        elif sample_result_gt==1:
            Total_gt['N']+=1
        elif sample_result_gt==2:
            Total_gt['O']+=2
        elif sample_result_gt==3:
            Total_gt['P']+=1
        else: print('Error! gt label',sample_result_gt,'not defined.')
    if PrintEvalIntermediates: print('Confusion matrix\n',Normal,'\n',A_F,'\n',Other,'\n',Noisy,'\n',Total_gt,'\n',Total_pred)
    F1n=(2*Normal['Nn'])/(Total_gt['N']+Total_pred['n'])
    F1a=(2*A_F['Aa'])/(Total_gt['A']+Total_pred['a'])
    F1o=(2*Other['Oo'])/(Total_gt['O']+Total_pred['o'])
    F1p=(2*Noisy['Pp'])/(Total_gt['P']+Total_pred['p'])
    F1=(F1a+F1n+F1o+F1p)/4
    Acc=(Normal['Nn']+A_F['Aa']+Other['Oo']+Noisy['Pp'])/(Total_gt['N']+Total_gt['A']+Total_gt['O']+Total_gt['P'])
    print(F1,Acc)
    return F1,Acc,F1n,F1a,F1o,F1p

        

        
