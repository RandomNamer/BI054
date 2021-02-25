#Dataset generating:
data_path='./data/Training/'
label_path='./data/REFERENCE-v3-training.csv'
val_data_path='./data/Validation/'
val_label_path='./data/REFERENCE-v3-validation.csv'
batch_size = 1024
window_size=3000
window_generate_stride=500

#Learning parameters:
lr=0.0001
weight_decay=1e-3
ReduceLRFactor=0.1
ReduceLRPatience=3 
num_epochs=200

#HyperParams:
out_channels=256


#Monitoring the training process:
print_freq=20
loss_weight=[7,1,1.8,20]

#Checkpoint:
checkpoint_path='./checkpoint_2'
save_freq=1

#Debug Features:
NoTestTrainSplit=False
PrintTrainShape=False
PrintEvalIntermediates=True
PrintTestShape=True