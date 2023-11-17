import torch
torch.cuda.empty_cache()
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split # # pip install sklearn==0.23.1 
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm ## Progress bar
from sklearn.preprocessing import OneHotEncoder # need to upgrade to new version
from torch.utils.data import Dataset, random_split, DataLoader
import scipy.io
import os


def load_tensor(g, data, balance=False):
    df = data.loc[data.iloc[:,2701]==g,:]
    init = True
    for i in range(min(333,df.shape[0]//1000)): # 3 is for sample testing
        if init:
            x = torch.load("./data/x_"+g+"_sub"+str(i)+".pth")
            y = torch.load("./data/y_"+g+"_sub"+str(i)+".pth")
            p = torch.load("./data/pid_"+g+"_sub"+str(i)+".pth")
            x_spec = torch.load("./data/x_spec_"+g+"_sub"+str(i)+".pth")
            init = False
        else: 
            x_sub = torch.load("./data/x_"+g+"_sub"+str(i)+".pth")
            x_spec_sub = torch.load("./data/x_spec_"+g+"_sub"+str(i)+".pth")
            y_sub = torch.load("./data/y_"+g+"_sub"+str(i)+".pth")
            p_sub = torch.load("./data/pid_"+g+"_sub"+str(i)+".pth")
            x = torch.cat((x, x_sub), dim=0)
            y = torch.cat((y, y_sub), dim=0)
            p = torch.cat((p, p_sub), dim=0)
            x_spec = torch.cat((x_spec, x_spec_sub), dim=0)

    if balance:
        y_label = np.argmax(y, axis=1).to('cpu').numpy()
        unique, counts = np.unique(y_label, return_counts=True)
        size_per_group = np.floor(np.mean(counts))
        sampled_idx = []
        for l in range(4):
            replace = True
            y_label_group_idx = np.where(y_label==l)[0].astype(int)
            if len(y_label_group_idx)>size_per_group:
                replace = False
            sampled_idx.extend(np.random.choice(y_label_group_idx, size = size_per_group.astype(int), replace = replace))
        y = y[sampled_idx]
        p = p[sampled_idx]
        x = x[sampled_idx]
        x_spec = x_spec[sampled_idx]
        
    return x, x_spec, y, p



def train_model(model, optimizer, train_loader, valid_loader, loss_module, num_epochs, device,
                train_pid, valid_pid):
    

    # Set model to train mode
    model.train()

    # Training loop
    train_loss_trace = list()
    train_acc_trace = list()
    train_f1_trace = list()
    train_f1_c_trace = list()
    valid_loss_trace = list()
    valid_acc_trace = list()
    valid_f1_trace = list()
    valid_f1_c_trace = list()

    for epoch in tqdm(range(num_epochs)):
        
        for data_inputs, data_labels in train_loader:
            
            ## Step 1: Move input data to device (only strictly necessary if we use GPU)
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)

            ## Step 2: Run the model on the input data
            preds = model(data_inputs)
            # preds = preds.squeeze(dim=1) # Output is [Batch size, 1], but we want [Batch size]

            ## Step 3: Calculate the loss
            # loss = loss_module(preds, data_labels.float())
            loss = loss_module(preds, data_labels)

            ## Step 4: Perform backpropagation
            # Before calculating the gradients, we need to ensure that they are all zero.
            # The gradients would not be overwritten, but actually added to the existing ones.
            optimizer.zero_grad()
            # Perform backpropagation
            loss.backward()

            ## Step 5: Update the parameters
            optimizer.step()

        # monitor evaluation matrics on train / valid sets for each epoch
        print("--- train ---")
        train_loss, train_acc, train_f1, train_f1_c = eval_model(model, train_loader, loss_module, device, train_pid)
        print("--- valid ---")
        valid_loss, valid_acc, valid_f1, valid_f1_c = eval_model(model, valid_loader, loss_module, device, valid_pid)
        print("\n")
        # early stop criteria here
        if len(train_f1_trace)>0:
            if train_f1 < 0.1*max(train_f1_trace): # if current f1 drop 10% of the best, stop the training
                print("early stop because malfunctioned training performance!")
                break
            # if training performance way surpass validation performance (0.2 in f1 scale)
            if valid_f1 <= train_f1 - 0.3:
                print("early stop because potential overfitting, train >> valid!")
                break
        
        
        train_loss_trace.append(train_loss)
        train_acc_trace.append(train_acc)
        train_f1_trace.append(train_f1)
        train_f1_c_trace.append(train_f1_c)
        valid_loss_trace.append(valid_loss)
        valid_acc_trace.append(valid_acc)
        valid_f1_trace.append(valid_f1)
        valid_f1_c_trace.append(valid_f1_c)
        
        
#     # for dubugging purpose only
#     for parameter in model.parameters(): print(parameter)

    return train_loss_trace, valid_loss_trace, train_acc_trace, valid_acc_trace, train_f1_trace, valid_f1_trace, train_f1_c_trace, valid_f1_c_trace


# define the evaluation matrix as physionet challenge 2017
def eval_model(model, data_loader, loss_module, device, data_pid):
    
    model.eval() # Set model to eval mode
    true_preds, num_preds = 0., 0.

    with torch.no_grad(): # Deactivate gradients for the following code
        preds = torch.empty(0,4).to(device)
        trues = torch.empty(0,4).to(device)
        pred_ls = torch.empty(0).to(device)
        data_ls = torch.empty(0).to(device)


        for data_inputs, data_labels in data_loader:

            # Determine prediction of model on dev set
            data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)

            pred_l = torch.max(model(data_inputs).data, 1)[1]
            data_l = torch.max(data_labels, 1)[1] # convert onehot responce to labels

            # append pred
            pred_ls = torch.cat((pred_ls, pred_l), 0)
            data_ls = torch.cat((data_ls, data_l), 0)

            # concatenate predicted values of all batchs of current epoch, is there more efficient way of collecting predictions?
            preds = torch.cat((preds, model(data_inputs)), 0)
            trues = torch.cat((trues, data_labels), 0)

            # Keep records of predictions for the accuracy metric (true_preds=TP+TN, num_preds=TP+TN+FP+FN)
            true_preds += (pred_l == data_l).sum()
            num_preds += data_labels.shape[0]

    # f1 # ["~","A","O","N"] = [3, 0, 2, 1]
    f1_0 = 2*( (pred_ls==0) & (data_ls==0) ).sum() / ((pred_ls==0).sum() + (data_ls==0).sum() )
    f1_1 = 2*( (pred_ls==1) & (data_ls==1) ).sum() / ((pred_ls==1).sum() + (data_ls==1).sum() )
    f1_2 = 2*( (pred_ls==2) & (data_ls==2) ).sum() / ((pred_ls==2).sum() + (data_ls==2).sum() )
    f1_3 = 2*( (pred_ls==3) & (data_ls==3) ).sum() / ((pred_ls==3).sum() + (data_ls==3).sum() )
    f1 = (f1_0 + f1_1 + f1_2 + f1_3)/4
    
    # f1_c (f1 score clustered by patient)
    data_pid = data_pid.to("cpu").numpy()
    pred_ls = pred_ls.to("cpu").numpy()
    data_ls = data_ls.to("cpu").numpy()
    res_df = pd.DataFrame({'pid':data_pid, 'pred':pred_ls, 'true':data_ls})
    def clustered_pred(df):
        # ["~","A","O","N"] = [3, 0, 2, 1]
        if any(df.pred == 0):
            df.pred = 0
        if any(df.pred == 2):
            df.pred = 2
        if any(df.pred == 1):
            df.pred = 1
        if any(df.pred == 3):
            df.pred = 3
        if any(df.true == 0):
            df.true = 0
        if any(df.true == 2):
            df.true = 2
        if any(df.true == 1):
            df.true = 1
        if any(df.true == 3):
            df.true = 3
        return df.drop_duplicates()

    res_df = res_df.groupby('pid', group_keys=False).apply(clustered_pred).reset_index()
    pred_ls_c = torch.tensor(np.array(res_df['pred']))
    data_ls_c = torch.tensor(np.array(res_df['true']))
    
    f1_0_c = 2*( (pred_ls_c==0) & (data_ls_c==0) ).sum() / ((pred_ls_c==0).sum() + (data_ls_c==0).sum() )
    f1_1_c = 2*( (pred_ls_c==1) & (data_ls_c==1) ).sum() / ((pred_ls_c==1).sum() + (data_ls_c==1).sum() )
    f1_2_c = 2*( (pred_ls_c==2) & (data_ls_c==2) ).sum() / ((pred_ls_c==2).sum() + (data_ls_c==2).sum() )
    f1_3_c = 2*( (pred_ls_c==3) & (data_ls_c==3) ).sum() / ((pred_ls_c==3).sum() + (data_ls_c==3).sum() )
    f1_c = (f1_0_c + f1_1_c + f1_2_c + f1_3_c)/4

    # accuracy
    acc = true_preds / num_preds # this is a tensor object
    acc = acc.to('cpu').numpy().reshape(-1)[0] # this is probably a stupid way to get the value out of a tensor

    # loss
    loss = loss_module(preds, trues)
    loss = loss.to('cpu').numpy().reshape(-1)[0] # this is probably a stupid way to get the value out of a tensor
   #     # for dubugging purpose only
#     print(preds)
#     print(trues)
        
    print(f" accuracy: {100.0*acc:4.2f}%" + f" loss: {loss:1.4f}" + f" f1: {f1:1.4f}" + f" clustered_f1: {f1_c:1.4f}")
    return loss, acc, f1, f1_c