# -*- coding: utf-8 -*-
"""
@author: huseyin.tunc
"""
import scipy.io
from scipy.io import savemat
import numpy as np
import torch 
from torch.utils.data import Subset
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Xs is input for PHYSCO7 features for Stanford Dataset (8 PI, seperately)
mat = scipy.io.loadmat('Xs.mat')
Xs=mat['Xs']
Xs=np.array(Xs[0])

# Ys is output for Stanford Dataset (8 PI, seperately)
mat = scipy.io.loadmat('Ys.mat')
Ys=mat['Ys']
Ys=np.array(Ys[0])

# F_Xs is combined input consisting of PHYSCO7 features and inhibitor representations (Chemprop) for Stanford Dataset.
mat = scipy.io.loadmat('F_Xs.mat')
F_Xs=mat['F_Xs']
F_Xs=np.array(F_Xs)
# F_Ys is the corresponding output (fold change IC50 values)
mat = scipy.io.loadmat('F_Ys.mat')
F_Ys=mat['F_Ys']
F_Ys=np.array(F_Ys)

#EXTERNAL_FULL_PROCESSED_DATA_CHEMPROP.mat file contains necessary files for external dataset.
mat = scipy.io.loadmat('EXTERNAL_FULL_PROCESSED_DATA_CHEMPROP.mat')
#External_F_Xs is the input matrix for external data.
External_F_Xs=mat['F_Xs']
External_F_Xs=np.array(External_F_Xs)
#External_F_Ys is the output vector for external data.
External_F_Ys=mat['F_Ys']
External_F_Ys=np.array(External_F_Ys)


n1 =[20]
n2 =[10]
d_emb=[16]

#ADJ.xlsx is the adjacency matrix representing the Delaunay triangulation of 198 carbon alpha atom positions in the 3OXC crystal structure.
adj=pd.read_excel('ADJ.xlsx',index_col=None, header=None)


def train_val_dataset(dataset, L,T,INTRA):
    train_idx, val_idx = T,L
    val_idx=np.array(val_idx)+21792
    train_idx=np.concatenate((train_idx+21792, np.array(INTRA)), axis=0)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

def datapreprocess_tra_val(Xs,Ys,Xs2,Ys2,adjM,L,T,INTRA):
    X=torch.tensor(Xs,dtype=torch.float32)
    X2=torch.tensor(Xs2,dtype=torch.float32)
    X=torch.cat((X,X2), dim=0)
    #X=torch.reshape(X,(-1,198,8))
    Y=torch.tensor(Ys,dtype=torch.float32)
    Y2=torch.tensor(Ys2,dtype=torch.float32)
    Y=torch.cat((Y,Y2), dim=0)
    data_tensor = data_utils.TensorDataset(X,Y)
    datasets=train_val_dataset(data_tensor,L,T,INTRA)
    train_loader = data_utils.DataLoader(dataset = datasets['train'], batch_size = 256)
    tadj=torch.tensor(adjM,dtype=torch.float32)
    tadj_tra=tadj.repeat(len(datasets['train']),1,1).to(device)
    tadj_val=tadj.repeat(len(datasets['val']),1,1).to(device)
    val_loader = data_utils.DataLoader(dataset = datasets['val'], batch_size = 256)
    return train_loader,val_loader,tadj_tra,tadj_val

def datapreprocess_test(Xs,Ys,adjM):
    X=torch.tensor(Xs,dtype=torch.float32)
    #X=torch.reshape(X,(-1,198,8))
    Y=torch.tensor(Ys,dtype=torch.float32)
    data_tensor = data_utils.TensorDataset(X,Y)
    test_loader = data_utils.DataLoader(dataset = data_tensor, batch_size = len(data_tensor))
    tadj=torch.tensor(adjM,dtype=torch.float32)
    tadj_test=tadj.repeat(len(data_tensor),1,1).to(device)
    return test_loader,tadj_test

def get_external_indexes(A):
    B=[]
    for i in A:
        for j in i:
            for q in j:
                B.append(q-1)
    return B


def get_external_drug_indexes(L,U,B):
    B1=[]
    A1=[]
    for i in range(B):
        if i>=L and i<=U:
            A1.append(i)
        else:
            B1.append(i)
    return A1,B1

def shuf_to_cv(slist,cv):
    rt=1/cv
    L=[]
    T=[]
    m=0
    for i in range(cv):
        
        if i<cv-1:
            L.append(slist[m:math.floor(m+len(slist)*rt)])
            T.append(np.setdiff1d(slist,L[i]))
            m=int(m+len(slist)*rt)
        else:
            L.append(slist[m:])
            T.append(np.setdiff1d(slist,slist[m:]))
                
    
    return L,T 

class GCNLayer(nn.Module):

    def __init__(self, c_in, c_out,hidden1,hidden2,nnf):
        super().__init__()
        self.nnf=nnf
        self.hidden1=hidden1
        self.hidden2=hidden2
        self.a = torch.nn.Parameter(torch.ones((1,198)))
        self.projection = nn.Linear(c_in, c_out)
        self.fc1=nn.Linear(self.nnf*c_out+300,self.hidden1)
        if self.hidden2==0:
            self.fc2=nn.Linear(self.hidden1,1)
        else:
            self.fc2=nn.Linear(self.hidden1,self.hidden2)
            self.fc3=nn.Linear(self.hidden2,1)

    def forward(self, node_feats, adj_matrix,L2):
        M=node_feats[:,198*7:]
        nf=node_feats[:,:198*7]
        node_feats=torch.reshape(nf,(-1,198,7))
        V=self.a.repeat(node_feats.shape[0],198,1).to(device)
        wadj_matrix=adj_matrix[:node_feats.shape[0],:,:]*V
        num_neighbours = adj_matrix[:node_feats.shape[0],:,:].sum(dim=-1, keepdims=True)
        node_feats = self.projection(node_feats)
        node_feats = torch.bmm(wadj_matrix[:node_feats.shape[0],:,:], node_feats)
        node_feats = node_feats / num_neighbours
        node_feats=F.relu(torch.flatten(node_feats,start_dim=1,end_dim=-1))
        node_feats=torch.cat((node_feats,M), 1)
        if L2==0:
            y=self.fc2(torch.tanh(self.fc1(node_feats)))
        else:
            y=self.fc3(torch.tanh(self.fc2(torch.tanh(self.fc1(node_feats)))))
        return y


def trainfun(train_loader,val_loader,test_loader,adj_tra,adj_val,adj_test,cout,L1,L2,lr):
    LTrain_=torch.tensor(0.)
    LTest=torch.tensor(0.)
    LVal=torch.tensor(0.)
    R2_Test=torch.tensor(0.)
    R2_Val=torch.tensor(0.)
    R2_Tra=torch.tensor(0.)
    model = GCNLayer(c_in=7, c_out=cout,hidden1=L1,hidden2=L2,nnf=198).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch=0
    while epoch<=500:
        epoch+=1
        for x,y in train_loader:
            optimizer.zero_grad()
            y_pred = model(x.to(device),adj_tra,L2)
            loss = criterion(y_pred, y.to(device))
            loss.backward()
            optimizer.step()

        """
        if (epoch) % 50 == 0:
            print("Epoch [{}/{}], Loss: {:.4f}".format(epoch, 1000, loss.item()))
        """

        if ((epoch) % 100 == 0) & (loss.item()>0.5):
            break

    with torch.no_grad():
        i=0
        Y_PRED=[]
        for x,y in test_loader:
            i+=1
            optimizer.zero_grad()
            y_pred = model(x.to(device),adj_test,L2)
            Y_PRED.append(y_pred.cpu().detach().numpy())
            LTest += criterion(y_pred.to(device), y.to(device)).cpu()
            R2_Test+=np.corrcoef(np.squeeze(y.cpu().detach().numpy()), np.squeeze(y_pred.cpu().detach().numpy()))[0][1]
        LTest=LTest/i
        R2_Test=R2_Test/i
    with torch.no_grad():
        i=0
        for x,y in val_loader:
            i+=1
            optimizer.zero_grad()
            y_pred = model(x.to(device),adj_val,L2)
            LVal += criterion(y_pred.to(device), y.to(device)).cpu()
            R2_Val+=np.corrcoef(np.squeeze(y.cpu().detach().numpy()), np.squeeze(y_pred.cpu().detach().numpy()))[0][1]
        LVal=LVal/i
        R2_Val=R2_Val/i
    with torch.no_grad():
        i=0
        for x,y in train_loader:
            i+=1
            optimizer.zero_grad()
            y_pred = model(x.to(device),adj_tra,L2)
            LTrain_ += criterion(y_pred.to(device), y.to(device)).cpu()
            R2_Tra+=np.corrcoef(np.squeeze(y.cpu().detach().numpy()), np.squeeze(y_pred.cpu().detach().numpy()))[0][1]
        LTrain_=LTrain_/i
        R2_Tra=R2_Tra/i
    return Y_PRED,LTrain_,LVal,LTest,R2_Tra,R2_Val,R2_Test

def external_test_fun(Xs,Ys,d_emb,n1,n2,adj):
    lr=0.001
    L1=n1[0]
    L2=n2[0]
    cout=int(d_emb[0])

    TR_MSE=np.zeros(5)
    VL_MSE=np.zeros(5)
    TS_MSE=np.zeros(5)
    TR_R2=np.zeros(5)
    VL_R2=np.zeros(5)
    TS_R2=np.zeros(5)
                 
   
    INTST,INTRA=get_external_drug_indexes(-1,-1,F_Xs.shape[0])
    X_T_V=F_Xs[INTRA,:]
    Y_T_V=F_Ys[INTRA,:]

    X_T=External_F_Xs
    Y_T=External_F_Ys
    tensor_list = torch.tensor(list(range(len(Y_T))))
    shuffled_tensor = tensor_list[torch.randperm(len(tensor_list))]
    shuffled_list = shuffled_tensor.tolist()
    V,TR=shuf_to_cv(shuffled_list,5)
                 
    V_indices = []
    TR_indices = []
    YPRED=[]             
    for cvcase in range(5):
        train_loader,val_loader,adj_tra,adj_val=datapreprocess_tra_val(X_T_V,Y_T_V,X_T,Y_T,adj.values,V[cvcase],TR[cvcase],INTRA)
        test_loader,adj_test=datapreprocess_test(X_T,Y_T,adj.values)
                     
        V_indices.append(V)
        TR_indices.append(TR)
                     
        
        lr=0.001
        Y_PRED,TrError,VError,Terror,R2Train,R2Val,R2Test=trainfun(train_loader,val_loader,test_loader,adj_tra,adj_val,adj_test,cout,int(L1),int(L2),lr)
        if TrError>0.5:
            lr=lr/5
            Y_PRED,TrError,VError,Terror,R2Train,R2Val,R2Test=trainfun(train_loader,val_loader,test_loader,adj_tra,adj_val,adj_test,cout,int(L1),int(L2),lr)
            if TrError>0.5:
                lr=lr/5
                Y_PRED,TrError,VError,Terror,R2Train,R2Val,R2Test=trainfun(train_loader,val_loader,test_loader,adj_tra,adj_val,adj_test,cout,int(L1),int(L2),lr)
        TR_MSE[cvcase]=TrError
        VL_MSE[cvcase]=VError
        TS_MSE[cvcase]=Terror
        TR_R2[cvcase]=R2Train
        VL_R2[cvcase]=R2Val
        TS_R2[cvcase]=R2Test
        YPRED.append(Y_PRED)
        mdict={"YPRED":YPRED,"TR_MSE":TR_MSE,"VL_MSE":VL_MSE,"TS_MSE":TS_MSE,"TR_R2":TR_R2,"VL_R2":VL_R2,"TS_R2":TS_R2,"V_indices":V_indices,"TR_indices":TR_indices}
        savemat("EXTER_TEST_RESULTS_WGCN_CPROP.mat", mdict)

    return TR_MSE,VL_MSE,TS_MSE,TR_R2,VL_R2,TS_R2


TR_MSE,VL_MSE,TS_MSE,TR_R2,VL_R2,TS_R2=external_test_fun(Xs,Ys,d_emb,n1,n2,adj)


