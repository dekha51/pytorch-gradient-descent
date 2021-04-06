# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 09:04:24 2020

@author: ASUS
"""
import lasio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

##sumur 1

las = lasio.read(r'puk1.las')

df=las.df()
gr_max=df['GR'].max()
gr_min=df['GR'].min()
df['VSH']=(df['GR']-gr_min)/(gr_max-gr_min)
df['Vp']=1/df['DT']*10**(6)
df['P_imp']=df['Vp']*df['RHOB']

#rename column
df=df.rename(columns={'NPHI_LS':'NPHI'})

#drop column which have no record
df_drop=df.dropna(subset=['P_imp'],axis=0)

#select only at the target area
df_target=df_drop.loc[3360.91:3751.5].reset_index()

df_corr=df_target.corr()

# #standardizing data
# from sklearn.preprocessing import StandardScaler

# sc=StandardScaler().fit(df_target.DT)
# df_target.DT=sc.transform(df_target.DT)
# df_target.NPHI=sc.transform(df_target.NPHI)

#X axis and y axis
X=df_target.DT.to_numpy()
y=df_target.NPHI.to_numpy()

#conditioning data to tensor format
A=[]
for i in X:
    i=[i]
    A.append(i)

b=[]
for i in y:
    i=[i]
    b.append(i)

X = torch.Tensor(A)
y = torch.Tensor(b)


# #data random
# torch.manual_seed(93)
# X = torch.randn(100,1)*10
# y = X + torch.randn(100,1) *5




# class LR(nn.Module):
#     def __init__(self,input_size,output_size):
#         super().__init__()
#         self.linear = nn.Linear(input_size,output_size)
#     def forward(self,x):
#         pred = self.linear(x)
#         return pred
    
# model=LR(1,1)
model=nn.Linear(1,1)
#linear regression
#in features = 1 --> DT (X) parameter
#out features = 1 --> NPHI (y) parameter

#initial model
[a,b]=model.parameters()

# x = torch.tensor([1.0])
# print(model.forward(x))

xa=np.array(X)
a=a[0][0].item()
#since this data is on list inside list
b=b[0].item()


ya=a*xa+b
#ya is y regression

plt.figure()
plt.plot(xa,ya,'k')
plt.scatter(X,y)
plt.xlabel('DT (us/ft)')
plt.ylabel('NPHI (v/v)')

criterion = nn.MSELoss()
#Loss measurement based on Mean Squared Error
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
#Using SGD (Stochastic Gradient Descent) optimizer algoritm



iterations = 101
losses = []
for i in range (iterations):
    y_pred=model.forward(X)
    loss=criterion(y_pred,y)
    print ("Iterations {} Loss {}".format(i,loss))
    losses.append(loss)
    
    optimizer.zero_grad()
    #prevent gradient measurement to accumulate
    loss.backward() 
    #calculate gradient in each iteration
    optimizer.step()
    
plt.figure()
plt.plot(range(iterations),losses)
plt.xlabel('iteration')
plt.ylabel('error')

#obtain the new (a,b) parameters after 100 iterations
[a,b]=model.parameters()
a=a[0][0].item()
#since the default of this parameter is on list inside list
b=b[0].item()
xa=np.array(X)
ya=a*xa+b
plt.figure()
plt.plot(xa,ya,'k')
plt.scatter(X,y)
plt.xlabel('DT (us/ft)')
plt.ylabel('NPHI (v/v)')
