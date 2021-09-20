#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV, train_test_split
import torch
from tqdm import trange
from IPython.display import clear_output


# In[2]:


df=pd.read_csv('/home/mohammadreza/Kaggle/ATLAS-Higgs-Train/datasets/training.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.isnull().sum()


# In[7]:


df.hist(bins=50, figsize=(30, 25))
plt.show()


# In[4]:


from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
df['Labels']=label.fit_transform(df['Label'])
df.drop('Label', axis=1, inplace=True)


# In[5]:


df['Labels']
#s=1 , b=1


# In[6]:


corr_mx=df.corr()
print(corr_mx['Labels'].sort_values(ascending=False))


# In[7]:


#Let's drop some useless data 
df.drop(columns=['EventId','PRI_lep_eta','PRI_tau_eta','PRI_tau_phi'], axis=1, inplace=True)


# In[8]:


df.columns


# In[13]:


df.info()


# In[14]:


import seaborn as sns

data= df[df.columns[1:28]]

for i in range(0, data.shape[1]):
    plt.figure()
    sns.distplot(data.iloc[:, i], bins=40,color="r")
    
plt.show()


# In[9]:


train, test = train_test_split(df, test_size=0.2, random_state=42)
print('train set shape: ', train.shape)
print('---'*10)
print('test set shape: ', test.shape)


# In[10]:


X_train=train.drop('Labels', axis=1, inplace=False)
y_train=train['Labels'].copy()


# In[11]:


y_train


# In[87]:


result_tab=pd.DataFrame({
    'Model':[],
    'Accuracy':[],
    'Recall score':[],
    'Precision score':[],
    'F1 score':[],
    'ROC_AUC score':[],
})
result_tab.shape


# In[23]:


def model_train(model, X, y, cv=10):
    model.fit(X, y)
    model_score=cross_val_predict(model, X, y, cv=cv)
    acc=accuracy_score(model_score, y)
    recall=recall_score(model_score, y)
    prec=precision_score(model_score, y)
    f1=f1_score(model_score, y)
    roc=roc_auc_score(model_score, y)
    
    row=result_tab.shape[0]
    result_tab.loc[row]=['%s' %model, acc, recall, prec, f1, roc]
    
    return result_tab


# In[24]:


scale=StandardScaler()
X_train_scale=scale.fit_transform(X_train)


# In[25]:


log_reg=LogisticRegression(penalty='none', max_iter=100, C=1)
model_train(log_reg, X_train_scale, y_train)


# In[26]:


knn=KNeighborsClassifier(n_neighbors=2)
model_train(knn, X_train_scale, y_train)


# In[27]:


tree=DecisionTreeClassifier()
model_train(tree, X_train_scale, y_train)


# In[28]:


log_reg_default=LogisticRegression()
model_train(log_reg_default, X_train_scale, y_train)


# In[29]:


random_forest=RandomForestClassifier()
model_train(random_forest, X_train_scale, y_train)


# In[30]:


sgd=SGDClassifier()
model_train(sgd, X_train_scale, y_train)


# In[35]:


#from sklearn.svm import SVC, LinearSVC

#lin_scv=LinearSVC(C=10 ,max_iter=1000, random_state=42, loss='hinge')
#model_train(lin_scv, X_train_scale, y_train)


# In[ ]:


#svc_rbf=SVC(kernel='rbf', C=100, gamma=0.03, random_state=42)
#model_train(svc_rbf, X_train, y_train)


# In[88]:


X_test = test.drop('Labels', axis=1, inplace=False)
X_test_scaled = scale.transform(X_test)
y_test = test['Labels'].copy()


# In[89]:


def validating(model, X, y):
    pred=model.predict(X)
    acc=accuracy_score(pred, y)
    recall=recall_score(pred, y)
    prec=precision_score(pred, y)
    f1=f1_score(pred, y)
    roc=roc_auc_score(pred, y)
    
    row=result_tab.shape[0]
    result_tab.loc[row]=['Test data: %s' %model, acc, recall, prec, f1, roc]
    
    return result_tab


# In[ ]:





# In[90]:


validating(sgd, X_test_scaled, y_test)


# In[91]:


validating(log_reg_default, X_test_scaled, y_test)


# In[92]:


validating(log_reg, X_test_scaled, y_test)


# In[93]:


validating(knn, X_test_scaled, y_test)


# In[3]:


df_test=pd.read_csv('C://Users/Al-Mahdi/Desktop/MLHEP/higgs-boson/test.csv')
#df_test['Labels']=label.fit(df_test['Label'])
#df_test.drop('Label', axis=1, inplace=True)


# In[41]:


device = torch.cuda.is_available()
device


# In[42]:


device = print('cuda' if torch.cuda.is_available() else 'cpu')


# In[43]:


from torch import nn


# In[44]:


y_train.shape


# In[46]:


len(X_train_scale)
X_train_scale.shape


# In[47]:


model = nn.Sequential(
        nn.Linear(28, 10),
        nn.ReLU(),
        nn.Linear(10, 2)
).to(device)


# In[48]:


for w in model.parameters():
    print(" ", w.shape)


# In[110]:


#X_train = np.asarray(X_train)
X = torch.tensor(X_train_scale, dtype=torch.float32)


# In[111]:


#y_train = np.asarray(y_train)
y = torch.tensor(y_train, dtype=torch.long)


# In[112]:


X_test.shape


# In[113]:


X_val = torch.tensor(X_test_scaled, dtype=torch.float32)
y_val = torch.tensor(y_test.values, dtype=torch.long)


# In[114]:


X_val.shape


# In[115]:


#trainloader = torch.utils.data.DataLoader(X, batch_size=2, 
                                         #shuffle=True)


# In[116]:


#X_batch, y_batch = iter(trainloader).next()
#print("batch size:", len(X_batch), "batch dimensions:", X_batch.shape)


# In[117]:


prediction = model(X_val.to(device))
print(prediction.shape)


# In[118]:


loss_fn = nn.CrossEntropyLoss()
loss_fn(prediction, y_val.to(device))


# In[119]:


num_epochs = 10
batch_size = 512

# some quantities to plot
train_losses = []
test_losses = []
test_accuracy = []

# "epoch" = one pass through the dataset
for i_epoch in range(num_epochs):
    shuffle_ids = np.random.permutation(len(X)) # shuffle the data
    for idx in trange(0, len(X), batch_size):
        # get the next chunk (batch) of data:
        batch_X = X[shuffle_ids][idx : idx + batch_size].to(device)
        batch_y = y[shuffle_ids][idx : idx + batch_size].to(device)

        # all the black magic:
        loss = loss_fn(model(batch_X), batch_y)
        loss.backward()
        opt = torch.optim.Adam(params=model.parameters(), lr=1e-3)
        opt.step()
        opt.zero_grad()

        # remember the loss value at this step
        train_losses.append(loss.item())

    # evaluate test loss and metrics
    test_prediction = model(X_val.to(device))
    test_losses.append(
        loss_fn(test_prediction, y_val.to(device)).item()
    )
    test_accuracy.append(
        (test_prediction.argmax(axis=1) == y_val.to(device)).to(float).mean()
    )

    # all the rest is simply plotting

    clear_output(wait=True)
    plt.figure(figsize=(8, 3), dpi=100)
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='train')
    plt.plot(
        np.linspace(0, len(train_losses), len(test_losses) + 1)[1:],
        test_losses, label='test'
    )
    plt.ylabel("Loss")
    plt.xlabel("# steps")
    plt.legend();

    plt.subplot(1, 2, 2)
    plt.plot(test_accuracy)
    plt.ylabel("Test accuracy")
    plt.xlabel("# epochs");
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




