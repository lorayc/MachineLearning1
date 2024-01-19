#!/usr/bin/env python
# coding: utf-8

# # 1.) Pull in Data and Convert ot Monthly

# In[16]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[17]:


apple_data = yf.download('AAPL')
df = apple_data.resample("M").last()[["Adj Close"]]


# # 2.) Create columns. 
#   - Current Stock Price, Difference in stock price, Whether it went up or down over the next month,  option premium

# In[18]:


df.head()


# In[19]:


# Difference in stock price
df["Diff"] = df["Adj Close"].diff().shift(-1)


# In[20]:


#Target
df["Target"] = np.sign(df["Diff"])


# In[21]:


df["Premium"] = .08 *df ["Adj Close"]


# # 3.) Pull in X data, normalize and build a LogReg on column 2

# In[22]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[23]:


X = pd.read_csv("Xdata.csv", index_col="Date", parse_dates=["Date"])


# In[24]:


y = df.loc[:"2023-09-30","Target"].copy()
df = df.loc[:"2023-09-30",:].copy()


# In[25]:


# fit a logistic regression
logreg = LogisticRegression()
logreg.fit(X,y)


# # 4.) Add columns, prediction and profits.

# In[26]:


y_pred = logreg.predict(X)


# In[27]:


df["Predictions"] = y_pred


# In[28]:


df["Profits"] = 0


# In[29]:


# True Positive
df.loc[(df["Target"] == 1) & (df["Predictions"] == 1), "Profits"] = df["Premium"]

# false Positive
df.loc[(df["Target"] == -1) & (df["Predictions"] == 1), "Profits"] = (100*df["Diff"] + df["Premium"])


# # 5.) Plot profits over time

# In[30]:


plt.plot (np.cumsum(df["Profits"]))
plt.grid()
plt.xlabel("Time")
plt.ylabel("Profits")
plt.title("Profits over time")


# # 5.5) How you see my skills valuable to PJ and/or Phillip Liu? 

# I am building a strong base of knowledge in Machine Learning, which is backed by my experience in Economics, Statisitics and Finance. This gives me a unique quality that I can incorporate data science into an economic or financial issue and have the flexibility to merge my knowledge and experience in those areas. That diversity would definately benefit the business model of PJ's company, as well as in the line of work of Phillip Liu with blockchain, inscriptions and cloud computing. 
