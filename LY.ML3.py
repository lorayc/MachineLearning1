#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
from pytrends.request import TrendReq
import sklearn
import matplotlib.pyplot as plt 


# In[9]:


y = pd.read_csv("AAPL_quarterly_financials.csv")


# # Clean the Apple Data to get a quarterly series of EPS.

# In[10]:


y.head()


# In[11]:


y.index = y.name


# In[12]:


y = pd.DataFrame(y.loc["BasicEPS", :]).iloc[2:,:]


# In[13]:


y.index =pd.to_datetime(y.index)


# In[14]:


#Assumption: nulls are 0, need to investigate
y = y.fillna(0.).sort_index()
y


# # Come up with 6 search terms you think could nowcast earnings. (Different than the ones I used) Add in 3 terms that that you think will not Nowcast earnings. Pull in the gtrends data. Clean it to have a quarterly average.

# In[49]:


# Create pytrends object
pytrends = TrendReq(hl='en-US', tz=360)

# Set up the keywords and the timeframe
keywords = ["iPhone", "Apple Layoffs", "MacBook", "iPad", "Apple CEO", "Apple Share Price", "Recession", "Chip Costs", "Apple OS", "Apple Watch", "Mac Mini", "iTunes", "iCloud", "Apple Store","Apple Vision", "Leo Messi", "String theory", "Mall"]
start_date = '2004-01-01'
end_date = '2024-01-01'

# Create an empty DataFrame to store the results
df = pd.DataFrame()

# Iterate through keywords and fetch data
for keyword in keywords:
    pytrends.build_payload([keyword], cat=0, timeframe=f'{start_date} {end_date}', geo='', gprop='')
    interest_over_time_df = pytrends.interest_over_time()
    df[keyword] = interest_over_time_df[keyword]


# In[50]:


X = df.resample("Q").mean()


# In[51]:


temp = pd.concat([y,X], axis = 1).dropna()
y = temp["BasicEPS"].copy()
X = temp.iloc[:,1:].copy()


# In[52]:


df


# # Normalize all the X data

# In[53]:


from sklearn.preprocessing import StandardScaler


# In[54]:


scaler = StandardScaler()


# In[55]:


X_scaled = scaler.fit_transform(X)


# # Run a Lasso with lambda that reduces less than half of your variables. Plot a bar chart.

# In[56]:


from sklearn.linear_model import Lasso


# In[57]:


lasso = Lasso(alpha = .1)


# In[58]:


lasso.fit(X_scaled, y)


# In[59]:


coefficients = lasso.coef_


# In[60]:


plt.figure(figsize = (20,8))
plt.bar(range(len(coefficients)), coefficients, tick_label = X.columns)
plt.axhline(0., color = "red")
plt.xticks(rotation=45, ha='right')
plt.grid()
plt.show()


# # Do these coefficient magnitudes make sense?

# Lambda is the tuning parameter, which controls the strength of the penalty and larger values of lambda make the models simpler by shrinking more coefficients to 0. 
# 
# I think that the coefficient magnitudes make sense. It is interesting to see the high peak for Apple Watch, wich was reduced the least and in the plot it peaks above the coefficients for iPhone, Apple Share Price and Apple Ceo. 
# 
# The amount of reduction of each coefficient depends on its association with the the target variable and its correlation with the other features.
# 
# I am interested in the cause for the term Leo Messi not to be completely reduced, as it does not relate to Apple. Whereas MacBook, iPad, iCloud and others, which are Apple products or services, are reduced to 0. Some topics to consider are: noise, interactions, correlations, scaling and regularization. 
