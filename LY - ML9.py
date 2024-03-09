#!/usr/bin/env python
# coding: utf-8

# # 0.) Import and Clean data

# In[1]:


import pandas as pd
# from google.colab import drive
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# In[2]:


#drive.mount('/content/gdrive/', force_remount = True)
df = pd.read_csv("Country-data.csv", sep = ",")
df.head(35)


# In[12]:


# Remove the string column to standarize the data
names = df['country'].copy
X = df.drop('country', axis = 1)


# In[13]:


scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)


# # 1.) Fit a kmeans Model with any Number of Clusters

# In[14]:


# KMeans


# In[15]:


kmeans = KMeans(n_clusters=5)
kmeans.fit(X)


# # 2.) Pick two features to visualize across

# In[16]:


X.columns


# In[25]:


import matplotlib.pyplot as plt

x1_index = 0
x2_index = 3


scatter = plt.scatter(X_scaled[:, x1_index], X_scaled[:, x2_index], c=kmeans.labels_, cmap='viridis', label='Clusters')


centers = plt.scatter(kmeans.cluster_centers_[:, x1_index], kmeans.cluster_centers_[:, x2_index], marker='o', color='black', s=100, label='Centers')

plt.xlabel(X.columns[x1_index])
plt.ylabel(X.columns[x2_index])
plt.title('Scatter Plot of Customers')

# Generate legend
plt.legend()

plt.grid()
plt.show()


# # 3.) Check a range of k-clusters and visualize to find the elbow. Test 30 different random starting places for the centroid means
# 

# In[26]:


WCSSs = []
Ks = range(1,15)
for k in Ks: 
    kmeans = KMeans(n_clusters=k, n_init = 30).fit(X_scaled)
    WCSSs.append(kmeans.inertia_)


# In[27]:


# Optional 
WCSSs = [KMeans(n_clusters=5, n_init=30).fit(X_scaled).inertia_ for k in range(1, 15)]


# # 4.) Use the above work and economic critical thinking to choose a number of clusters. Explain why you chose the number of clusters and fit a model accordingly.

# In[28]:


plt.plot(Ks, WCSSs, 'bx-')
plt.xlabel('k')
plt.ylabel('WCSS')
plt.title('Elbow Method For Optimal k')
plt.show()


# In[29]:


# No real elbow - later yes 


# # 6.) Do the same for a silhoutte plot

# In[30]:


from sklearn.metrics import silhouette_score


# In[31]:


SSs  = []
Ks = range(2,15)  # Update the range to start from 2
for k in Ks: 
    kmeans = KMeans(n_clusters=k, n_init = 30).fit(X_scaled)
    sil = silhouette_score(X_scaled, kmeans.labels_)
    SSs.append(sil)


# In[32]:


plt.plot(Ks, SSs, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different Numbers of Clusters')
plt.show()


# # 7.) Create a list of the countries that are in each cluster. Write interesting things you notice.

# In[33]:


kmeans = KMeans(n_clusters=2, n_init = 30).fit(X_scaled)


# In[34]:


preds = pd.DataFrame(kmeans.labels_)


# In[35]:


output = pd.concat([preds, df], axis = 1)
output.head(35)


# In[36]:


print("Cluster 1:")
list(output.loc[output[0] == 0, 'country'])


# In[37]:


print("Cluster 2:")
list(output.loc[output[0] == 1, 'country'])


# In[38]:


print("Cluster 0: ", output[output[0] == 0]["country"].values)
print("Cluster 1: ", output[output[0] == 1]["country"].values)


# In[39]:


#### Write an observation
# Cluster 0 has countries with lower income, lower gdpp, lower health, lower life expectancy, lower fertility and lower imports
# Cluster 1 has countries with higher income, higher gdpp, higher health, higher life expectancy, higher fertility and higher imports
# The clusters are based on the income, gdpp, health, life expectancy, fertility and imports of the countries
# The clusters are not based on the exports, inflation, child mortality and total fertility of the countries
# The clusters are not based on the population, literacy and net income of the countries


# ## Observations: 
# - Cluster 0 has countries with lower income, lower gdpp, lower health, lower life expectancy, lower fertility and lower imports
# - Cluster 1 has countries with higher income, higher gdpp, higher health, higher life expectancy, higher fertility and higher imports
# - The clusters are based on the income, gdpp, health, life expectancy, fertility and imports of the countries
# - The clusters are not based on the exports, inflation, child mortality, total fertility, population, literacy and net income of the countries
# 

# ## Explain why you choose the number of clusters 
# The elbow method is useful for determining the optimal number of clusters by plotting the within-cluster sum of squares (WCSS) against the number of clusters. The elbow point is the point at which the decreased in WCSS levels off. 
# 
# The chosen clusters in class were 2 for several reasons: 
# - Simplicity: simplified interpretation as a starting point in an exploratory analysis
# - Data Understanding: we were interested in the separation to the data in two general clusters of developed vs developing countries using the data that is available.
# 
# The two cluster provide us with insights even if we do not observe a clear elbow point. 

# # 8.) Create a table of Descriptive Statistics. Rows being the Cluster number and columns being all the features. Values being the mean of the centroid. Use the nonscaled X values for interprotation

# In[40]:


output.drop("country", axis = 1, inplace = True)


# In[41]:


#output = output.apply(pd.to_numeric, errors='coerce')
output.groupby(0).mean()
output.groupby(0).std()


# # 9.) Write an observation about the descriptive statistics.

# About the data: 
# - The mean and standard deviation of the clusters are different. 
# - The clusters are different from each other. 

# According to the descriptive statistics: 
# - Cluster 0: has lower values for child mortality, exports, health, imports, income, inflation, life expectancy, total fertility, and GDP per capita compared to Cluster 1.
# - Cluster 1: has higher values for child mortality, exports, health, imports, income, inflation, life expectancy, total fertility, and GDP per capita compared to Cluster 0.
# 
# The differences between the two clusters clearly show that there is a big distinction between the two clusters of countries as they present different socio-economic characteristics. This aligns with our initial goal to successfully classify and separate the countries in two groups as developed or developing. An explanation of the higher child mortality rate  and lower life expectancy in Cluster 1 could be due to poorer health services, lower overall health standards, lower living conditions or other factors that affect the health in the countries in that cluster, a problem associated with developing countries. Cluster 1 also has a higher total fertility that can be caused by cultural factors, low women participation in workforce ,and limited access to education and contraception, which is usually observed in developing countries. Additionally, income and GDP, exports and imports have a higher value in Cluster 0, indicating that these counties have a higher standard of living, economic prosperity and are more involved in international trade, a quality associated with developed countries. Lastly, the average rate of inflation is also lower for the countries in Cluster 1 than in Cluster 0, confirming the more stable economic environment. 
