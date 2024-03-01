#!/usr/bin/env python
# coding: utf-8

# # 0.) Import and Clean data

# In[39]:


#!pip install google.colab
# ! pip install imblearn
import pandas as pd
#from google.colab import drive
import matplotlib.pyplot as plt
import numpy as np


# In[40]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
import seaborn as sns
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")


# In[41]:


#drive.mount('/content/gdrive/', force_remount = True)


# In[42]:


df = pd.read_csv("bank-additional-full (1).csv", sep =";") # read csv separated by a ; because of how the data was uploaded


# In[43]:


df.head()


# In[44]:


df = df.drop(["default", "pdays",	"previous",	"poutcome",	"emp.var.rate",	"cons.price.idx",	"cons.conf.idx",	"euribor3m",	"nr.employed"], axis = 1)
df = pd.get_dummies(df, columns = ["loan", "job","marital","housing","contact","day_of_week", "campaign", "month", "education"],drop_first = True)


# In[45]:


df.head()


# In[46]:


y = pd.get_dummies(df["y"], drop_first = True)
X = df.drop(["y"], axis = 1)


# In[47]:


obs = len(y)
plt.bar(["No","Yes"],[len(y[y.yes==0])/obs,len(y[y.yes==1])/obs])
plt.ylabel("Percentage of Data")
plt.show()


# In[48]:


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler().fit(X_train)

X_scaled = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# #1.) Based on the visualization above, use your expert opinion to transform the data based on what we learned this quarter

# In[49]:


###############
###TRANSFORM###
###############
#X_scaled = #???
#y_train = #???
smote = SMOTE()
X_scaled_resampled, y_train_resampled = smote.fit_resample(X_scaled, y_train)


# # 2.) Build and visualize a decision tree of Max Depth 3. Show the confusion matrix.

# In[50]:


dtree_main = DecisionTreeClassifier(max_depth = 3)
dtree_main.fit(X_scaled, y_train)


# In[51]:


fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
plot_tree(dtree_main, filled = True, feature_names = X.columns, class_names=["No","Yes"])


#fig.savefig('imagename.png')


# # 1b.) Confusion matrix on out of sample data. Visualize and store as variable

# In[52]:


y_pred = dtree_main.predict(X_test)
y_true = y_test
cm_raw = confusion_matrix(y_true, y_pred)


# In[53]:


class_labels = ['Negative', 'Positive']

# Plot the confusion matrix as a heatmap
sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# # 3.) Use bagging on your descision tree

# In[54]:


#placegolder for optimizing max_depth
dtree = DecisionTreeClassifier(max_depth = 3)


# In[55]:


bagging = BaggingClassifier(estimator = dtree, 
                            n_estimators = 100, 
                            max_samples = .5, 
                            max_features = 1.)


# In[56]:


#go to where I resampled x_scaled before and afte resample 
bagging.fit(X_scaled, y_train)
y_pred = bagging.predict(X_test)


# In[57]:


y_pred = bagging.predict(X_test)
y_true = y_test
cm_raw = confusion_matrix(y_true, y_pred)


# In[58]:


class_labels = ['Negative', 'Positive']

# Plot the confusion matrix as a heatmap
sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[59]:


#Marginally improved based on what we are targeting 


# # 4.) Boost your tree

# In[60]:


from sklearn.ensemble import AdaBoostClassifier


# In[61]:


#placegolder for optimizing max_depth
dtree = DecisionTreeClassifier(max_depth = 3)

boost = AdaBoostClassifier(estimator = dtree, 
                           n_estimators = 100, 
                          learning_rate = .1)


# In[62]:


boost.fit(X_scaled, y_train)

y_pred = boost.predict(X_test)


# In[63]:


y_true = y_test
cm_raw = confusion_matrix(y_true, y_pred)


# In[64]:


class_labels = ['Negative', 'Positive']

# Plot the confusion matrix as a heatmap
sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[65]:


# Did a little bit better
#those are two types of ensemble models, 5 a bit different


# # 5.) Create a superlearner with at least 4 base learner models. Use a logistic reg for your metalearner. Interpret your coefficients and save your CM.

# In[ ]:


pip install mlens


# In[67]:


from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier

#from mlens.ensemble import SuperLearner


# In[68]:


# Stacking - decision tree, boodted tree, bags tree -> train Logistic regression


# In[69]:


# 3 classifiers, 3 trained models
base_predictions = [list(dtree_main.predict(X_train)), 
                    list(boost.predict(X_train)), 
                    list(bagging.predict(X_train))]


# In[70]:


np.array(base_predictions)


# In[71]:


base_predictions_transpose = np.array(base_predictions).transpose()
#we have to restructure base predicitions before running the LogisticRegression
#sklearn import pipe? piping? allows to do this easier  # first value of every one of the 3 lists


# ### Interpret Coefficients

# In[72]:


super_learner = LogisticRegression()
super_learner.fit(base_predictions_transpose, y_train)


# In[73]:


super_learner.coef_


# ***Interpret the coefficients of the super learner*** - The coefficients are the weights of the models in the superlearner (n ensemble machine learning algorithm that combines multiple machine learning algorithms into a single model). The coefficients you provided are the weights assigned by the superlearner to each of the base models' predictions when making a final prediction. The sign of the weight indicates the direction of the relationship, and the closer the weight is to 0, the less the model is trusted/the further the weight is from 0, the more the model is trusted. Furthermore, the sign of the weight determines the dirrection of the relationship - positive weight indicates a positive relationship, while negative weight indicates a negative relationship. 
# 
# - The first model [1.17798336]: The positive coefficient means that the superlearner puts a high value to the model's prediction. The decision tree and the de super learner are more likely to predict the same class. 
# 
# - The second model [-0.97371215]: The negative coefficient means that the superlearner puts an inverse value to the model's prediction. Model might be more likely to predict one class at the expence of the other. The boosting model and the superlearner are more likely to predict the opposite class from each other. 
# 
# - The third model [-0.97124981]: The negative coefficient means that the superlearner puts an inverse value to the model's prediction. Model might be more likely to predict one class at the expence of the other. The bagging model and the superlearner are more likely to predict the opposite class from each other. 
# 

# # 6.) Conclusions
# ***Confusion Matrices for each model*** - show the TP, TN, FP and FN predicted by each model. 
# 
# - *Decision Tree*: tend to have more false positives and false negatives because they can overfit to the training data, especially if the tree depth is not controlled. This can lead to a lower accuracy, precision, recall, and F1 score.
# 
# - *Bagging*: usually perform better than a single Decision Tree because they reduce variance by averaging multiple decision trees trained on different subsets of the data. This typically results in fewer false positives and false negatives, leading to higher accuracy, precision, recall, and F1 score.
# 
# - *Boosting*: AdaBoost create a sequence of models that attempt to correct the mistakes of the previous models. This can lead to even fewer false positives and false negatives than Bagging, resulting in even higher accuracy, precision, recall, and F1 score.
# 
# We can see that the first model has the lowest errors in predicting the positive and the negative class, which is consistent to the coefficient value results from the superlearner. 
