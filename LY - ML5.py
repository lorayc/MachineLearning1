#!/usr/bin/env python
# coding: utf-8

# **Lora Yovcheva**

# # 0.) Import the Credit Card Fraud Data From CCLE

# In[46]:


import pandas as pd
#from google.colab import drive
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import seaborn as sns


# In[4]:


df = pd.read_csv("fraudTest.csv")


# In[5]:


df.head()


# In[6]:


#df_select = df[["trans_date_trans_time", "category", "amt", "city_pop", "is_fraud"]]

#df_select["trans_date_trans_time"] = pd.to_datetime(df_select["trans_date_trans_time"])
#df_select["time_var"] = [i.second for i in df_select["trans_date_trans_time"]]

#X = pd.get_dummies(df_select, ["category"]).drop(["trans_date_trans_time", "is_fraud"], axis = 1)
#y = df["is_fraud"]


# In[7]:


df_select = df[["trans_date_trans_time", "category", "amt", "city_pop", "is_fraud"]].copy()

df_select["trans_date_trans_time"] = pd.to_datetime(df_select["trans_date_trans_time"])
df_select["time_var"] = [i.second for i in df_select["trans_date_trans_time"]]

X = pd.get_dummies(df_select, ["category"]).drop(["trans_date_trans_time", "is_fraud"], axis = 1)
y = df["is_fraud"]


# # 1.) Use scikit learn preprocessing to split the data into 70/30 in out of sample

# In[8]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3)


# In[10]:


X_test, X_holdout, y_test, y_holdout = train_test_split(X_test, y_test, test_size = .5)


# In[11]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_holdout = scaler.transform(X_holdout)


# # 2.) Make three sets of training data (Oversample, Undersample and SMOTE)

# In[12]:


from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE


# In[13]:


ros = RandomOverSampler()
over_X, over_y = ros.fit_resample(X_train, y_train)

rus = RandomUnderSampler()
under_X, under_y = rus.fit_resample(X_train, y_train)

smote = SMOTE()
smote_X, smote_y = smote.fit_resample(X_train, y_train)


# # 3.) Train three logistic regression models

# In[14]:


from sklearn.linear_model import LogisticRegression


# In[15]:


over_log = LogisticRegression().fit(over_X, over_y)

under_log = LogisticRegression().fit(under_X, under_y)

smote_log = LogisticRegression().fit(smote_X, smote_y)


# # 4.) Test the three models

# In[16]:


over_log.score(X_test, y_test)


# In[17]:


under_log.score(X_test, y_test)


# In[18]:


smote_log.score(X_test, y_test)


# We see SMOTE performing with higher accuracy but is ACCURACY really the best measure? Accuracy score is not that important. We need to see if undersampling or oversampling is a better approach for the model. 

# # 5.) Which performed best in Out of Sample metrics?

# In[20]:


# Sensitivity here in credit fraud is more important as seen from last class


# In[21]:


from sklearn.metrics import confusion_matrix


# In[22]:


y_true = y_test


# In[23]:


y_pred = over_log.predict(X_test)
cm = confusion_matrix(y_true, y_pred)
cm


# In[24]:


print("Over Sample Sensitivity : ", cm[1,1] /( cm[1,0] + cm[1,1]))


# In[25]:


y_pred = under_log.predict(X_test)
cm = confusion_matrix(y_true, y_pred)
cm


# In[26]:


print("Under Sample Sensitivity : ", cm[1,1] /( cm[1,0] + cm[1,1]))


# In[27]:


y_pred = smote_log.predict(X_test)
cm = confusion_matrix(y_true, y_pred)
cm


# In[36]:


print("SMOTE Sample Sensitivity : ", cm[1,1] /( cm[1,0] + cm[1,1]))


# # 6.) Pick two features and plot the two classes before and after SMOTE.

# In[42]:


# Get the column names after get_dummies
column_names = pd.get_dummies(df_select, columns=["category"]).drop(["trans_date_trans_time", "is_fraud"], axis = 1).columns.tolist()

# Convert numpy arrays to pandas DataFrames
X_train_df = pd.DataFrame(X_train, columns=column_names)
y_train_df = pd.DataFrame(y_train, columns=['is_fraud'])

# Concatenate the DataFrames
raw_temp = pd.concat([X_train_df, y_train_df], axis=1)

#raw_temp = pd.concat([X_train, y_train], axis =1)


# In[48]:


plt.figure(figsize=(8, 6))

# Scatter plot for non-fraudulent transactions (class 0)
sns.scatterplot(x=raw_temp[raw_temp["is_fraud"] == 0]["amt"],
                y=raw_temp[raw_temp["is_fraud"] == 0]["city_pop"],
                label="Non-Fraudulent",
                color="lightblue")

# Scatter plot for fraudulent transactions (class 1)
sns.scatterplot(x=raw_temp[raw_temp["is_fraud"] == 1]["amt"],
                y=raw_temp[raw_temp["is_fraud"] == 1]["city_pop"],
                label="Fraudulent",
                color="orange")

plt.xlabel("Amount")
plt.ylabel("Population")
plt.title("Scatter Plot: Amount vs. City Population")
plt.legend()
plt.show()


# In[49]:


# Convert numpy arrays to pandas DataFrames
smote_X_df = pd.DataFrame(smote_X)
smote_y_df = pd.DataFrame(smote_y)

# Concatenate the DataFrames
raw_temp = pd.concat([smote_X_df, smote_y_df], axis=1)

#raw_temp = pd.concat([smote_X, smote_y], axis =1)


# In[50]:


plt.scatter(df[df["is_fraud"] == 1]["amt"], df[df["is_fraud"] == 1]["city_pop"])


# # 7.) We want to compare oversampling, Undersampling and SMOTE across our 3 models (Logistic Regression, Logistic Regression Lasso and Decision Trees).
# 
# # Make a dataframe that has a dual index and 9 Rows.
# # Calculate: Sensitivity, Specificity, Precision, Recall and F1 score. for out of sample data.

# In[51]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd


# In[52]:


resampling_methods = {
    'over': RandomOverSampler(), 
    "under": RandomUnderSampler(),
    "smote": SMOTE()
}


# In[53]:


model_configs = {
    "Log": LogisticRegression(),
    "Lasso": LogisticRegression(penalty = "l1",
                                 C = 2., solver = "liblinear"),
    "Dtree": DecisionTreeClassifier()
}


# In[54]:


trained_models = {}


# In[55]:


for resample_key, resampler in resampling_methods.items():
    resample_X, resample_y = resampler.fit_resample(X_train, y_train) 
    
    for model_key, model in model_configs.items():
        combined_key = f"{resample_key}_{model_key}"
        trained_models[combined_key] = model.fit(resample_X, resample_y)
        


# In[56]:


#trained_models["key1"] = "value1"
#trained_models


# In[57]:


def calc_perf_metric(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    F1 = f1_score(y_true, y_pred)
    
    return(sensitivity, specificity, precision, recall, F1)


# In[58]:


trained_models = {}
results = []


# In[59]:


for resample_key, resampler in resampling_methods.items():
    resample_X, resample_y = resampler.fit_resample(X_train, y_train)
    
    for model_key, model in model_configs.items():
        combined_key = f"{resample_key}_{model_key}"
        
        m = model.fit(resample_X, resample_y)
        
        trained_models[combined_key] = m 
        
        y_pred = m.predict(X_test)
        
        sensitivity, specificity, precision, recall, F1 = calc_perf_metric(y_test, y_pred)
        
        results.append({"Model": combined_key, 
                       "Sensitivity": sensitivity, 
                       "Specificity": specificity,
                       "Precision": precision, 
                       "Recall": recall,
                       "F1": F1})
        
        #####
        #results.append(calc_perf_metric(y_test, y_pred))
        


# In[60]:


results_df = pd.DataFrame(results)
results_df


# # Notice any patterns across perfomance for this model. Does one totally out perform the others IE. over/under/smote or does a model perform better DT, Lasso, LR?
# # Choose what you think is the best model and why. Test on Holdout

# In[61]:


# Predict on the holdout set
y_pred = model.predict(X_test)

# Calculate metrics
print(classification_report(y_test, y_pred))


# According to the results for precision, the model performs really well in correctly predicting the 0 class, but has a high rate of falsely ptrdicting positive. Based on the values of recall, we can see that the model misses some of the actual positive class instances, but correctly identifiest most of the negative class. The F1-score is the weighted average of precision and recall, which means that it takes into consideration the value of both of those metrics. For the negative class, which is non-fradulent transactions it preforms great, however for the positive class of fradulent transactions it performs much poorer and needs improvement. It is important to note that there is an imbalance between the classes (305 instances of class 1 compared to 83,053 instances of class 0) and judging performance based on a single metric might not be optimal. In addition, the macro avg takes into consideration both values equally and performs slightly lower at 0.7p. The weighted avg accounts for class imbalance by weighting each class based on its instances. Overall, choosing a model is a comprehensive desicion that requires considering the tradeoff between accuracy and interpretability. 
