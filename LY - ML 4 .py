#!/usr/bin/env python
# coding: utf-8

# # HR ATTRIBUTION

# In[30]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree  
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score


# # 1.) Import, split data into X/y, plot y data as bar charts, turn X categorical variables binary and tts.

# In[31]:


df = pd.read_csv("HR_Analytics.csv")
df.head()


# In[32]:


y = df[["Attrition"]].copy()
X = df.drop("Attrition", axis = 1)


# In[33]:


y["Attrition"] = [1 if i == "Yes" else 0 for i in y["Attrition"]]


# In[34]:


#
print("Unique values in df['Attrition']: ", df['Attrition'].unique())


# In[35]:


class_counts = y.value_counts()


plt.figure(figsize=(8, 6))
class_counts.plot(kind='bar', color='skyblue')
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.xticks(rotation=0)  # Remove rotation of x-axis labels
plt.show()


# In the last year we kept 1200 employees and 200 left. 

# In[36]:


# Step 1: Identify string columns
string_columns = X.columns[X.dtypes == 'object']

# Step 2: Convert string columns to categorical
for col in string_columns:
    X[col] = pd.Categorical(X[col])

# Step 3: Create dummy columns
X = pd.get_dummies(X, columns=string_columns, prefix=string_columns,drop_first=True)


# In[37]:


x_train,x_test,y_train,y_test=train_test_split(X,
 y, test_size=0.20, random_state=42)


# # 2.) Using the default Decision Tree. What is the IN/Out of Sample accuracy?

# In[38]:


clf = DecisionTreeClassifier()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_train)
acc=accuracy_score(y_train,y_pred)
print("IN SAMPLE ACCURACY : " , round(acc,2))

y_pred=clf.predict(x_test)
acc=accuracy_score(y_test,y_pred)
print("OUT OF SAMPLE ACCURACY : " , round(acc,2))


# To increse the out of sample accuracy: reduce the inputs using your intuition to reduce model complexity. The model is a bit overfit -> reduce in sample accuracy(100%), out of sample accuracy will get higher. Bias - Variance tradeoff. 

# # 3.) Run a grid search cross validation using F1 score to find the best metrics. What is the In and Out of Sample now?

# In[39]:


# Define the hyperparameter grid to search through
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': np.arange(1, 11),  # Range of max_depth values to try
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}


dt_classifier = DecisionTreeClassifier(random_state=42)

scoring = make_scorer(f1_score, average='weighted')

grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, scoring=scoring, cv=5)

grid_search.fit(x_train, y_train)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best F1-Score:", best_score)


# Fit those parameters in the model ,new model is clf:

# In[40]:


clf = tree.DecisionTreeClassifier(**best_params, random_state =42)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_train)
acc=accuracy_score(y_train,y_pred)
print("IN SAMPLE ACCURACY : " , round(acc,2))

y_pred=clf.predict(x_test)
acc=accuracy_score(y_test,y_pred)
print("OUT OF SAMPLE ACCURACY : " , round(acc,2))


# # 4.) Plot the confusion matrix, sort the most important features for the decision tree and plot the decision tree. 

# In[41]:


# Make predictions on the test data
y_pred = clf.predict(x_test)
y_prob = clf.predict_proba(x_test)[:, 1]

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(conf_matrix))
plt.xticks(tick_marks, ['Class 0', 'Class 1'], rotation=45)
plt.yticks(tick_marks, ['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

feature_importance = clf.feature_importances_

# Sort features by importance and select the top 10
top_n = 10
top_feature_indices = np.argsort(feature_importance)[::-1][:top_n]
top_feature_names = X.columns[top_feature_indices]
top_feature_importance = feature_importance[top_feature_indices]

# Plot the top 10 most important features
plt.figure(figsize=(10, 6))
plt.bar(top_feature_names, top_feature_importance)
plt.xlabel('Feature')
plt.ylabel('Importance Score')
plt.title('Top 10 Most Important Features - Decision Tree')
plt.xticks(rotation=45)
plt.show()

# Plot the Decision Tree for better visualization of the selected features
plt.figure(figsize=(17, 6))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=["Yes", "No"], rounded=True, fontsize=5)
plt.title('Decision Tree Classifier')
plt.show()


# The more blue: lower Ginis

# # 5.) Looking at the graphs. what would be your suggestions to try to improve customer retention? What additional information would you need for a better plan. Calculate anything you think would assist in your assessment.

# ## ANSWER :

# Feautre importance: pay them more. 
# 
# Run a regression on one feature.

# In[42]:


np.corrcoef(np.array(X["OverTime_Yes"]), [1 if i == "Yes" else 0 for i in y["Attrition"]])


# In[43]:


# Calculating the correlation between the features and the target variable
def calculate_correlation(X, feature_name, y):
    feature = X[feature_name]
    
    coef, _ = np.polyfit(feature, y, 1)
    coef = coef[0]
    return coef * 100


# Calculate the correlation between the top 10 features and the target variable
correlations = {}
for feature in top_feature_names:
    coef = calculate_correlation(X, feature, y)
    correlations[feature] = coef

# Print the correlation matrix
correlation_matrix = X.corr()  # Fix: Replace df with X
print(correlation_matrix)


# In[44]:


# Plot the correlation between the top 10 features and the target variable
plt.figure(figsize=(10, 6))
plt.bar(correlations.keys(), correlations.values())
plt.xlabel('Feature')
plt.ylabel('Correlation (%)')
plt.title('Correlation between Top 10 Features and Attrition')
plt.xticks(rotation=45)
plt.show()


# # 6.) Using the Training Data, if they made everyone stop overtime work. What would have been the expected difference in client retention?

# In[45]:


x_train_experiment = x_train.copy()


# In[46]:


x_train_experiment["OverTime_Yes"] = 0.


# In[47]:


y_pred_experiment = clf.predict(x_train_experiment)
y_pred= clf.predict(x_train)


# In[48]:


# All the people we would have saved from leaving
sum(y_pred - y_pred_experiment)
print("Stopping overtime work would have prevented people from leaving:", sum(y_pred - y_pred_experiment)) # finish this 


# # 7.) If they company loses an employee, there is a cost to train a new employee for a role ~2.8 * their monthly income.
# # To make someone not work overtime costs the company 2K per person.
# # Is it profitable for the company to remove overtime? If so/not by how much? 
# # What do you suggest to maximize company profits?

# In[49]:


x_train_experiment["Y"] = y_pred 
x_train_experiment["Y_exp"] = y_pred_experiment
x_train_experiment["Ret_Change"] = x_train_experiment["Y"] - x_train_experiment["Y_exp"]


# In[50]:


# Saving - Change in Training Cost 
sav = sum(x_train_experiment["Ret_Change"] * 2.8 * x_train_experiment["MonthlyIncome"])


# In[51]:


# Cost of lost overtime: 
cost = 2000 * len(x_train[x_train["OverTime_Yes"] == .1])


# In[52]:


print("Profit from this experiment:", sav-cost)


# ## ANSWER : 

# # 8.) Use your model and get the expected change in retention for raising and lowering peoples income. Plot the outcome of the experiment. Comment on the outcome of the experiment and your suggestions to maximize profit.

# Don't care about the attrition, but no company wil do that. 

# In[53]:


raise_amount1 = 500
x_train_experiment = x_train.copy()
x_train_experiment['MonthlyIncome'] = x_train_experiment['MonthlyIncome'] + raise_amount
y_pred_experiment = clf.predict(x_train_experiment)
y_pred = clf.predict(x_train) 
x_train_experiment["Y"] = y_pred 
x_train_experiment["Y_exp"] = y_pred_experiment
x_train_experiment["Ret_Change"] = x_train_experiment["Y"] - x_train_experiment["Y_exp"]
# Saving: Change in Trading Cost
sav = sum(x_train_experiment["Ret_Change"] * 2.8 * x_train_experiment["MonthlyIncome"])
# Cost of lost OverTime
cost = raise_amount * len(x_train)

print("Profit is:", sav-cost)
profits.append(sav-cost)


# In[54]:


profits = []
for raise_amount in range (-1000,1000, 100):
    x_train_experiment = x_train.copy()
    x_train_experiment['MonthlyIncome'] = x_train_experiment['MonthlyIncome'] + raise_amount
    y_pred_experiment = clf.predict(x_train_experiment)
    y_pred = clf.predict(x_train) 
    x_train_experiment["Y"] = y_pred 
    x_train_experiment["Y_exp"] = y_pred_experiment
    x_train_experiment["Ret_Change"] = x_train_experiment["Y"] - x_train_experiment["Y_exp"]
    # Saving: Change in Trading Cost
    sav = sum(x_train_experiment["Ret_Change"] * 2.8 * x_train_experiment["MonthlyIncome"])
    # Cost of lost OverTime
    cost = raise_amount * len(x_train)

    print("Profit is:", sav-cost)
    profits.append(sav-cost)

plt.plot(range(-1000,1000, 100),profits)
plt.xlabel('Raise Amount')
plt.ylabel('Profit')
plt.title('Profit vs Raise Amount')
plt.show()


# ## ANSWER : 

# With the first raise_amount = $ 500 we still loose money, however we saved 22 people from leaving by giving them 500 dollars. When we run the loop for -1000, 1000 and 100, the result is a downward sloping curve, which represent the tradeoff between profit for the company and rais amount. Maximizing profit would require lower salaries of the employees, which in turn will lower the retention rate of the company. On the other hand, giving raises and retaining employees will lower the profits of the company. Therefore, the decision for finding the optimum point of balance depends on the preferences of the company, their goals and values as we saw in the Amazon example. 
