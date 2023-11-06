# importing all necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score
warnings.filterwarnings("ignore")

# Loading the dataset
df = pd.read_csv('data_mobile_price_range.csv')

# Handling missing values: Replace sc_w and px_height with mean values
df['sc_w'].replace(0, df['sc_w'].mean(), inplace=True)
df['px_height'].replace(0, df['px_height'].mean(), inplace=True)

# Defining X and y
X = df.drop(['price_range'], axis=1)
y = df['price_range']

# Splitting dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Applying logistic regression
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Prediction
y_pred_test = lr.predict(X_test)

# Evaluation metrics for test
from sklearn.metrics import classification_report
print('Classification report for Logistic Regression (Test set)= ')
print(classification_report(y_pred_test, y_test))


# Random Forest
clsr = RandomForestClassifier(n_estimators=300)
clsr.fit(X_train, y_train)
y_pred_test = clsr.predict(X_test)

print('Classification report for RandomForestClassifier (Test set):')
print(classification_report(y_test, y_pred_test))

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [10, 50, 200], # so luong cay quyet dinh
    'max_depth': [10, 20, 40], #chieu sau toi da cua cay quyet dinh
    'min_samples_split': [2, 4, 6], #so luong mau de chia nut 
    'max_features': ['sqrt', 'auto'], # so luong features 
    'max_leaf_nodes': [10, 20, 40] # so luong nut la max
}
rf = RandomForestClassifier()
grid = GridSearchCV(rf, param_grid, scoring='accuracy', cv=3)
grid.fit(X, y)

# Print best parameters and best score for Random Forest
print("Best parameters for Random Forest:", grid.best_params_)
print()
# Hyperparameter tuned Random Forest
clsr = grid.best_estimator_
clsr.fit(X_train, y_train)

# Prediction and evaluation for Random Forest
y_pred_test = clsr.predict(X_test)
y_pred_train = clsr.predict(X_train)

print('Classification report for tuned Random Forest (Test set):')
print(classification_report(y_test, y_pred_test))

# Decision Tree
dtc = DecisionTreeClassifier(max_depth=5)
dtc.fit(X_train, y_train)
y_pred_test = dtc.predict(X_test)
print('Classification report for DecisionTreeClassifier (Test set):')
print(classification_report(y_test, y_pred_test))

# Hyperparameter tuning for Decision Tree
param_grid = {'max_depth': [5, 30], 'max_leaf_nodes': [10, 100]}
grid = GridSearchCV(dtc, param_grid, scoring='accuracy', cv=3)
grid.fit(X_train, y_train)

# Print best parameters for Decision Tree
print("Best parameters for Decision Tree:", grid.best_params_)
print()
# Hyperparameter tuned Decision Tree
dtc = grid.best_estimator_
dtc.fit(X_train, y_train)

# Prediction and evaluation for Decision Tree
y_pred_test = dtc.predict(X_test)

print('Classification report for tuned Decision Tree (Test set):')
print(classification_report(y_test, y_pred_test))

# XGBoost
xgb = XGBClassifier(max_depth=5, learning_rate=0.1)
xgb.fit(X_train, y_train)

# Prediction and evaluation for XGBoost

y_pred_test = xgb.predict(X_test)

print('Classification report for XGBoost (Test set):')
print(classification_report(y_test, y_pred_test))
# Hyperparameter tuning for XGBoost
param_grid = {
    'n_estimators': [10, 200], # so luong cay 
    'learning_rate': [1, 0.01, 0.001], # toc do hoc 
    'max_depth': [5, 10], #chieu sau toi da cua cay 
    'gamma': [1.5, 1.8], #kiem soat pruning 
    'subsample': [0.3, 0.5, 0.8] # ti le mau su dung trong vong lap 
}
grid = GridSearchCV(xgb, param_grid, cv=3, scoring='accuracy')
grid.fit(X_train, y_train)

# Hyperparameter tuned XGBoost
xgb = grid.best_estimator_
xgb.fit(X_train, y_train)

# Print best parameters for tuned XGBoost
print("Best parameters for tuned XGBoost:", grid.best_params_)
print()

# Prediction and evaluation for tuned XGBoost
y_pred_test = xgb.predict(X_test)

# import seaborn as sns
# from sklearn.metrics import confusion_matrix

# #Generate the confusion matrix
# cf_matrix = confusion_matrix(y_test, y_pred_test)

# print(cf_matrix)

# ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

# ax.set_title('Seaborn Confusion Matrix with labels\n\n');
# ax.set_xlabel('\nPredicted Values')
# ax.set_ylabel('Actual Values ');

# ## Ticket labels - List must be in alphabetical order
# ax.xaxis.set_ticklabels([0,1,2,3])
# ax.yaxis.set_ticklabels([0,1,2,3])

# ## Display the visualization of the Confusion Matrix.
# plt.show()
     
print('Classification report for tuned XGBoost (Test set):')
print(classification_report(y_test, y_pred_test))

