#!/usr/bin/env python
# coding: utf-8

# ## Dragon Real State Price Predictor

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv("data.csv")


# In[3]:


housing .head()


# In[4]:


housing.info()


# In[5]:


housing['CHAS'].value_counts()


# In[6]:


housing.describe()


# In[7]:


housing['AGE'].value_counts()


# In[8]:


#get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


'''import matplotlib.pyplot as plt
housing.hist(bins= 50,figsize=(20,15))'''


# ## train test spillting

# In[10]:


#for learning purpose
import numpy as np
def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[11]:


#train_set , test_set = split_train_test(housing, 0.2)


# In[12]:


#print(f"Rows in training set : {len(train_set)} \nRows in test set : {len(test_set)}\n")


# In[13]:


from sklearn.model_selection import train_test_split
train_set , test_set = train_test_split(housing, test_size = 0.2,random_state=42)
print(f"Rows in training set : {len(train_set)} \nRows in test set : {len(test_set)}\n")


# In[14]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits = 1 , test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(housing , housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[15]:


strat_test_set['CHAS'].value_counts()


# In[16]:


strat_train_set['CHAS'].value_counts()


# ## Looking for correlation

# In[17]:


from pandas.plotting import scatter_matrix
attributes = ["MEDV","RM","ZN","LSTAT"]
scatter_matrix(housing[attributes], figsize = (12,8))


# In[18]:


housing.plot(kind="scatter",x="RM",y="MEDV",alpha=0.8)


# ## Trying out Attributest combination

# In[19]:


housing["TaxRM"] = housing['TAX']/housing['RM']


# In[20]:


housing.head()


# In[21]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending = False)


# In[22]:


housing.plot(kind="scatter",x="TaxRM",y="MEDV",alpha=0.8)


# In[23]:


housing = strat_train_set.drop("MEDV",axis = 1)
housing_labels = strat_train_set["MEDV"].copy()


# ## Creating a Pipeline

# In[24]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler()),
])


# In[25]:


housing_num = my_pipeline.fit_transform(housing)


# In[26]:


housing_num.shape


# ## Selecting  a desired model for Dragon Real Estates

# In[27]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model = LinearRegression()
#model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num, housing_labels)


# In[28]:


some_data = housing.iloc[:5]


# In[29]:


some_labels = housing_labels.iloc[:5]


# In[30]:


prepared_data = my_pipeline.transform(some_data)


# In[31]:


model.predict(prepared_data)


# In[32]:


list(some_labels)


# In[33]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num)
mse = mean_squared_error(housing_labels,housing_predictions)
rmse = np.sqrt(mse)


# In[34]:


rmse


# ## Using better evaluation technique - Cross Validation

# In[35]:


from sklearn.model_selection import cross_val_score
score = cross_val_score(model, housing_num , housing_labels,scoring = "neg_mean_squared_error", cv = 10)
rmse_score = np.sqrt(-score)


# In[36]:


rmse_score


# In[37]:


def print_scores(score):
    print("scores",score)
    print("Mean : ", score.mean())
    print("Standard deviation : ",score.std())


# In[38]:


print_scores(rmse_score)


# ## Saving the model

# In[40]:


from joblib import dump, load
dump(model, 'Dragon.joblib') 


# ## Testing the model

# In[50]:


X_test = strat_test_set.drop("MEDV",axis=1)
Y_test = list(strat_test_set["MEDV"].copy())
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = list(model.predict(X_test_prepared))
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


# In[54]:


l = []
for i in range (len(final_predictions)) :
    for j in range(len(Y_test)):
        if i==j:
            l.append([final_predictions[i],Y_test[j]])
        else:
            continue
    else:
        continue
print(l)


# In[55]:


print(final_rmse)


# In[57]:


prepared_data[0]


# In[ ]:




