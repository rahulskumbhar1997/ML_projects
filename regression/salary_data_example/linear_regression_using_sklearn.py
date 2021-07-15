#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ### Read the dataset

# In[3]:


df = pd.read_csv('dataset/Salary_Data.csv')


# ### Get some insights

# In[4]:


print(df.head())


# In[5]:


print(df.info())


# ### Plot the data to get some insights

# In[6]:


plt.scatter(df["YearsExperience"], df["Salary"])
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()


# ### Prepare training and testing dataset

# In[7]:


from sklearn.model_selection import train_test_split

x = df.drop("Salary", axis=1)
y = df["Salary"]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)


# ### Create a model

# In[8]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()


# ### Train the model

# In[9]:


model.fit(x_train, y_train)


# ### Predict the result for test dataset

# In[10]:


y_prediction = model.predict(x_test)


# ### Check the accuracy of model

# In[12]:


from sklearn.metrics import mean_squared_error, mean_absolute_error

mae = mean_absolute_error(y_test, y_prediction)
mse = mean_squared_error(y_test, y_prediction)

print("Mean Absolute error : ",mae)
print("Mean squared error : ", mse)
print("Root mean squared error : ", np.sqrt(mse))


# ### Plot the predicted result

# In[20]:


plt.scatter(x_test, y_test, label="Actual values")
plt.scatter(x_test, y_prediction, c='red', label="Predicted values")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()


# In[29]:


# Intercept value
print("Intercept value : ",model.intercept_)

# Coefficient value 
print("Coefficient value : ", model.coef_)


# In[37]:


y_predict_eq = x * model.coef_ + model.intercept_

plt.scatter(x, y, label="Actual data")
plt.plot(x, y_predict_eq, c='red', label="Best fit line")
plt.legend()
plt.xlabel("Years of Experience")
plt.ylabel("Salary")


# In[ ]:




