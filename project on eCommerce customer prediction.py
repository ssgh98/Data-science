#!/usr/bin/env python
# coding: utf-8

# In[1]:


#packeges

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#  Get the Data
#
#  Avg. Session Length: Average session of in-store style advice sessions.
#  Time on App: Average time spent on App in minutes
#  Time on Website: Average time spent on Website in minutes
#  Length of Membership: How many years the customer has been a member. 

customers = pd.read_csv("EcommerceCustomers.csv")


print(customers.head())


# In[4]:


print(customers.describe())


# In[5]:


print(customers.info())


# In[6]:


# Compare Website vs App on impact to Yearly Spending
sns.set_palette("GnBu_d")
#sns.set_style('whitegrid')
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)
sns.plt.show()


# In[7]:


sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)
sns.plt.show()


# In[8]:


# ** Use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership.**

sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=customers)
sns.plt.show()


# In[9]:


# ** Explore relationships across the entire data set. 

sns.pairplot(customers)
sns.plt.show()


# In[10]:


# **Create a linear model plot of  Yearly Amount Spent vs. Length of Membership. **

sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)
sns.plt.show()


# In[11]:


#  Training and Testing Data

# Split the data into training and testing sets.

y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]


# In[15]:


# ** Use model_selection.train_test_split from sklearn to split the data into training and testing sets. Set test_size=0.3 and random_state=101**

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[16]:


#  Training the Model
 
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)


# In[17]:


#The coefficients

print('Coefficients: \n', lm.coef_)


# In[18]:


# ## Predicting Test Data

predictions = lm.predict( X_test)


# In[19]:


# Create a scatterplot of the real test values versus the predicted values. **

plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()


# In[20]:


# Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error.
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[21]:


#  Residuals
# 
# Explore the residuals to make sure everything was okay with our data. 
# 
# Plot a histogram of the residuals and make sure it looks normally distributed. 

sns.distplot((y_test-predictions),bins=50)


# In[ ]:




