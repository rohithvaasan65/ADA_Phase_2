#!/usr/bin/env python
# coding: utf-8

# ### **PRODUCT DEMAND PREDICTION**

# Problem Definition:

# The problem is to develop a machine learning model that can predict product
# demand based on historical sales data and external factors.
# 
# This model will help businesses optimize their inventory management and production planning to meet customer needs efficiently.
# 
# The project will involve data collection, data preprocessing, feature engineering, model selection, training, and evaluation.
# 

# IMPORTING LIBRARIES

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# DATA COLLECTION

# In[ ]:


file_path = '/content/product demand.csv'
data = pd.read_csv(file_path)


# DATA PREPROCESSING

# In[ ]:


data.fillna(0, inplace=True)


# In[ ]:


data.isnull().sum()


# FEATURE SELECTION

# In[ ]:


# Feature Selection
features = ['ID', 'Store ID', 'Total Price', 'Base Price']  # Features
target = 'Units Sold'  # Target variable


#  Histograms and Box Plots:

# In[ ]:


import matplotlib.pyplot as plt

# Histograms
data[features].hist(bins=20, figsize=(12, 10))
plt.suptitle("Histograms of Features")
plt.show()

# Box Plots
data[features].plot(kind='box', vert=False, figsize=(12, 6))
plt.title("Box Plots of Features")
plt.show()


# Correlation Matrix:

# In[ ]:


import seaborn as sns

correlation_matrix = data[features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# Pair Plot:

# In[ ]:


sns.pairplot(data[features])
plt.suptitle("Pair Plot of Features")
plt.show()


#  Target Variable Distribution:

# In[ ]:


plt.figure(figsize=(8, 6))
sns.histplot(data[target], bins=20, kde=True)
plt.title("Distribution of Target Variable")
plt.xlabel(target)
plt.ylabel("Frequency")
plt.show()


#  Feature vs. Target Plots:

# In[ ]:


for feature in features:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data[feature], y=data[target])
    plt.title(f"{feature} vs. {target}")
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.show()


#  Box Plot of Target Variable Grouped by Categorical Feature

# In[ ]:


categorical_feature = 'Store ID'  # Example categorical feature
plt.figure(figsize=(10, 6))
sns.boxplot(x=categorical_feature, y=target, data=data)
plt.title(f"Box Plot of {target} Grouped by {categorical_feature}")
plt.xlabel(categorical_feature)
plt.ylabel(target)
plt.xticks(rotation=45)
plt.show()


# ## ARIMA MODEL

# In[4]:


data


# In[16]:


pro = data['Units Sold']
autocorr_values = pro.autocorr()
print("Autocorrelation:", autocorr_values)


# In[18]:


plot_acf(data['Units Sold'])


# In[19]:


# Plot the PACF
fig, ax = plt.subplots(figsize=(10, 6))
plot_pacf(data['Units Sold'], ax=ax)
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.title('Partial Autocorrelation Function (PACF)')

plt.show()


# In[20]:


data['diff_price'] = data ['Units Sold'].diff()


# In[21]:


data = data.dropna()


# In[22]:


print(data['diff_price'])


# In[24]:


data['diff_price']= data ['Units Sold'] - data ['Units Sold']. shift(1)


# In[25]:


data = data.dropna()


# In[26]:


print(data['diff_price'])


# In[27]:


data.head()


# In[30]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import pearsonr
import itertools
from statsmodels.tsa.stattools import kpss
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller


# In[31]:


#Compute the autocorrelation
autocorrelation = sm.tsa.acf(data['diff_price'], nlags=20)

# Plot the autocorrelation chart
plt.figure(figsize=(10, 6))
plt.stem(range(len(autocorrelation)), autocorrelation, use_line_collection=True)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Chart')
plt.show()


# In[32]:


ts = data['diff_price']


# In[33]:


# Perform the ADF test
result = adfuller(ts)

# Extract and print the test statistics and p-value
adf_statistic = result[0]
p_value = result[1]
print("ADF Statistic:", adf_statistic)
print("p-value:", p_value)


# In[34]:


result = kpss(ts)

# Extract and print the test statistic and p-value
kpss_statistic = result[0]
p_value = result[1]
print("KPSS Statistic:", kpss_statistic)
print("p-value:", p_value)


# In[35]:


#Plot the Autocorrelation Function (ACF)
plt.figure(figsize=(10, 4))
ax1 = plt.subplot(121)
plot_acf(data['diff_price'], ax=ax1)

# Plot the Partial Autocorrelation Function (PACF)
ax2 = plt.subplot(122)
plot_pacf(data['diff_price'], ax=ax2)

plt.tight_layout()
plt.show()


# In[43]:


p = 2

d = 1

q = 1


# In[44]:


datas = data['diff_price'].values.astype('float64')
model = sm.tsa.ARIMA(datas, order=(p, d, q))

result = model.fit()


# In[45]:


print(result.summary())


# In[46]:


# Make predictions
start_idx = len(datas)
end_idx = len(datas) + len(data) - 1
predictions = result.predict(start=start_idx, end=end_idx)

# Print the predictions
print(predictions)


# In[47]:


actual_values = data['diff_price']


# In[48]:


predictions = predictions[:len(actual_values)]


mae = np.mean(np.abs(predictions - actual_values))
mse = np.mean((predictions - actual_values) ** 2)
rmse = np.sqrt(mse)


print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)


# # **CONCLUSION**

# The ARIMA model is trained and tested for the specific 'ID' provided.
# The model's performance is evaluated using Mean Squared Error (MSE), providing a quantitative measure of the prediction accuracy.
# The actual demand and ARIMA predictions are visualized, allowing for a qualitative assessment of the model's performance over time.
