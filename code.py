#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
url='http://bit.ly/w-data'
s_data=pd.read_csv(url)
s_data.head(10)


# In[12]:


s_data.plot(x='Hours',y='Scores',style='o')
plt.xlabel('study_Hours')
plt.ylabel('percentage_Scores')
plt.show()


# In[18]:


x=s_data.iloc[:,:-1].values
y=s_data.iloc[:,1].values


# In[25]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
regressor1=LinearRegression()
regressor1.fit(x,y)


# In[26]:


line=regressor1.coef_*x+regressor1.intercept_
plt.scatter(x,y)
plt.plot(x,line)
plt.show()


# In[27]:


print(x_test)
y_pred=regressor1.predict(x_test)


# In[28]:


df=pd.DataFrame({'actual':y_test , 'predicted':y_pred})
df


# In[31]:


print('training score:', regressor1.score(x_train,y_train))
print('testing score:', regressor1.score(x_test,y_test))


# In[33]:


df.plot(kind='bar', figsize=(7,7))


# In[34]:


hours=9.25
test=np.array([hours])
test=test.reshape(-1,1)
own_pred= regressor1.predict(test)
print('no of hours={}'.format(hours))
print('predicted scores={}'.format(own_pred[0]))


# In[37]:


import numpy as np
from sklearn import metrics
print('mean absolute error:',metrics.mean_absolute_error(y_test,y_pred))
print('mean squared error:',metrics.mean_squared_error(y_test,y_pred))
print('root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print('explained variance score:',metrics.explained_variance_score(y_test,y_pred))


# In[ ]:




