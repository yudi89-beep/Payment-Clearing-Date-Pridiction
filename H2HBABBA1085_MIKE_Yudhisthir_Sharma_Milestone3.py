#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import math
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv('H2HBABBA1085.csv')  # importing the Dataset


# In[3]:


df_N =df[df.clear_date.isnull()==True] # Slicing out the data with clear_date=NaN


# In[4]:


df


# In[5]:


main_train = df[df.clear_date.isnull()==False]  # Creating new dataframe after slicing out the null data from clear_date


# In[6]:


main_train = main_train[main_train.due_in_date.isnull()==False] # Creating new dataframe after slicing out the null data from due_in_date


# In[7]:


main_train.shape


# # Date Time Conversion of columns 

# In[8]:


main_train.info()


# In[9]:


main_train['posting_date'] = pd.to_datetime(main_train['posting_date'])


# In[10]:


main_train['clear_date'] = pd.to_datetime(main_train['clear_date'])


# In[11]:


main_train['due_in_date'] = pd.to_datetime(main_train['due_in_date'], format='%Y%m%d')


# # Preprocessing

# ### Null Imputation

# In[12]:


main_train.isnull().sum()   # checking for number of null values in the feature


# In[13]:


main_train.drop(['area_business'],axis=1,inplace=True)  # removing feature wit null value


# In[14]:


main_train = main_train[main_train.invoice_id.isnull() == False]


# In[15]:


main_train.shape


# ### Constant Columns, Duplicate Data removal

# In[16]:


[x for x in main_train.columns if main_train[x].nunique()==1]   # checking for constant columns 


# In[17]:


main_train.drop(['document type'],axis=1,inplace=True)     # Removal of Constant Columns 
main_train.drop(['posting_id'],axis=1,inplace=True)


# In[18]:


main_train.duplicated().sum()


# In[19]:


main_train.drop_duplicates(keep='first')  # Removal of Duplicate Data while keeping the first occurance


# ### Creation of Target Feature 

# In[20]:


main_train['delay'] = main_train['clear_date'] - main_train['due_in_date']


# In[21]:


main_train['delay'] = main_train['delay'].dt.days


# In[22]:


main_train['delay'] # Target Feature Created
# Here +ve value shows the delay in days for the payment
# -ve value shows that the payment has been done before the due date.


# In[23]:


# Droping the due date and clearing date column as we have created the target feature thus we will not be using these features.
main_train.drop(['clear_date'],axis=1,inplace=True)
main_train.drop(['due_in_date'],axis=1,inplace=True)


# In[24]:


main_train


# ### Sorting the data in ascending order based on the Posting Date

# In[25]:


main_train.sort_values(by=['posting_date'],inplace=True)


# # Encoding The Data

# In[26]:


main_train.info()


# In[27]:


name_customer_enc = LabelEncoder()
name_customer_enc.fit(main_train['name_customer'])
main_train['name_customer_enc'] = name_customer_enc.transform(main_train['name_customer'])


# In[28]:


main_train.drop(['name_customer'],axis=1,inplace=True)


# In[29]:


business_code_enc = LabelEncoder()
business_code_enc.fit(main_train['business_code'])
main_train['business_code_enc'] = business_code_enc.transform(main_train['business_code'])


# In[30]:


main_train.drop(['business_code'],axis=1,inplace=True)


# In[31]:


cust_number_enc = LabelEncoder()
cust_number_enc.fit(main_train['cust_number'])
main_train['cust_number_enc'] = cust_number_enc.transform(main_train['cust_number'])


# In[32]:


main_train.drop(['cust_number'],axis=1,inplace=True)


# In[33]:


invoice_currency_enc = LabelEncoder()
invoice_currency_enc.fit(main_train['invoice_currency'])
main_train['invoice_currency_enc'] = invoice_currency_enc.transform(main_train['invoice_currency'])


# In[34]:


main_train.drop(['invoice_currency'],axis=1,inplace=True)


# In[35]:


cust_payment_terms_enc = LabelEncoder()
cust_payment_terms_enc.fit(main_train['cust_payment_terms'])
main_train['cust_payment_terms_enc'] = cust_payment_terms_enc.transform(main_train['cust_payment_terms'])


# In[36]:


main_train.drop(['cust_payment_terms'],axis=1,inplace=True)


# In[37]:


main_train['buisness_year'] = main_train['buisness_year'].astype(int)


# In[38]:


main_train['doc_id'] = main_train['doc_id'].astype(int)


# In[39]:


main_train.drop(['posting_date'],axis=1,inplace=True)


# # Spliting the data into train test and validate

# In[40]:


x = main_train.drop(['delay'],axis=1)


# In[41]:


y = main_train['delay']


# In[42]:


x_train,x_int_test,y_train,y_int_test = train_test_split(x,y,test_size=0.3,random_state=0,shuffle=False)


# In[43]:


x_train.shape,x_int_test.shape


# In[44]:


x_val,x_test,y_val,y_test = train_test_split(x_int_test,y_int_test,test_size=0.3,random_state=0,shuffle=False)


# In[45]:


x_train.shape,x_val.shape,x_test.shape


# In[46]:


y_train.shape,y_val.shape,y_test.shape


# # Exploratory Data Analysis

# In[47]:


sns.distplot(y_train)


# In[48]:


x_train.merge(y_train, on=x_train.index)


# In[49]:


# Target column is left skewed with oltliers not prominent


# In[50]:


sns.scatterplot( data = x_train.merge(y_train, on=x_train.index), x = 'delay', y='name_customer_enc')


# In[51]:


sns.scatterplot( data = x_train.merge(y_train, on=x_train.index), x = 'delay', y='total_open_amount')


# # Feature Selection

# In[52]:


x_train.merge(y_train,on=x_train.index).corr()  # corelation check 


# In[53]:


corrmat = x_train.merge(y_train,on=x_train.index).corr()
plt.subplots(figsize=(12,10))
sns.heatmap(corrmat)


# In[54]:


colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
sns.heatmap(x_train.merge(y_train,on=x_train.index).corr(),linewidth=0.1,vmax=1.0,square=True,
            cmap=colormap,linecolor='white',annot=True)


# In[55]:


from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(0.8)
sel.fit(x_train)


# In[56]:


sel.variances_


# # Modeling

# In[57]:


## Creating a base model


# In[58]:


base_model = LinearRegression()
base_model.fit(x_train,y_train)   # using Linear Regression


# In[59]:


y_pred = base_model.predict(x_val)


# In[60]:


from sklearn.metrics import mean_squared_error

mean_squared_error(y_val, y_pred)    #checking mean squared error


# In[61]:


math.sqrt(mean_squared_error(y_val, y_pred))  # checking root mean squared error


# In[62]:


pd.DataFrame(zip(y_pred,y_test), columns= ['pred','actual'])   # viewing and comparing the actual value and predicted value.


# In[63]:


from sklearn.linear_model import LogisticRegression      # Checking with Logestic Regression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)


# In[64]:


y_pred = classifier.predict(x_val)


# In[65]:


mean_squared_error(y_val, y_pred)


# In[66]:


math.sqrt(mean_squared_error(y_val, y_pred))


# In[67]:


from sklearn.metrics import accuracy_score

y_pred =classifier.predict(x_test)

accuracy_score(y_test,y_pred)


# In[68]:


from sklearn.linear_model import Ridge   # Checking with Ridge 
model = Ridge()
model.fit(x_train,y_train)


# In[69]:


y_pred = model.predict(x_val)


# In[70]:


mean_squared_error(y_val,y_pred)


# In[71]:


math.sqrt(mean_squared_error(y_val, y_pred))


# In[72]:


from sklearn.ensemble import RandomForestRegressor     # checking with RandomForestRegressor


# In[73]:


rfr = RandomForestRegressor()
rfr.fit(x_train,y_train)


# In[74]:


y_pred = rfr.predict(x_val)


# In[75]:


pd.DataFrame(zip(y_pred,y_test), columns= ['pred','actual'])


# # Tree Based Model

# In[76]:


from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(random_state=0)


# In[77]:


reg.fit(x_train,y_train)


# In[78]:


y_pred = reg.predict(x_val)


# In[79]:


mean_squared_error(y_val,y_pred)


# In[80]:


math.sqrt(mean_squared_error(y_val,y_pred))


# In[81]:


y_pred_test = reg.predict(x_test)
mean_squared_error(y_test, y_pred_test)


# In[82]:


pd.DataFrame(zip(y_pred,y_test), columns= ['pred','actual'])


# # Mapping selected features on the df_N data set that we scliced earlier containing the null values of clear_date

# In[83]:


df_N


# ## Preprocessing and Feature engineering of the dataset with clear_date=null

# In[84]:


df_N.isna().sum()


# In[85]:


x_main_test = df_N.drop(['clear_date'],axis=1)


# In[86]:


x_train.info()


# In[87]:


x_main_test.info()


# In[88]:


x_main_test.drop(['area_business'],axis=1,inplace=True)


# In[89]:


temp = set(x_main_test['name_customer'])-set(name_customer_enc.classes_)
for i in temp:
    name_customer_enc.classes_  = np.append(name_customer_enc.classes_,i)


# In[90]:


x_main_test['name_customer_enc'] = name_customer_enc.transform(x_main_test['name_customer'])


# In[91]:


x_main_test['business_code_enc'] = business_code_enc.transform(x_main_test['business_code'])


# In[92]:


temp2 = set(x_main_test['cust_number'])-set(cust_number_enc.classes_)
for i in temp2:
    cust_number_enc.classes_  = np.append(cust_number_enc.classes_,i)


# In[93]:


x_main_test['cust_number_enc'] = cust_number_enc.transform(x_main_test['cust_number'])


# In[94]:


x_main_test['invoice_currency_enc'] = invoice_currency_enc.transform(x_main_test['invoice_currency'])


# In[95]:


temp3 = set(x_main_test['cust_payment_terms'])-set(cust_payment_terms_enc.classes_)
for i in temp3:
    cust_payment_terms_enc.classes_  = np.append(cust_payment_terms_enc.classes_,i)


# In[96]:


x_main_test['cust_payment_terms_enc'] = cust_payment_terms_enc.transform(x_main_test['cust_payment_terms'])


# In[97]:


x_main_test.drop(['name_customer','business_code','cust_number','invoice_currency','cust_payment_terms','posting_date','due_in_date','posting_id','document type'],axis=1,inplace=True)


# In[98]:


x_main_test['buisness_year'] = x_main_test['buisness_year'].astype(int)


# In[99]:


x_main_test['doc_id'] = x_main_test['doc_id'].astype(int)


# In[100]:


x_main_test.info()


# In[101]:


x_train.info()


# ## Creating new database and printing the prediction for the clear_date = null rows

# In[102]:


Final = base_model.predict(x_main_test)  


# In[103]:


Final = pd.Series(Final,name='delay')


# In[104]:


df_N.reset_index(drop=True,inplace=True)


# In[105]:


Final = df_N.merge(Final, on = x_main_test.index)


# In[106]:


Final.info()


# In[107]:


Final['due_in_date'] = pd.to_datetime(Final['due_in_date'], format='%Y%m%d')


# In[108]:


Final['Pridected_Payment_Date'] = Final['due_in_date'] + pd.to_timedelta(Final['delay'], unit='D')


# In[109]:


Final


# # Bucketizing the Delay into the Aging Bucket 

# In[110]:


bins =[-float('inf'),0,15,float('inf')]


# In[111]:


Final['aging_bucket'] = pd.cut(Final['delay'],bins)


# In[112]:


Final


# In[ ]:




