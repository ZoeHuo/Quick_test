#!/usr/bin/env python
# coding: utf-8

# # import libraries

# In[2]:

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from sklearn.model_selection import train_test_split


# # load data

# In[3]:


train = pd.read_csv('train_merged.csv')
test = pd.read_csv('test_merged.csv')


# # Model building

#features  = list(train.columns)
target = 'redemption_status'
#features.remove(target)

features = ['avg_coupon_discount',
 'month_trans_ave',
  'avg_use_other_discount',
   'cus_Fuel',
    'cus_Established',
     'cus_Meat',
      'income_bracket_income_11',
       'brand_type_Local',
        'income_bracket_income_12']

X_train, X_valid, y_train, y_valid =  train_test_split(train[features], 
                                                     train[target], 
                                                     test_size = 0.5, 
                                                     random_state = 2021)

X_test = test[features]






class Transformer (object):
    
    def fit(self, X, y = None):
        pass
    
    def transform(self, X, y = None):
        
        features = X.columns
        fea_drop = ['Unnamed: 0',                           'id',
                    'customer_id',                   'coupon_id', 
                    'cp_in_camp',                  'cus_in_camp',
                     'campaign_id',       
                     'brand_0',
                      'avg_sales', 
                     'monthly_cp_disc', 
                    'monthly_ct_trans',
                    'monthly_other_disc', 'monthly_sales',
                    'monthly_freq_cp_disc', 'monthly_freq_other_disc',
#                     'avg_coupon_discount', 
                    'avg_other_discount',
#                     'avg_use_other_discount',    
                     'cus_Alcohol',                      'cus_Bakery',
      'cus_Dairy, Juices & Snacks',            'cus_Flowers & Plants',                      'cus_Garden',               'cus_Miscellaneous',
            'cus_Natural Products',              'cus_Pharmaceutical',
               'cus_Prepared Food',                 'cus_Restauarant',
                      'cus_Salads',                     'cus_Seafood',
            'cus_Skin & Hair Care',                      'cus_Travel',
            'cus_Vegetables (cut)',                      
                 'category_Bakery', 'category_Dairy, Juices & Snacks',
       'category_Flowers & Plants',                   'category_Meat',
       'category_Natural Products',          'category_Packaged Meat',
          'category_Prepared Food',                'category_Seafood',
       'category_Skin & Hair Care', 
                   'cost_saved_coupon', 'coupon_use_rate', 'cost_saved_total']
                   
            
       # features = features.drop(fea_drop)
        features = ['avg_coupon_discount',
                    'month_trans_ave',
                    'avg_use_other_discount',
                    'cus_Fuel',
                    'cus_Established',
                    'cus_Meat',
                                                                                                                                        # 'num_it_in_cp_bin_(0, 5]', 
                                                                                                                                        'income_bracket_income_11',                                                          
                                                                                                                                         'brand_type_Local',
                                                                                                                                        'income_bracket_income_12']

        df = X[features]
#       df = df.drop(df.columns[0], axis = 1)
        
        for col in df.columns:
            df[col] = df[col].abs()
        
        return df
    
    def fit_transform(self, X, y = None):
        return self.transform(X)





# ## logistic regression

steps = [#('tf', Transformer()),
         ('resacle', MinMaxScaler()),
         ('logr', LogisticRegression(class_weight = 'balanced', random_state = 2021))]
model = Pipeline(steps)
model = model.fit(X_train, y_train)

#scaler = MinMaxScaler()
#scaler.fit(X_train)
#X_train = scaler.transform(X_train)

#model = LogisticRegression(class_weight = 'balanced')
#model.fit(X_train, y_train)


with open('new_model.pkl', 'wb') as f:
    pickle.dump(model, f)
# In[28]:


#y_train_pred = model.predict(X_train)


# In[29]:


#confusion_matrix(y_train, y_train_pred)


# In[30]:


#classification_report(y_train, y_train_pred)


# In[31]:


#y_train_proba = model.predict_proba(X_train)


# In[32]:


#fpr, tpr, threshold = roc_curve(y_train, y_train_proba[:,1])
#roc_auc = auc(fpr, tpr)

#plt.title('Receiver Operating Characteristic')
#plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
#plt.legend(loc = 'lower right')
#plt.plot([0, 1], [0, 1],'r--')
#plt.xlim([0, 1])
#plt.ylim([0, 1])
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.show()


# In[33]:


#y_valid_pred = model.predict(X_valid)


# In[34]:


#y_valid_proba = model.predict_proba(X_valid)


# In[35]:


#fpr, tpr, threshold = roc_curve(y_valid, y_valid_proba[:,1])
#roc_auc = auc(fpr, tpr)

#plt.title('Receiver Operating Characteristic')
#plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
#plt.legend(loc = 'lower right')
#plt.plot([0, 1], [0, 1],'r--')
#plt.xlim([0, 1])
#plt.ylim([0, 1])
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.show()


# In[36]:


#y_test_proba = pd.DataFrame(model.predict_proba(X_test))


# In[37]:

#y_test_proba[1]
# In[38]:


#result = pd.DataFrame()
#result['id'] = test['id']
#result['redemption_status'] = y_test_proba[1]


# In[39]:


#result.to_csv( 'y_test_pred_sm.csv', index=False)


# In[40]:


#importances = pd.DataFrame({'coef': model.steps[2][1].coef_[0]})
#importances['features'] = features
#importances= importances.sort_values(by = ['coef'], ascending = False)


# In[41]:


#importances.head(10)


# In[42]:


#importances.tail(10)


# In[44]:


#plt.figure(figsize = (15,4))
#plt.bar(range(importances.shape[0]),importances['coef'], color="r", align = "center")
#plt.xticks(range(importances.shape[0]), importances['features'])


# In[ ]:




