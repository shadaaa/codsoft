#!/usr/bin/env python
# coding: utf-8

# # build a model that predicts whether a passenger on the Titanic survived or not. This is a classic beginner project with readily available data.

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
dt = pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\archive (10)\Titanic-Dataset.csv")
dt['Title'] = dt['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())


# In[4]:


dt.isnull().sum()


# In[8]:


dt=dt.drop(['Pclass','SibSp','Parch','Ticket','Fare','Cabin'],axis=1)


# In[6]:


dt['Age'].fillna(dt['Age'].median(),inplace= True)


# In[9]:


dt.isnull().sum()


# In[11]:


dt = dt.drop(['Embarked'],axis=1)


# In[12]:





# In[14]:


#Sex and Title is categorical.Encode it.
l_encoder = LabelEncoder()
dt['sex_encoded']=l_encoder.fit_transform(dt['Sex'])
l_encoder_t =LabelEncoder() 
dt['title_enc'] = l_encoder_t.fit_transform(dt['Title'])
#Feature Selection
x = dt[['Age','sex_encoded','title_enc']]
y = dt['Survived']


# In[30]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# In[31]:


model = RandomForestClassifier(random_state=42)
model.fit(x_train,y_train)


# In[32]:


#predict the model
y_predict = model.predict(x_test)
accuracy=accuracy_score(y_test,y_predict)


# In[33]:


accuracy


# In[34]:


print(classification_report(y_test,y_predict))


# In[44]:


def predict_survival(name):
    # Extract title from name
    title = name.split(',')[1].split('.')[0].strip()
    person_data = dt[dt['Name'].str.contains(name.split(',')[0].strip(), case=False)]
    if person_data.empty:
        return "Name not found in dataset."
    
    age = person_data['Age'].values[0]
    sex = person_data['Sex'].values[0]
    sex_encoded = l_encoder.transform([sex])[0]
    title_enc = l_encoder_t.transform([title])[0]
    input_data = pd.DataFrame([[age, sex_encoded, title_enc]], columns=['Age', 'sex_encoded', 'title_enc'])
    prediction = model.predict(input_data)
    survived = "Survived" if prediction[0] == 1 else "Did not survive"
    
    return f"Prediction: {survived} (Age: {age}, Sex: {sex}, Title: {title})"


name = input("Enter the name:")
print(predict_survival(name))


# In[ ]:





# In[ ]:





# In[ ]:




