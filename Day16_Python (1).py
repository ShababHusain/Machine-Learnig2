#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
df=pd.read_csv('D:\\yt\\train.csv')

df.head()


# In[6]:


df.describe()


# In[7]:


#seperating features (X) and outcome(y) from historical data
X=df.drop(['Loan_Status','Loan_ID'], axis=1)
y=df['Loan_Status']
print(X.shape)


# In[8]:


#Handling Null Values
X.isnull().sum()


# In[9]:


X['Gender'].value_counts()


# In[10]:


X['Gender'].fillna("Male", inplace=True)


# In[11]:


X.isnull().sum()


# In[12]:


X['Married'].value_counts()


# In[13]:


X['Married'].fillna("Yes", inplace=True)
X.isnull().sum()


# In[14]:


X['Dependents'].value_counts()


# In[15]:


X['Dependents'].fillna("0", inplace=True)
X.isnull().sum()


# In[16]:


X['Self_Employed'].value_counts()


# In[17]:


X['Self_Employed'].fillna("No", inplace=True)
X.isnull().sum()


# In[18]:


mean_loan=X['LoanAmount'].mean()
X['LoanAmount'].fillna(mean_loan,inplace=True)
X.isnull().sum()


# In[19]:


X['Loan_Amount_Term'].fillna(X['Loan_Amount_Term'].mean(),inplace=True)

X['Credit_History'].fillna(X['Credit_History'].mean(),inplace=True)

X.isnull().sum()


# In[20]:


#Now X does not have any null value
#One hot Encoding- Changing Categorical Values into numerical values
X=pd.get_dummies(X)
X.head()


# In[21]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.30,random_state=30)
print(X_train.shape)

print(X_test.shape)

print(y_test.shape)
X_train.head()


# In[22]:


#Applying Machine Learning Algorithm – Logistic Regression

from sklearn.linear_model import LogisticRegression
Lr = LogisticRegression()

Lr.fit(X_train,y_train)
#Lr now contains the model


# In[23]:


#Applying Machine Learning Algorithm – Support Vector Machines
from sklearn.svm import SVC
svc = SVC()

svc.fit(X_train, y_train)
#svc is another ML model


# In[24]:


#Applying Machine Learning Algorithm – Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

dtf = DecisionTreeClassifier()

dtf.fit(X_train, y_train)


# In[25]:


from sklearn.naive_bayes import GaussianNB

n_b = GaussianNB()

n_b.fit(X_train, y_train)


# In[26]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()  

knn.fit(X_train, y_train)
print(knn.score(X_test,y_test))


# In[27]:


#How to Predict using model
y_predict=Lr.predict(X_test)
y_predict1=svc.predict(X_test)
y_predict2=dtf.predict(X_test)
y_predict3=n_b.predict(X_test)
y_predict4=knn.predict(X_test)
df1=pd.DataFrame({'Actual':y_test,'Predicted_LR':y_predict,'Predicted_svc':y_predict1,'Predicted_dtr':y_predict2,'Predicted_nb':y_predict3,'Predicted_knn':y_predict4 })
df1.to_csv("Day16_Output.csv")


# In[28]:


#Score is simplest parameter to evaluate ML Model for classification problem
# Score=Total Matching/Total data
# Goal of a ML Classification problem is to achieve a score closer to 1.
print(Lr.score(X_test,y_test))
print(svc.score(X_test,y_test))
print(dtf.score(X_test,y_test))
print(n_b.score(X_test,y_test))
print(knn.score(X_test,y_test))


# In[29]:


#The model made by Logistic Regression Algorithm Lr is the best model so far


# In[30]:


#Deployment of model 
gender=input("What is your gender:")
married=input("Married:")
dependents=input("dependents value:")
Education=input("enter your education")
SelfEmployed=input("Self Employed:")
Applicantincome=int(input("enter applicant income"))
coapplicantincome=int(input("enter co applicant income:"))
loanamount=int(input("enter loan amount:"))
loanamountterm=int(input("enter loan amount term:"))
credithistory=int(input("enter credit history:"))
propertyarea=input("enter property area:")
data = [[gender,married,dependents,Education,SelfEmployed,Applicantincome,coapplicantincome,loanamount,loanamountterm,credithistory,propertyarea]]

newdf = pd.DataFrame(data, columns = ['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area'])
newdf.head()


# In[31]:


newdf = pd.get_dummies(newdf)
newdf.head()


# In[32]:


X_train.columns


# In[33]:


missing_cols = set( X_train.columns ) - set( newdf.columns )
print(missing_cols)


# In[34]:


for c in missing_cols:
    newdf[c] = 0


# In[35]:


newdf = newdf[X_train.columns]


# In[36]:


yp=Lr.predict(newdf)
print(yp)


# In[37]:


if (yp[0]=='Y'):
    print("Your Loan is approved, Please contact at HDFC Bank Any Branch for further processing")
else:
    print("Sorry ! Your Loan is not approved")


# In[ ]:





# In[ ]:




