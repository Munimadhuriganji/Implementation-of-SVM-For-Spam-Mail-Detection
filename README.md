# Ex.No: 11  Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the packages.
2.Analyse the data.
3.Use modelselection and Countvectorizer to preditct the values.
4.Find the accuracy and display the result.
   
## Program:
```
Program to implement the SVM For Spam Mail Detection..
DEVELOPED BY: GANJI MUNI MADHURI
REGISTER NUMBER : 212223230060
```
```py
import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:
## DATA:
![image](https://github.com/user-attachments/assets/163eb21c-56f6-4d7e-8e25-189b635da175)
## data.shape():
![image](https://github.com/user-attachments/assets/9abf506d-423c-468c-9269-bcd04b9f1ffe)
## x.shape():
![image](https://github.com/user-attachments/assets/e4510765-589c-4e09-8b34-f5cba64e2969)
## y.shape():
![image](https://github.com/user-attachments/assets/4af887a8-36e3-4ce5-8f99-f37c49afd0f6)
## x_train:
![image](https://github.com/user-attachments/assets/9e3f160b-723d-42f1-a653-edd1dae912b9)
## x_train.shape():
![image](https://github.com/user-attachments/assets/6fbb9bab-b2b9-4d2d-8583-a293969a5153)
## y_pred:
![image](https://github.com/user-attachments/assets/7bf6cc42-2a18-4c74-87d4-214f9918d594)
## acc (accuracy):
![image](https://github.com/user-attachments/assets/a3ed050d-b5b2-4f5d-8528-5be66abb2065)
## con (confusion matrix):
![image](https://github.com/user-attachments/assets/49a3e70e-8b9d-4e6c-b3aa-33962c44927e)
## cl (classification report):
![image](https://github.com/user-attachments/assets/d78577fa-e5dd-4a70-a771-54178195e0aa)
## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
