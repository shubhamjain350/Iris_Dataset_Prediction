import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


names=['sepal-length','sepal-width','petal-length','petal-width','class']
dataset=pd.read_csv('iris.data',names=names)

x=dataset.iloc[:,0:4].values
y=dataset.iloc[:,4]
#op=pd.get_dummies(data=dataset['class'],drop_first=True)

#print dimension
print(dataset.shape)


#print head 
print (dataset.head())

#describe
print(dataset.describe())

#info
print(dataset.info())

#to see classwise dist
print(dataset.groupby('class'))

#To show box plot in  a single window
dataset.plot(kind='box',subplots=True,layout=(2,2))
plt.show()


dataset.plot(kind='hist',subplots=True,layout=(2,2))
plt.show()

sns.pairplot(data=dataset)
plt.show()

#Dividing into training and testing data
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=7)

#Fitting the model
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


models=[]
models.append(('LR',LogisticRegression()))
models.append(('DTC',DecisionTreeClassifier()))
models.append(('KNC',KNeighborsClassifier()))
models.append(('DA',LinearDiscriminantAnalysis()))
models.append(('NB',GaussianNB()))
models.append(('SVC',SVC()))

result=[]
names=[]

for name,model in models:
    kfold=model_selection.KFold(n_splits=10,random_state=7)
    var=model_selection.cross_val_score(model,x_train,y_train,cv=kfold,scoring='accuracy')
    result.append(var)
    names.append(name)
    msg = "%s: %f (%f)" % (name, var.mean(), var.std())
    print(msg)
	
    svmclassifier=SVC()
    svmclassifier.fit(x_train,y_train)
    y_pred=svmclassifier.predict(x_test)
    print(accuracy_score(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    
    

