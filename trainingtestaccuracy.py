from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
iris=load_iris()
X=iris.data
Y=iris.target
print("Size of Dataset 13{}".format(len(X)))
logreg=LogisticRegression()
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.5,random_state=50)
logreg.fit(x_train,y_train)
predict=logreg.predict(x_test)
print("Accuracy of training set ========::: {}".format(accuracy_score(logreg.predict(x_train),y_train)))
print("Accuracy of test set ========::: {}".format(accuracy_score(predict,y_test)))

