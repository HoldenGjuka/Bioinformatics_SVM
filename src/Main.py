#Authors: San Nge, Holden Gjuka, Yaroslav Kravchuk, Keller Debord
#sources: https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python

print("Hello, World!")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split()


from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

from sklearn import metrics

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))