import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
with open('wdbc.data') as input_file:
   lines = input_file.readlines()
   newLines = []
   for line in lines:
      newLine = line.strip().split(',')
      newLines.append( newLine )

with open('output.csv', 'wb') as test_file:
   file_writer = csv.writer(test_file)
   file_writer.writerows(newLines)
   
dataset = pd.read_csv('cancer.csv')
dataset.info()

mapping = {'M':1, 'B':0}
dataset['diagnosis'] = dataset['diagnosis'].map(mapping)
print dataset['diagnosis']
dataset.groupby('diagnosis').hist(color = 'green', figsize = (15,15))

dataset.isnull().sum()
dataset.isna().sum()
dataset = dataset.drop(columns = ['id '])


X = dataset.iloc[:, 1:31].values
Y = dataset.iloc[:, 0].values
"""
dataframe = pd.DataFrame(Y)

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
"""

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""SVM """

classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

confmatrix = confusion_matrix(y_test, y_pred)

print("Accuracy score with SVM: ",accuracy_score(y_test, y_pred))
#98%

"""Decision Tree"""
    

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

confmatrix = confusion_matrix(y_test, y_pred)

print("Accuracy score with Decision Tree: ",accuracy_score(y_test, y_pred))
#95%
