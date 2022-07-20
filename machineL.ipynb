#importing necessary libraries
import pandas as pd
import seaborn as sns
#acquire the data (load the dataset) : 
df = pd.read_csv(r"C:\Users\lenovo\Desktop\Identify Fake Job Posting.csv")
#preprocess the data :
df.head()
df.shape
df.describe()
df.isnull().sum()
#filling necessary columns with respect to mode
df["location"] = df["location"].fillna(df["location"].mode()[0])
df["department"] = df["department "].fillna(df["department "].mode()[0])
df["salary_range"] = df["salary_range "].fillna(df["salary_range "].mode()[0])
df["required_experience "] = df["required_experience "].fillna(df["required_experience 
"].mode()[0])
df["required_education "] = df["required_education "].fillna(df["required_education 
"].mode()[0])
#dropping unnecessary columns
df = df.drop(["company_profile","description","requirements","benefits",
"employment_type","industry","function"],axis=1)
df.columns
#plotting fraudulent and non-fraudulent
sns.countplot(df.fraudulent).set_title('Real & Fradulent')
df.groupby('fraudulent').count()['title'].reset_index().sort_values(by='title',
ascending=False)
from sklearn.model_selection import train_test_split
X = df.drop('fraudulent', axis=1)
y = df['fraudulent']
#converting all categorical columns to numerical
from sklearn.preprocessing import LabelEncoder
Encoder_X = LabelEncoder()
for col in X.columns:
X[col] = Encoder_X.fit_transform(X[col])
Encoder_y = LabelEncoder()
y = Encoder_y.fit_transform(y)
df.head()
#dividing data into training and testing model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#applying Logistic Regression algorithm
from sklearn.linear_model import LogisticRegression
my_model = LogisticRegression()
result = my_model.fit(X_train, y_train)
predictions = result.predict(X_test)
predictions
#finding accuracy score
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)
#finding confusion matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
confusion_mat = confusion_matrix(y_test, predictions)
confusion_df = pd.DataFrame(confusion_mat, index=['Actual neg','Actual pos'], 
columns=['Predicted neg','Predicted pos'])
confusion_df
Color_conf_matrix = sns.heatmap(confusion_df, cmap='coolwarm', annot=True)
#finding classification matrix
from sklearn import metrics
print('\nClassification Report:\n', metrics.classification_report(y_test,predictions))
#deploying the model
pred_new = my_model.predict([[0,6043,2535,758,0,0,1,0,4,1]])
pred_new
#applying DecisionTree algorithm
from sklearn.tree import DecisionTreeClassifier
#dividing data into training and testing model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
my_model1 = DecisionTreeClassifier(random_state=0)
result = my_model1.fit(X_train, y_train)
#dividing model into training into testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
my_model1 = DecisionTreeClassifier(random_state=0)
result = my_model1.fit(X_train, y_train)
#making predictions
predictions = result.predict(X_test)
predictions
#finding accuracy score
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)
#finding confusion and classification matrix
from sklearn import metrics
print("Classification Report\n")
print(metrics.classification_report(y_test, predictions))
print("Confusion Matrix\n")
print(metrics.confusion_matrix(y_test, predictions))
#deploying the model
pred_new = my_model.predict([[0,6043,2535,758,0,0,1,0,4,1]])
pred_new
#applying random forest algorithm
from sklearn.ensemble import RandomForestClassifier
my_model2 = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', 
random_state = 40)
result = my_model2.fit(X_train, y_train)
#making predictions
predicitions = result.predict(X_test)
predicitions
#finding accuracy score
from sklearn import metrics
print("Accuracy : ", metrics.accuracy_score(y_test, predicitions))
#finding confusion and classification matrix
from sklearn import metrics
print("Classification Report\n")
print(metrics.classification_report(y_test, predictions))
print("Confusion Matrix\n")
print(metrics.confusion_matrix(y_test, predictions))
#deploying the model
pred_new = my_model.predict([[0,6043,2535,758,0,0,1,0,4,1]])
pred_new
#applying KNN algorithm
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_test
y_test
#applying standard scaler for better results
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
my_model3 = KNeighborsClassifier(n_neighbors=3)
result = my_model3.fit(X_train, y_train)
#making predictions
predictions = result.predict(X_test)
predictions
#finding accuracy
print('With KNN (n=3) accuracy is : ', result.score(X_test, y_test))
#deploying the model
pred_new = list(result.predict([[0,6043,2535,758,0,0,1,0,4,1]]))
pred_new
#applying Standard Scaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#applying SVM algorithm
from sklearn.svm import SVC
my_model4 = SVC(kernel = 'rbf', random_state = 0)
result = my_model4.fit(X_train, y_train)
#making predictions
predictions = result.predict(X_test)
predictions
#finding confusion matrix
import seaborn as sn
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
sn.heatmap(cm, annot = True, fmt='2.0f')
#finding accuracy
from sklearn import metrics
print("Accuracy : ", metrics.accuracy_score(y_test, predictions))
#deploying the model
new_pred = list(result.predict([[1,4,4,7,0,0,1,1,1]]))
new_pred





