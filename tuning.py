import sklearn
import numpy
import pandas as pd
from sklearn.impute import SimpleImputer
import sklearn.preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing 
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import statistics

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier 

import xlsxwriter 

train = pd.read_csv (r'dataSet.csv')
test = pd.read_csv (r'testdata.csv')

train['train'] = 1
test['train'] = 0
test['A16'] = 'Success'

df=pd.concat([train,test])

df= df.replace('[?]', numpy.nan, regex=True)
imputer = SimpleImputer(missing_values=numpy.nan, strategy='mean')
imputer=imputer.fit(df[['A2']])
df[['A2']]=imputer.transform(df[['A2']])
imputer=imputer.fit(df[['A14']])
df[['A14']]=imputer.transform(df[['A14']])
df=df.ffill(axis = 0) 

df['A1'] = pd.Categorical(df['A1'])
df['A3'] = pd.Categorical(df['A3'])
df['A4'] = pd.Categorical(df['A4'])
df['A6'] = pd.Categorical(df['A6'])
df['A9'] = pd.Categorical(df['A9'])
df['A15'] = pd.Categorical(df['A15'])
df['A8'] = pd.Categorical(df['A8'])
df['A11'] = pd.Categorical(df['A11'])
df['A13'] = pd.Categorical(df['A13'])

# A2 A5 A7 A10 A12 A14 - scaling to the range 0-1
min_max_scaler = preprocessing.MinMaxScaler()
df[['A2', 'A5', 'A7', 'A10', 'A12', 'A14']] = min_max_scaler.fit_transform(df[['A2', 'A5', 'A7', 'A10', 'A12', 'A14']])

dfDummies = pd.get_dummies(df['A1'], prefix = 'A1')

final = pd.concat([dfDummies, df['A2']], axis=1)

dfDummies = pd.get_dummies(df['A3'], prefix = 'A3')
final = pd.concat([final, dfDummies], axis=1)

dfDummies = pd.get_dummies(df['A4'], prefix = 'A4')
final = pd.concat([final, dfDummies], axis=1)

final = pd.concat([final, df['A5']], axis=1)

dfDummies = pd.get_dummies(df['A6'], prefix = 'A6')
final = pd.concat([final, dfDummies], axis=1)

final = pd.concat([final, df['A7']], axis=1)

dfDummies = pd.get_dummies(df['A8'], prefix = 'A8')
final = pd.concat([final, dfDummies], axis=1)

dfDummies = pd.get_dummies(df['A9'], prefix = 'A9')
final = pd.concat([final, dfDummies], axis=1)

final = pd.concat([final, df['A10']], axis=1)

dfDummies = pd.get_dummies(df['A11'], prefix = 'A11')
final = pd.concat([final, dfDummies], axis=1)

final = pd.concat([final, df['A12']], axis=1)

dfDummies = pd.get_dummies(df['A13'], prefix = 'A13')
final = pd.concat([final, dfDummies], axis=1)

final = pd.concat([final, df['A14']], axis=1)

dfDummies = pd.get_dummies(df['A15'], prefix = 'A15')
final = pd.concat([final, dfDummies], axis=1)

label_encoder = preprocessing.LabelEncoder() 
df['A16']= label_encoder.fit_transform(df['A16'])

final = pd.concat([final, df['A16']], axis=1)

final = pd.concat([final, df['train']], axis=1)

train_df=final[final['train']==1]
test_df=final[final['train']==0]

train_df_new=train_df.drop(['train'], axis=1)           
test_df_new=test_df.drop(['train'], axis=1)

# #shuffle
# train_df_new_shuffled = train_df_new.sample(frac=1).reset_index(drop=True)

X=train_df_new.iloc[:, 0:46]
x_train, x_test, y_train, y_test = train_test_split(X,train_df_new['A16'], test_size=0.2,random_state = 5)

# Hyper parameters to be tested for random forrest classifier
param_grid = {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
 
clf = RandomForestClassifier(criterion='entropy')
CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 5, verbose=10)
CV_rfc.fit(x_train, y_train)

#print the best parameters
print(CV_rfc.best_params_)
# print the final score
print("score={}".format(CV_rfc.score(x_test,y_test)))

