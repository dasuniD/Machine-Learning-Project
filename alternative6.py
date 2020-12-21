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
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
import xlsxwriter 

#Read train data set and test data set
train = pd.read_csv (r'trainData.csv')
test = pd.read_csv (r'testdata.csv')

#Add a column to identify whether train or test data
train['train']=1
test['train']=0

#Concaternate train set and test set 
df=pd.concat([train,test])

#replace missing values with NaN
df= df.replace('[?]', numpy.nan, regex=True)

#replace numerical missing values mean 
imputer = SimpleImputer(missing_values=numpy.nan, strategy='mean')
imputer=imputer.fit(df[['A2']])
df[['A2']]=imputer.transform(df[['A2']])
imputer=imputer.fit(df[['A14']])
df[['A14']]=imputer.transform(df[['A14']])

#replace non-numerical missing values using forward fill
df=df.ffill(axis = 0) 

#hot encode categorical attributes 
df['A1'] = pd.Categorical(df['A1'])
df['A3'] = pd.Categorical(df['A3'])
df['A4'] = pd.Categorical(df['A4'])
df['A6'] = pd.Categorical(df['A6'])
df['A9'] = pd.Categorical(df['A9'])
df['A15'] = pd.Categorical(df['A15'])
df['A8'] = pd.Categorical(df['A8'])
df['A11'] = pd.Categorical(df['A11'])
df['A13'] = pd.Categorical(df['A13'])


#make dummy columns and concaternate them
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



#A16 encode
label_encoder = preprocessing.LabelEncoder() 
df['A16']= label_encoder.fit_transform(df['A16'])

final = pd.concat([final, df['A16']], axis=1)

final = pd.concat([final, df['train']], axis=1)


#devide to train and test again
train_df=final[final['train']==1]
test_df=final[final['train']==0]

#drop last column in both
train_df_new=train_df.drop(['train'],axis=1)           #train set after hot encode
test_df_new=test_df.drop(['train'],axis=1)             #test set after hot encode


#take full attributes set 
X=train_df_new.iloc[:, 0:46]


#split data to train (80%) and test set(20%)
x_train, x_test, y_train, y_test = train_test_split(X,train_df_new['A16'], test_size=0.2,random_state = 5)

#TRAIN THE MODEL----------------------------------------------------------------------------------------------
#----Different algorithms checked-----

model = DecisionTreeClassifier()

model.fit(x_train,y_train)

#predict using test set
y_pred = model.predict(x_test)

#confusion matrix
cm=confusion_matrix(y_test,y_pred)
print(model.score(x_test,y_test))
cm

print('\n')

#Do Predictions--------------------------------------------------------------
#take full attributes set in test set
Xnew=test_df_new.iloc[:, 0:46]

y_prednew = model.predict(Xnew)

final=[]

for i in range(len(y_prednew)):
    if y_prednew[i]==1 :
        final.append('Success')
    if y_prednew[i]==0:
        final.append('Failure')
        
print('Final Predictions  ')        
print(final)

workbook = xlsxwriter.Workbook('submit.xlsx') 

  
# add_sheet is used to create sheet. 
sheet1 = workbook.add_worksheet()  
sheet1.write('A1', 'Id') 
sheet1.write('B1', 'Category') 


row = 1
column = 1
number=1

for item in final : 
  
    # write operation perform 
    sheet1.write(row, 0, number)
    sheet1.write(row, column, item) 
  
    # incrementing the value of row by one 
    # with each iteratons. 
    row += 1
    number=number+1
      

workbook.close() 

#-------------------------------------------------------------------------------------------