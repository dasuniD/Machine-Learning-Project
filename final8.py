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
import xlsxwriter 



#----------------FINAL APPROACH----------------------------------------------------------
#Read train data set and test data set
train = pd.read_csv (r'trainData.csv')
test = pd.read_csv (r'testdata.csv')

#Add a column to identify whether train or test data
train['train']=1
test['train']=0

#Concaternate train set and test set 
df=pd.concat([train,test])

#replace missing values NAN 
df= df.replace('[?]', numpy.nan, regex=True)

#replace numerical missing values mean 
imputer = SimpleImputer(missing_values=numpy.nan, strategy='mean')
imputer=imputer.fit(df[['A2']])
df[['A2']]=imputer.transform(df[['A2']])
imputer=imputer.fit(df[['A14']])
df[['A14']]=imputer.transform(df[['A14']])

#replace non-numerical missing values using forward fill
df=df.ffill(axis = 0) 

#one hot encode attributes 
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

#A16 column encode
label_encoder = preprocessing.LabelEncoder() 
df['A16']= label_encoder.fit_transform(df['A16'])

#concat last 2 columns
final = pd.concat([final, df['A16']], axis=1)
final = pd.concat([final, df['train']], axis=1)


#devide to train and test again
train_df=final[final['train']==1]
test_df=final[final['train']==0]

#drop last column in both
train_df_new=train_df.drop(['train'],axis=1)          
test_df_new=test_df.drop(['train'],axis=1)             

#take full attributes set 
X=train_df_new.iloc[:, 0:46]

#split data to train (80%) and test set(20%)
x_train, x_test, y_train, y_test = train_test_split(X,train_df_new['A16'], test_size=0.2,random_state = 5)

#TRAIN THE MODEL--------------------------------------------------------------

#random forest classifier model-used hyperparameter tuning
model = RandomForestClassifier(n_estimators=2000, criterion='entropy',random_state=1,max_features="sqrt",max_depth=100,min_samples_leaf=4,min_samples_split=10)
model.fit(x_train,y_train)


#predict using test set
y_pred = model.predict(x_test)


#confusion matrix
cm=confusion_matrix(y_test,y_pred)
cm

#model accuracy
print(model.score(x_test,y_test))


print('\n')

#Predictions--------------------------------------------------------------
#attributes set in test set
Xnew=test_df_new.iloc[:, 0:46]

#predict for the data
y_prednew = model.predict(Xnew)

final=[]


#determine success failure
for i in range(len(y_prednew)):
    if y_prednew[i]==1 :
        final.append('Success')
    if y_prednew[i]==0:
        final.append('Failure')
        
print('Final Predictions  ')        
print(final)

#write to excel
workbook = xlsxwriter.Workbook('submit6.xlsx') 
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


'''
#---------------for checking purposes-------------------------------------------------------
#compare with the previous submtted  excel file
df1=pd.read_excel('submit1.xlsx')
df2=pd.read_excel('submit6.xlsx')


df1_new=df1['Category']
df2_new=df2['Category']



comparison_values = df1_new.values == df2_new.values
for item in comparison_values :
    if(item==False):
        print (item)

#-------------------------------------------------------------------------------------------
'''