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

#Read train data set 
df = pd.read_csv (r'trainData.csv')

#replace missing values NAN 
df= df.replace('[?]', numpy.nan, regex=True)


#drop rows with missing values
df=df.dropna() 


#label encode 
label_encoder = preprocessing.LabelEncoder() 
df['A1']= label_encoder.fit_transform(df['A1']) 
df['A3']= label_encoder.fit_transform(df['A3']) 
df['A4']= label_encoder.fit_transform(df['A4']) 
df['A6']= label_encoder.fit_transform(df['A6']) 
df['A9']= label_encoder.fit_transform(df['A9']) 
df['A15']= label_encoder.fit_transform(df['A15']) 
df['A16']= label_encoder.fit_transform(df['A16']) 
df['A8']= label_encoder.fit_transform(df['A8']) 
df['A11']= label_encoder.fit_transform(df['A11']) 
df['A13']= label_encoder.fit_transform(df['A13']) 


df['A16']= label_encoder.fit_transform(df['A16'])


#Get all attributes
X=df.iloc[:, 0:15]

#split data to train (80%) and test set(20%)
x_train, x_test, y_train, y_test = train_test_split(X,df['A16'], test_size=0.2)

#TRAIN THE MODEL----------------------------------------------------------------------

#random forest classifier model
model = RandomForestClassifier(n_estimators=100, criterion='entropy',random_state=0)
model.fit(x_train,y_train)

#predict using test set
y_pred = model.predict(x_test)

#confusion matrix
cm=confusion_matrix(y_test,y_pred)
cm

#model accuracy
print('Accuracy of test data  ')
print(model.score(x_test,y_test))

print('\n')


#Do Predictions--------------------------------------------------------------
#read test file 
new = pd.read_csv (r'testdata.csv')

#replace missing values with NaN
new= new.replace('[?]', numpy.nan, regex=True)


#replace numerical missing values mean 
imputer = SimpleImputer(missing_values=numpy.nan, strategy='mean')

# fill A2 and A14
imputer=imputer.fit(new[['A2']])
new[['A2']]=imputer.transform(new[['A2']])


imputer=imputer.fit(new[['A14']])
new[['A14']]=imputer.transform(new[['A14']])


#replace non numerical missing values using forward filling
new=new.ffill(axis = 0) 



#label encode 
new['A1']= label_encoder.fit_transform(new['A1']) 
new['A3']= label_encoder.fit_transform(new['A3']) 
new['A4']= label_encoder.fit_transform(new['A4']) 
new['A6']= label_encoder.fit_transform(new['A6']) 
new['A9']= label_encoder.fit_transform(new['A9']) 
new['A15']= label_encoder.fit_transform(new['A15']) 
new['A8']= label_encoder.fit_transform(new['A8']) 
new['A11']= label_encoder.fit_transform(new['A11']) 
new['A13']= label_encoder.fit_transform(new['A13']) 


#take full attributes set 
Xnew=new.iloc[:, 0:15]

#Do predictions
y_prednew = model.predict(Xnew)



#Write to excel--------------------------------------------------------------------
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