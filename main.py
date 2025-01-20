import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA

# Load the dataset
data = pd.read_csv('Copy of sonar data.csv', header=None)

#print(data.head()) //prints first 5 rows of the dataset
#print(data.info()) //prints the information about the dataset
#print(data.describe()) //prints the statistical summary of the dataset
#print(data[60].value_counts()) //prints the count of each class in the dataset
#print(data.groupby(60).mean()) //prints the mean of each class in the dataset

X=data.drop(columns=60,axis=1) #//dropping the target column from the dataset
Y=data[60] #//taking the target column from the dataset

scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X)
#print(X) //prints the features of the dataset
#print(Y) //prints the target of the dataset

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.1, stratify=Y, random_state=1) #//splitting the dataset into training and testing data
#print(X.shape, X_train.shape , X_test.shape ) //prints the shape of the dataset
#print(X_train) 
#print(Y_train) 
model = LogisticRegression()
model.fit(X_train,Y_train) #//training the model

X_train_prediction = model.predict(X_train) 
training_data_accuracy = accuracy_score(X_train_prediction, Y_train) 
print('Accuracy score on training data :', training_data_accuracy)

X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on testing data :', testing_data_accuracy)

scores = cross_val_score(model, X_scaled, Y, cv=5) 
print("Cross-validated accuracy:", scores.mean())

input_data = (0.0079,0.0086,0.0055,0.0250,0.0344,0.0546,0.0528,0.0958,0.1009,0.1240,0.1097,0.1215,0.1874,0.3383,0.3227,0.2723,0.3943,0.6432,0.7271,0.8673,0.9674,0.9847,0.9480,0.8036,0.6833,0.5136,0.3090,0.0832,0.4019,0.2344,0.1905,0.1235,0.1717,0.2351,0.2489,0.3649,0.3382,0.1589,0.0989,0.1089,0.1043,0.0839,0.1391,0.0819,0.0678,0.0663,0.1202,0.0692,0.0152,0.0266,0.0174,0.0176,0.0127,0.0088,0.0098,0.0019,0.0059,0.0058,0.0059,0.0032)
#//input data is an example of the data that we want to predict
input_data_np = np.asarray(input_data)

input_data_reshaped = input_data_np.reshape(1,-1)  #//reshaping the input data
#print(input_data_reshaped) 

prediction = model.predict(input_data_reshaped) 
print(prediction)