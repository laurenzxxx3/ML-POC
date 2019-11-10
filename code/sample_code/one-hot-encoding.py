#Feature Selection

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
# from sklearn.feature_selection import SelectFromModel
# from sklearn.metrics import accuracy_score

'''General Book Data'''
book_data = pd.read_csv('bestselling_new_matchingbooks.csv')
# print(book_data.head())
# print(book_data.describe())


'''Convert Categorical Data into Numerical Data'''
#general variables
book_category = book_data.Category
book_group = book_data.SubjectGroup
book_publisher = book_data.Publisher
book_date = book_data.PublishDate
book_price = book_data.Price
book_popularity = book_data.popularity
book_gender = book_data.Gender
#book_title = book_data.Title

#First Convert Categorical Data into Numerical Numbers : (Label Enconding)
labelencoder = LabelEncoder()
#Update column with Label Encoder and Create Separate Encoder column(1)
#book_data['CategoryEncoded'] = labelencoder.fit_transform(book_data.Category)
#Update Category wth Label Encoder
book_data.Category = labelencoder.fit_transform(book_data.Category)
book_data.SubjectGroup = labelencoder.fit_transform(book_data.SubjectGroup)
book_data.PublishDate = labelencoder.fit_transform(book_data.PublishDate)
book_data.Gender = labelencoder.fit_transform(book_data.Gender)
book_data.Publisher = labelencoder.fit_transform(book_data.Publisher)

#print(book_data.Publisher)
# book_data.Category = book_data.Category.values.reshape(-1,1)
# print("Gender Array 1")
#print(book_category)
#print(book_data.Category)

#Use of One Hot Encoding with Label Enconders
cat_ohe=OneHotEncoder()
sub_ohe=OneHotEncoder()
gen_ohe=OneHotEncoder()
pub_ohe=OneHotEncoder()
cat_array=cat_ohe.fit_transform(book_data.Category.values.reshape(-1,1)).toarray()
sub_array=sub_ohe.fit_transform(book_data.SubjectGroup.values.reshape(-1,1)).toarray()
gen_array=gen_ohe.fit_transform(book_data.Gender.values.reshape(-1,1)).toarray()
pub_array=pub_ohe.fit_transform(book_data.Publisher.values.reshape(-1,1)).toarray()
#book_data.Gender = ohe.fit_transform(book_data.Gender.values.reshape(-1,1)).toarray()
#book_data.Publisher = ohe.fit_transform(book_data.Publisher.values.reshape(-1,1)).toarray()

# construct dataframe from array, concatenate to original dataframe
book_data_onehot=pd.DataFrame(cat_array,columns = ["Category"+str(int(i)) for i in range(cat_array.shape[1])])
book_data_new=pd.concat([book_data,book_data_onehot],axis=1)

book_data_onehot=pd.DataFrame(sub_array,columns = ["Subject"+str(int(i)) for i in range(sub_array.shape[1])])
book_data_new=pd.concat([book_data_new,book_data_onehot],axis=1)

book_data_onehot=pd.DataFrame(gen_array,columns = ["Gender"+str(int(i)) for i in range(gen_array.shape[1])])
book_data_new=pd.concat([book_data_new,book_data_onehot],axis=1)

book_data_onehot=pd.DataFrame(pub_array,columns = ["Publisher"+str(int(i)) for i in range(pub_array.shape[1])])
book_data_new=pd.concat([book_data_new,book_data_onehot],axis=1)



#book_data_new.to_csv('bestselling_onehot_test.csv')

#print(cat_array)
#print(sub_array)
#book_data.to_csv('bestselling_onehot_test.csv')
#print(book_data.Category)
#Method 2 for OneHotEncoder


'''Split Data into Training and Test datasets, need of splitting the data into training and test datasets'''
X = book_data_new.iloc[:, 4:5] #X (independent columns) will split our data into features
Y = book_data_new.iloc[:, 20] #Y (target data) will split our data in target
#print(X)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)


# #Embedded Method with Random Forest/Tree Classifiers
# ''''Feature Importance will give a score for each feature of the data the higher the score the more important or relvant is the feature towards outputs'''
# #1 Create a random forest classifier
clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
#
# #2 Train the classifier
clf.fit(x_train, y_train)

# #3 Print the name and gini importance of each feature
for feature in zip(clf.feature_importances_):
	print(feature)

sfm=SelectFromModel(clf)
sfm.fit(x_train,y_train)
for i in sfm.get_support(indices=True):
	print(i)
clf.predict(x_test)
# #Dimension Reduction Feature Selection with PCA and SVD
# '''Determine Best Features based on Explained Variance'''
#
# #Filtering Methods
# '''This method will acount for ranking technique
# and uses the rank ordering for variable selection'''
# #Using Pearson Correlation
# plt.figure(figsize=(12,10))
# cor = book_data.corr()
# sns.heatmap(book_data, annot=True, cmap=plt.cm.Reds)
# plt.show()
#
# #Correlation with output variable
# cor_target = abs(cor["popularity"]) #use of correlation of variables based upon popularity metric (target prediction)
#
# #Selecting highly correlated features
# relevant_features = cor_target[cor_target>0.5]
# relevant_features
#
# #ROC Plot
# '''Use of an ROC plot to determine TPR vs FPR, with aim of selecting best feature selection
# This will also be used when comparing the best Machine Learning Models'''
