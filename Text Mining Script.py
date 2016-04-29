# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 09:13:22 2016

@author: Team A-06
"""
import string
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
random.seed(120)
import os
os.getcwd()

Train = pd.read_csv('train.csv',encoding ="ISO-8859-1")
Test = pd.read_csv('test.csv',encoding ="ISO-8859-1")
len(Train)
Train_number = Train.shape[0]

Prod_Desc = pd.read_csv('product_descriptions.csv',encoding ="ISO-8859-1")
Prod_Attr = pd.read_csv('attributes.csv',encoding ="ISO-8859-1")
Prod_Attr.head()

Prod_attrBrandname = Prod_Attr[Prod_Attr.name== "MFG Brand Name"][["product_uid","value"]].rename(columns={"value":"Brand"})
len(Prod_attrBrandname)

# Add features to the Train set from product description file and the attributes file
Train_all = pd.merge(Train,Prod_Desc,on='product_uid',how='left')
Train_all = pd.merge(Train_all,Prod_attrBrandname,on='product_uid',how='left')

#Replacing missing values with Mode
Train_all['Brand'] = np.where(Train_all.Brand.isnull(),Train_all.Brand.mode(),Train_all.Brand)
Train_all.info()

# Add features to the Test set
Test_all = pd.merge(Test,Prod_Desc,on='product_uid',how='left')
Test_all = pd.merge(Test_all,Prod_attrBrandname,on='product_uid',how='left')
#Replacing missing values with Mode
Test_all['Brand'] = np.where(Test_all.Brand.isnull(),Test_all.Brand.mode(),Test_all.Brand)

Test_all.info()

Y_train = Train_all['relevance']

#Parsing Text Columns
from datetime import datetime
start = datetime.now()
# Preprocessing text data
#from nltk.corpus import stopwords
Num = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9}
from stemming.porter2 import stem
import re
#from textblob import TextBlob
def textprocess(term):
    if isinstance(term,str):
        term = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\inches ", term)
        term = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\feet ", term)
        term = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\lb ", term)
        term = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", term)
        term = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", term)
        term = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", term)
        term = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", term)
        term = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", term)
        term = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", term)
        term = term.replace("°"," degrees ")
        term = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", term)
        term = term.replace(" v "," volts ")
        term = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", term)
        term = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", term)
        term = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", term)
        term = term.replace("  "," ")
        term = term.replace(" . "," ")
        term = (" ").join([str(Num[z]) if z in Num else z for z in term.split(" ")])
        term = term.replace("toliet","toilet")
        term = term.replace("airconditioner","air conditioner")
        term = term.replace("vinal","vinyl")
        term = term.replace("vynal","vinyl")
        term = term.replace("skill","skil")
        term = term.replace("snowbl","snow bl")
        term = term.replace("plexigla","plexi gla")
        term = term.replace("rustoleum","rust-oleum")
        term = term.replace("whirpool","whirlpool")
        term = term.replace("whirlpoolga", "whirlpool ga")
        term = term.replace("whirlpoolstainless","whirlpool stainless")
    #Perform Stemming
        stemtxt = [stem(word) for word in term]
        lowertxt = [word.lower() for word in stemtxt]
        
    # Remove Punctuation
        nopunc = [word for word in lowertxt if word not in string.punctuation]
        nopunc = ''.join(nopunc)
    #Remove stopwords
#    cleantxt = [word for word in nopunc.split() if word not in stopwords.words('english')]
#    cleantxt = ''.join(cleantxt)
        return nopunc

#Preprocessing of Train set
start = datetime.now()    
Train_all['product_title'] = Train_all['product_title'].map(lambda x:textprocess(x))
Train_all['search_term'] = Train_all['search_term'].map(lambda x:textprocess(x))
Train_all['product_description'] = Train_all['product_description'].map(lambda x:textprocess(x))
Train_all['Brand'] = Train_all['Brand'].map(lambda x:textprocess(x))
Train_all['prodtext'] = Train_all['search_term']+" "+Train_all['product_title']+" "+Train_all['product_description']+" "+Train_all['Brand']
Train_all['prodtext1'] = Train_all['search_term']+"\t"+Train_all['product_title']+"\t"+Train_all['Brand']+"\t"+Train_all['product_description']
print(datetime.now() - start)

Train_all.prodtext.head()

#Preprocessing for Test set
start = datetime.now()    
Test_all['product_title'] = Test_all['product_title'].map(lambda x:textprocess(x))
Test_all['search_term'] = Test_all['search_term'].map(lambda x:textprocess(x))
Test_all['product_description'] = Test_all['product_description'].map(lambda x:textprocess(x))
Test_all['Brand'] = Test_all['Brand'].map(lambda x:textprocess(x))
Test_all['prodtext'] = Test_all['search_term']+" "+Test_all['product_title']+" "+Test_all['Brand']+" "+Test_all['product_description']
Test_all['prodtext1'] = Test_all['search_term']+"\t"+Test_all['product_title']+"\t"+Test_all['Brand']+"\t"+Test_all['product_description']
print(datetime.now() - start)
Test_all.prodtext.head()

Train_all.info()
Test_all.info()
Y_train = Train_all['relevance']
X_train = Train_all['prodtext']
X_test = Test_all['prodtext']


#Cross validation
start = datetime.now()  

from sklearn.utils import shuffle

X,y = shuffle(X_train,Y_train,random_state = 13)
offset = int(X.shape[0] *0.80)
X_crosstrain , y_crosstrain = X[:offset],y[:offset]
X_crosstest , y_crosstest = X[offset:],y[offset:]

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_df=0.4, max_features=None,analyzer='char_wb',ngram_range=(1,2), 
                        use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = 'english')


X_crosstrain = tfidf.fit_transform(X_crosstrain)

X_crosstest = tfidf.transform(X_crosstest)

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

svd = TruncatedSVD(n_components=300, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)
X_crosstrain_svd = svd.fit_transform(X_crosstrain)
svd.explained_variance_ratio_.sum()
X_crosstest_svd = svd.transform(X_crosstest)

scl = StandardScaler(copy=True, with_mean=True, with_std=True)

X_crosstrain_scl = scl.fit_transform(X_crosstrain_svd)
X_crosstest_scl = scl.transform(X_crosstest_svd)

from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

start = datetime.now() 
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

Model_one = SVR(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, 
                tol=0.001, cache_size=200,verbose=False, max_iter=30000)
Model_one.fit(X_crosstrain_scl,y_crosstrain)
SVMmse = Model_one.predict(X_crosstest_scl)
print("SVM_RMSE:",np.sqrt(mean_squared_error(y_crosstest,SVMmse)))
print(datetime.now() - start)

start = datetime.now() 
from sklearn.neighbors import KNeighborsRegressor
Model_two = KNeighborsRegressor(n_neighbors=11)
Model_two.fit(X_crosstrain_scl,y_crosstrain)
KNNmse = Model_two.predict(X_crosstest_scl)
print("KNN_RMSE:",np.sqrt(mean_squared_error(y_crosstest,KNNmse)))
print(datetime.now() - start)

start = datetime.now()   
from sklearn import ensemble
Model_three = ensemble.RandomForestRegressor(n_estimators = 500,verbose=1,n_jobs=-1,random_state = 120,max_depth=16)
Model_three.fit(X_crosstrain_svd,y_crosstrain)
RFmse = Model_three.predict(X_crosstest_svd)
print("RandomForest_RMSE:",np.sqrt(mean_squared_error(y_crosstest,RFmse)))
print(datetime.now() - start)

start = datetime.now()   
from sklearn.linear_model import BayesianRidge
BR = BayesianRidge(n_iter=500,tol= 0.001,normalize=True).fit(X_crosstrain_scl,y_crosstrain)
pred_BR = BR.predict(X_crosstest_scl)
print("BayesinRidge_RMSE:",np.sqrt(mean_squared_error(y_crosstest,pred_BR)))
print(datetime.now() - start)

start = datetime.now() 
from sklearn.linear_model import LinearRegression
LR = LinearRegression(fit_intercept = True,normalize = True,n_jobs=-1).fit(X_crosstrain_svd,y_crosstrain)
pred_LR = LR.predict(X_crosstest_svd)
print("LinearRegression_RMSE:",np.sqrt(mean_squared_error(y_crosstest,pred_LR)))

print(datetime.now() - start)

#decision tree along with Adaboost
start = datetime.now() 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
AR = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 100),n_estimators = 100, random_state=120).fit(X_crosstrain_scl,y_crosstrain)
pred_AR = AR.predict(X_crosstest_scl)
print("AdaboostDecisionTreeRegression_RMSE:",np.sqrt(mean_squared_error(y_crosstest,pred_AR)))
print(datetime.now() - start)


#***********************************Regular Features**************************************************

def findword(str1, str2):
	return sum(int(str2.find(word)>=0) for word in str1.split())
 
#Features in Train set
Train_all['length_pdt'] = Train_all['product_title'].apply(len)
Train_all['length_st'] = Train_all['search_term'].apply(len)    
Train_all['length_desc'] = Train_all['product_description'].apply(len)
Train_all['Length_Brand'] = Train_all['Brand'].apply(len)
Train_all['search_in_title'] = Train_all['prodtext1'].map(lambda x:findword(x.split('\t')[0],x.split('\t')[1]))
Train_all['search_in_description'] = Train_all['prodtext1'].map(lambda x:findword(x.split('\t')[0],x.split('\t')[2]))
Train_all['search_in_brand'] = Train_all['prodtext1'].map(lambda x:findword(x.split('\t')[0],x.split('\t')[3]))
Train_all['Ratio_title'] = Train_all['search_in_title']/Train_all['length_st']
Train_all['Ratio_desc'] = Train_all['search_in_description']/Train_all['length_st']
Train_all['Ratio_brand'] = Train_all['search_in_brand']/Train_all['length_st']

Train_all.head()
 # Exploratory Data Analysis
Histgram_pdt = Train_all['length_pdt'].plot(bins=50,kind='hist') # Normal
Histgram_st = Train_all['length_st'].plot(bins=50,kind='hist',color='green') #Normal
Histgram_desc = Train_all['length_desc'].plot(bins=100,kind='hist',color='purple') # Right Skwed
Histgram_searchttitle = Train_all['search_in_title'].plot(kind='hist',color='blue')
Histgram_searchbrand = Train_all['search_in_brand'].plot(kind='hist',color='black')
Histgram_Ratiotitle = Train_all['Ratio_title'].plot(kind='hist',color='purple')

# Summary statistics for engineered column - length
print(Train_all['length_pdt'].describe())
print(Train_all['length_st'].describe())
print(Train_all['length_desc'].describe())

# Check the lenghtiest product title and search term individually
print(Train_all[Train_all['length_pdt'] == 147]['product_title'])
print(Train_all[Train_all['length_st'] == 60]['search_term'])


# Histogram of relevance vs lenght of product title and search term
print(Train_all.hist(column='length_pdt',by ='relevance',bins = 50, figsize=(15,6)))
print(Train_all.hist(column='length_st',by ='relevance',bins = 50, figsize=(15,6)))
print(Train_all.hist(column='length_desc',by ='relevance',bins = 100, figsize=(15,6)))
print(Train_all.hist(column='search_in_title',by ='relevance',bins = 10, figsize=(15,6)))
print(Train_all.hist(column='search_in_brand',by ='relevance',bins = 10, figsize=(15,6)))
print(Train_all.hist(column='Ratio_title',by ='relevance',bins = 10, figsize=(15,6)))

#Features in Test set
Test_all['length_pdt'] = Test_all['product_title'].apply(len)
Test_all['length_st'] = Test_all['search_term'].apply(len)
Test_all['length_desc'] = Test_all['product_description'].apply(len)
Test_all['Length_Brand'] = Test_all['Brand'].apply(len)
Test_all['search_in_title'] = Test_all['prodtext1'].map(lambda x:findword(x.split('\t')[0],x.split('\t')[1]))
Test_all['search_in_description'] = Test_all['prodtext1'].map(lambda x:findword(x.split('\t')[0],x.split('\t')[2]))
Test_all['search_in_brand'] = Test_all['prodtext1'].map(lambda x:findword(x.split('\t')[0],x.split('\t')[3]))
Test_all['Ratio_title'] = Test_all['search_in_title']/Test_all['length_st']
Test_all['Ratio_desc'] = Test_all['search_in_description']/Test_all['length_st']
Test_all['Ratio_brand'] = Test_all['search_in_brand']/Test_all['length_st']

X1_train = Train_all.drop(['id','product_uid','relevance','product_title','prodtext','prodtext1','search_term','product_description','Brand'],axis=1)
X1_test = Test_all.drop(['id','product_uid','product_title','prodtext','search_term','prodtext1','product_description','Brand'],axis=1)
Y1_train = Train_all['relevance']

#Cross validation
start = datetime.now()  

X1,y1 = shuffle(X1_train,Y1_train,random_state = 13)
offset = int(X1.shape[0] *0.80)
X1_crosstrain , y1_crosstrain = X1[:offset],y1[:offset]
X1_crosstest , y1_crosstest = X1[offset:],y1[offset:]

#Calculate RMSE for Random Forest
RF1cross = ensemble.RandomForestRegressor(n_estimators = 500,verbose=1,n_jobs=-1,random_state = 120,max_depth=16)
RF1cross_fit = RF1cross.fit(X1_crosstrain,y1_crosstrain)
RF1mse  = RF1cross_fit.predict(X1_crosstest)
print("Random Forest:",np.sqrt(mean_squared_error(y1_crosstest,RF1mse)))
print(datetime.now() - start)

start = datetime.now() 
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
Model1_one = SVR(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, 
                tol=0.001, cache_size=200,verbose=False, max_iter=30000)
Model1_one.fit(X1_crosstrain,y1_crosstrain)
SVM1mse = Model1_one.predict(X1_crosstest)
print("SVM_RMSE:",np.sqrt(mean_squared_error(y1_crosstest,SVM1mse)))
print(datetime.now() - start)

start = datetime.now() 
from sklearn.neighbors import KNeighborsRegressor
Model2_two = KNeighborsRegressor(n_neighbors=31)
Model2_two.fit(X1_crosstrain,y1_crosstrain)
KNN1mse = Model2_two.predict(X1_crosstest)
print("KNN_RMSE:",np.sqrt(mean_squared_error(y1_crosstest,KNN1mse)))
print(datetime.now() - start)
 
## PCA Analysis for Regular Features
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
pca = PCA()
X1_reduced = pca.fit_transform(scale(X1_crosstrain))

np.cumsum(np.round(pca.explained_variance_ratio_,decimals=4)*100) # 12 components explain 90% of the variation
plt.clf()
plt.plot(pca.explained_variance_,linewidth=2)
plt.xlabel('n_components')
plt.ylabel('explained_variance')

RandomForest = ensemble.RandomForestRegressor(verbose=1,n_jobs=-1,random_state = 120)

start = datetime.now()
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

pipe = Pipeline(steps=[('pca', pca), ('RandomForest', RandomForest)])
estimator = GridSearchCV(pipe,dict(pca__n_components=[9,12,15],
                                   RandomForest__n_estimators=[250,500,750]))

estimator.fit(X1_crosstrain,y1_crosstrain)
RF2mse = estimator.predict(X1_crosstest)
print("RF2_RMSE:",np.sqrt(mean_squared_error(y1_crosstest,RF2mse)))
print(datetime.now() - start)

Train = tfidf.fit_transform(X_train)

Test = tfidf.transform(X_test)

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

svd = TruncatedSVD(n_components=300, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)
X_crosstrain_svd = svd.fit_transform(Train)
svd.explained_variance_ratio_.sum()
X_crosstest_svd = svd.transform(Test)

scl = StandardScaler(copy=True, with_mean=True, with_std=True)

X_crosstrain_scl = scl.fit_transform(X_crosstrain_svd)
X_crosstest_scl = scl.transform(X_crosstest_svd)

start = datetime.now()
start = datetime.now()   
from sklearn import ensemble
Model_three = ensemble.RandomForestRegressor(n_estimators = 500,verbose=1,n_jobs=-1,random_state = 120,max_depth=16)
Model_three.fit(X_crosstrain_scl,Y_train)
RFmse = Model_three.predict(X_crosstest_scl)
#print("RandomForest_RMSE:",np.sqrt(mean_squared_error(y_crosstest,RFmse)))
print(datetime.now() - start)

start = datetime.now()
RF1cross = ensemble.RandomForestRegressor(n_estimators = 500,verbose=1,n_jobs=-1,random_state = 120,max_depth=16)
RF1cross_fit = RF1cross.fit(X1_train,Y1_train)
RF1mse  = RF1cross_fit.predict(X1_test)
#print("Random Forest:",np.sqrt(mean_squared_error(y1_crosstest,RF1mse)))
WA = (RF1mse+RFmse)/2
pd.DataFrame({"id": Test_all.id, "relevance": WA}).to_csv('Relevance_file.csv',index=False)
print(datetime.now() - start)

