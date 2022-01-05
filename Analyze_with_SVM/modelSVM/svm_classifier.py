
import pandas as pd
import numpy as np
import re
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
from data_process import data_preprocessing 

from constants import path_test, path_train, model_name

testData = data_preprocessing(path_test)
testData = testData[:1000]
df=data_preprocessing(path_train)
df = df[:1000]
print("passes")

import re
def tokenize(txt):
    tokens=re.split('\W+', txt)
    return tokens

df['Commentaire']=df['Commentaire'].apply(lambda x: tokenize(x.lower()))


df['Commentaire']=[" ".join(review) for review in df['Commentaire'].values]
# print(type(df['Commentaire']))

Train_X = df['Commentaire']
Train_Y = df['Note']
Test_X = testData['Commentaire']
Test_Y = testData['Note']

from sklearn.preprocessing import LabelEncoder
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

# # 5. Word Vectorization

from sklearn.feature_extraction.text import TfidfVectorizer

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(df['Commentaire'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

print(Train_X_Tfidf.shape)
print(Test_X_Tfidf.shape)



# fit the training dataset on the classifier
SVM = svm.LinearSVC(C=1.0,tol=1e-4, multi_class='ovr',max_iter=5000, penalty='l2') #LinearSVC
SVM.fit(Train_X_Tfidf,Train_Y)


# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
print(predictions_SVM)

# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)

conf_mat = confusion_matrix(Test_Y, predictions_SVM)
print(conf_mat)

classification = classification_report(Test_Y, predictions_SVM)
print(classification)

def load_model():
    pickle.dump(SVM,open(model_name,'wb'))
    loaded_model = pickle.load(open(model_name,'rb'))
    return loaded_model



def evaluation_result():
    testData['predict'] = predictions_SVM
    labels_index ={'0,5':0, '1,0':1, '1,5':2, '2,0':3, '2,5':4, '3,0':5, '3,5':6, '4,0':7, '4,5':8, '5,0':9}
    df['Note'] = df['Note'].replace(labels_index)
    reversed_dict= {}
    for k in labels_index:
        reversed_dict[labels_index[k]]=k

    testData['predict'] = testData['predict'].replace(reversed_dict)
    np.savetxt('predicted_dataEval.txt', testData[['Review_id','predict']] , fmt='%s')


evaluation_result()


if __name__ ==" __main__":
    pass

