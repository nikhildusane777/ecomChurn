from logs.logger import AppLogs,writeLog
import constants as const
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn.metrics import confusion_matrix
import numpy as np
import time
import pathlib
import os
from sklearn.metrics import accuracy_score
import json
import pandas as pd
from imblearn.over_sampling import SMOTE


class MLModel():
    model_dir = 'artifacts'
    
    def __init__(self,df):
        self.df = df 
    
    def divideDataset(self):
        try:
            X = self.df.drop('Churn',axis=1)
            y = self.df['Churn']

            return X,y
        except Exception as e:
            print("Exception divide dataset")
            print(e)

    def split_train_test(self,X,y):
        try:
            
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2)
            return X_train,X_test,y_train,y_test

        except Exception as e:
            print("Exception Occcor in Split Train Test")
            print(e)
            raise Exception
    
    def upsampling(self,X,y):
        try:
            sm = SMOTE()
            X_resampled, y_resampled = sm.fit_resample(X,y)
            return X_resampled, y_resampled
        except Exception as e:
            print("Error in Upsampling")
            print(e)

    
    def fit_model(self):
        try:
            X,y = self.divideDataset()
            X_resampled,y_resampled = self.upsampling(X,y)
            print("Checkpoint 1")
            print(X_resampled)
            X_train,X_test,y_train,y_test = train_test_split(X_resampled, y_resampled,test_size=0.2)

            dtree = DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=6, min_samples_leaf=8)
            dtree.fit(X_train,y_train)
            ypred = dtree.predict(X_test)

            model_score_r = dtree.score(X_test, y_test)
            print(model_score_r)

            score = accuracy_score(y_test,ypred)

            results = {
                'Model_name':const.MODEL_TYPE,
                'Accuracy':score
            }

            
            return dtree,results

        except Exception as e:
            print("Exception occur in fit model")
            print(e)
            raise Exception

    
    def save_model(self,model):
        try:
            timestamp = int(time.time())
            file_path = os.path.join(os.getcwd(),MLModel.model_dir,str(timestamp))
            print(file_path)
            try:
                print("Checkpoint 1")
                rs = os.makedirs(file_path)
                print("Checkpoint 2")
                print(rs)
            except Exception as e:
                print(e)
            file_name = "DecisionTree_model.sav"
            pickle.dump(model,open(os.path.join(file_path,file_name),'wb'))
            print(file_name)
            return file_path
        except Exception as e:
            print("Error in model save")
            print(e)

    def load_model(self):
        try:
            model_path = max(pathlib.Path(MLModel.model_dir).glob('*/'),key=os.path.getmtime)
            file_path = os.path.join(os.getcwd(),model_path)
            file_name = "DecisionTree_model.sav"
            loaded_model = pd.read_pickle(open(os.path.join(file_path,file_name),'rb'))

            return loaded_model,file_path
        
        except Exception as e:
            print("Exception in load model")
            print(e)
