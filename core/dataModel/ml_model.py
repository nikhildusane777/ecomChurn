import constants as const
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn.metrics import confusion_matrix
import numpy as np
from imblearn.over_sampling import SMOTE


class MLModel():
    
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

            
            return dtree

        except Exception as e:
            print("Exception occur in fit model")
            print(e)
            raise Exception

    
    def save_model(self,model):
        try:
            file_name = "DecisionTree_model.sav"
            pickle.dump(model,open(file_name,'wb'))
        except Exception as e:
            print("Error in model save")
            print(e)

    def load_model(self):
        try:
            loaded_model = pickle.load(open('DecisionTree_model.sav','rb'))
            return loaded_model
        
        except Exception as e:
            print("Exception in load model")
            print(e)
