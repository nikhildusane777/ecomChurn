
import constants as const
import pandas as pd
import category_encoders as ce
import pickle


class DataEncoding():

    def __init__(self,df):
        self.df = df
        self.encoder = None

    
    def replaceTargetVariableData(self):
        self.df = self.df.replace({'Churn':const.TARGET_VARIABLE})
        return self.df
    
    def get_dummies(self):
        pass

    def binary_encoding(self,cols_name):
        self.encoder = ce.BinaryEncoder(cols=cols_name)
        if 'Churn' in self.df:
            df_temp = self.df.drop(['Churn'],axis=1)
        else:
            df_temp = self.df
        
        df_binary = self.encoder.fit_transform(df_temp)
        df_binary['Churn'] = self.df['Churn']
        return df_binary

    def save_encoder(self):
        try:
            if self.encoder:
                with open('binaryencoder.pickle','wb') as f:
                    pickle.dump(self.encoder,f)
        except Exception as e:
            print("Exception in encoder save")
            print(e)
    
    def retrive_encoder(self):
        try:
            with open('binaryencoder.pickle','rb') as f:
                encoder_object = pickle.load(f)
                return encoder_object
            
        except Exception as e:
            print("Exception in retrive encoder")
            print(e)

    def training_data_encode_pipeline(self,cols_name):
        try:
            self.replaceTargetVariableData()
            df = self.binary_encoding(cols_name)
            
            return df

        except Exception as e:
            print("Exception as Training Data Encode Pipeline")
            print(e)
