
from multiprocessing import pool
from unittest import result
import numpy as np
from flask import Flask , jsonify,request
from requests import api
from flask_restful import Api,Resource,reqparse
from flask_api import status
import pandas as pd
import constants as const

from core.databaseConnection.database_connection import DatabaseConfig
from core.preprocessing.dataPreprocessing import DataProcessing
from core.preprocessing.dataEncoding import DataEncoding
from core.dataModel.ml_model import MLModel


import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__,static_url_path="")
app.config['JSONIFY_PRETTY_PRINT_REGULAR'] = True
app.config['BUNDLE_ERRORS'] = True
api = Api(app)






class TrainController(Resource):

    def tempPreprocessing(self,df):
        # Preprocessing will be done here
        df=df.drop('Unnamed: 0',axis=1)
        return df
        
    
    def get(self):
        try:
            # This will be use when we will connect this to database
            
            # db_connection = DatabaseConfig()
            # db_connection_pool = db_connection.get_connection()
            # data_list = db_connection.get_data_from_db(db_connection_pool)

            data_list = pd.read_csv('newdf.csv')

            print(data_list.keys())

            if len(data_list) < const.TRAIN_TEST_SIZE_DATA:
                return jsonify({
                    "data":None,
                    "message":"error",
                    "status":500
                })
            else:
                # data_process = DataProcessing(data_list)


                # df = data_process.get_processed_dataframe()

                df = self.tempPreprocessing(data_list)



                dataEncode  = DataEncoding(df)
                df          = dataEncode.training_data_encode_pipeline(['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','tenure_group'])

                ml_model    = MLModel(df)
                dtree_model ,results= ml_model.fit_model()
                
                file_path = ml_model.save_model(dtree_model)
                dataEncode.save_encoder(file_path)

                return jsonify({
                    "data":None,
                    "message":"success",
                    "status":200
                })

    
        except Exception as e:
            print(f"Exception Occur in main {e}")
            return jsonify({
                "data":None,
                "message":"error",
                "status":500
            })


class MainController(Resource):

    def __init__(self):
        
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('gender',type=str,help="This field is required",required=False,location="json")
        self.reqparse.add_argument('SeniorCitizen',type=int,help="This field is required",required=False,location="json")
        self.reqparse.add_argument('Partner',type=str,help="This field is required",required=False,location="json")
        self.reqparse.add_argument('Dependents',type=str,help="This field is required",required=False,location="json")
        self.reqparse.add_argument('PhoneService',type=str,help="This field is required",required=False,location="json")
        self.reqparse.add_argument('MultipleLines',type=str,help="This field is required",required=False,location="json")
        self.reqparse.add_argument('InternetService',type=str,help="This field is required",required=False,location="json")
        self.reqparse.add_argument('OnlineSecurity',type=str,help="This field is required",required=False,location="json")
        self.reqparse.add_argument('OnlineBackup',type=str,help="This field is required",required=False,location="json")
        self.reqparse.add_argument('DeviceProtection',type=str,help="This field is required",required=False,location="json")
        self.reqparse.add_argument('TechSupport',type=str,help="This field is required",required=False,location="json")
        self.reqparse.add_argument('StreamingTV',type=str,help="This field is required",required=False,location="json")
        self.reqparse.add_argument('StreamingMovies',type=str,help="This field is required",required=False,location="json")
        self.reqparse.add_argument('Contract',type=str,help="This field is required",required=False,location="json")
        self.reqparse.add_argument('PaperlessBilling',type=str,help="This field is required",required=False,location="json")
        self.reqparse.add_argument('PaymentMethod',type=str,help="This field is required",required=False,location="json")
        self.reqparse.add_argument('MonthlyCharges',type=float,help="This field is required",required=False,location="json")
        self.reqparse.add_argument('TotalCharges',type=float,help="This field is required",required=False,location="json")
        self.reqparse.add_argument('tenure_group',type=str,help="This field is required",required=False,location="json")
         
    
    # ['Unnamed: 0', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    #    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    #    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    #    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    #    'MonthlyCharges', 'TotalCharges', 'Churn', 'tenure_group']
    
    def post(self):
        
        try:
            # data = request.json
            # print(data)
            requestbody = self.reqparse.parse_args()
            print(requestbody)
            args    = set(['gender', 'SeniorCitizen', 'Partner', 'Dependents','PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity','OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV','StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod','MonthlyCharges', 'TotalCharges', 'tenure_group'])
            param   = dict(requestbody)

            print("Checkpoint 33")
            
            print(param)
            #not args.issubset(param.keys())
            if None in list(param.values()):
                return jsonify({
                    'data':None,
                    'message':"Parameter missing",
                    'status':500
                })
            
            else:
                df = pd.DataFrame([param])
                ml_model = MLModel(df)
                encoder_obj = DataEncoding(df)

                model,file_path = ml_model.load_model()
                encoder = encoder_obj.retrive_encoder(file_path)

                if not model or not encoder:
                    raise Exception
                
                else:
                    new_df = df[['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','tenure_group']]

                    encoded_df = encoder.transform(df)
                    # encoded_df['SeniorCitizen'] = df['SeniorCitizen']
                    # encoded_df['MonthlyCharges'] = df['MonthlyCharges'] 
                    # encoded_df['TotalCharges'] = df['TotalCharges']
                    predicted = model.predict_proba(encoded_df)
                    pred = predicted.tolist()
                    print("Predictions")
                    churn = int(float(pred[0][1])*100)
                    nochurn = int(float(pred[0][0])*100)

                    result = {"Churn":churn,"No Churn":nochurn}
                    print(pred)

                    return jsonify({
                        'data':result,
                        'message':"success",
                        'status':200
                    })
        except Exception as e:
            print(e)
            return jsonify({
                        'data':'error',
                        'message':"error",
                        'status':500
                    })
    

api.add_resource(MainController,'/getrecommendation/',endpoint='getrecommendation')
api.add_resource(TrainController,'/trainmodel/',endpoint='trainmodel')

if __name__ == "__main__":
    app.run(host=const.HOST,port=const.PORT,debug=True)