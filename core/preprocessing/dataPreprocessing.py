

import constants as const
import pandas as pd

class DataProcessing():

    def __inti__(self,data_list):
        self.df = df
        self.data_list = data_list

    def get_dataframe(self):
        request_list = []
        for obj in self.data_list:
            request_body = dict()
            request_body['payment_amount']=obj.payment_amount
            request_body['decision']=obj.decision

            request_list.append(request_body)

        self.df = pd.DataFrame(request_list)

        self.df = self.df.astype({'decision':'str',
                                'reff_reason':'str'})

    
    def remove_null_from_df(self,cols_list):
        self.df = self.df.dropna(subset=cols_list)

    
    def replace_na_with_value(self,cols_name,val):
        self.df.fillna({cols_name:val})
            
    def get_processed_dataframe(self):
        self.get_dataframe()
        self.remove_null_from_df()
        return self.df