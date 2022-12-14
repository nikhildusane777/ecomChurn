import json
from sqlite3 import connect
import psycopg2
import os.path
from psycopg2 import pool
from core.model.recommendation_model import RecommendationModel
import constants as const

class DatabaseConfig:
    
    @classmethod
    def get_connection(cls,conn_config_file='core/databaseConnection/database_config.json'):
        with open(conn_config_file) as config_file:
            conn_config = json.load(config_file)

        schema = conn_config['schema']

        try:
            postgres_sql_pool = psycopg2.pool.SimpleConnectionPool(1,20,dbname=conn_config['dbname'],
                                                                    user=conn_config['user'],
                                                                    host=conn_config['host'],
                                                                    password=conn_config['password'],
                                                                    port=conn_config['port'],
                                                                    options=f'-c search_path={schema}')

            if postgres_sql_pool:
                print("Conenction Pool Created Successfully !!!")
            return postgres_sql_pool
        
        except (Exception,psycopg2.DatabaseError) as error:
            print("Error while connecting to Postgres SQL  :%s" %error)
            

    @classmethod
    def get_data_from_db(cls,postgres_sql_pool):
        connection = postgres_sql_pool.getconn()
        query = const.QUERY
        recommentation_list = []
        recommentation_records = None

        try:
            if connection:
                cursor = connection.cursor()
                cursor.execute(query)
                recommentation_records.cursor.fetchall()

                for row in recommentation_records:
                    recommentation_list.append(RecommendationModel(row[0],row[1]))

                cursor.close()

                postgres_sql_pool.putconn(connection)
                cls.close_connection(postgres_sql_pool)

                return recommentation_list
            
            else:
                return []
        
        except psycopg2.DatabaseError as err:
            print("Error in putting away sql connection")

    
    @classmethod
    def close_connection(cls,postgres_sql_pool):
        try:
            if postgres_sql_pool:
                postgres_sql_pool.closeall()
                print("Database Connection Closed")

        except ConnectionRefusedError as e:
            print("Error While closing postgres sql db connection")


