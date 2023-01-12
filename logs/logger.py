import logging
import os

from itsdangerous import exc

def writeLog(logger,log_level,log_msg):
    try:
        if log_level == "info":
            logger.info(log_msg)
        elif log_level == "debug":
            logger.debug(log_msg)
        else:
            logger.error(log_msg)
    except Exception as e:
        print("Exception in logger")
        print(e)


class AppLogs:
    traing_logger = None
    prediction_logger = None

    def __init__(self):
        self.formatter = logging.Formatter('[%(asctime)s]      %(levelname)s        %(message)s')

    
    def extendable_logger(self,log_name,file_name,level=logging.INFO):
        handler = logging.FileHandler(file_name)
        handler.setFormatter(self.formatter)
        specified_logger = logging.getLogger(log_name)
        specified_logger.setLevel(level)
        if not specified_logger.hasHandlers():
            specified_logger.addHandler(handler)
        specified_logger.propagate = False

        return specified_logger

    def get_training_logger(self):
        training_log_file_path = os.path.join(os.getcwd(),"logs","traning_logs.txt")
        AppLogs.traing_logger = self.extendable_logger("training_logs",training_log_file_path)
        return AppLogs.traing_logger
    
    def get_prediction_logger(self):
        prediction_log_file_path = os.path.join(os.getcwd(),"logs","prediction_logs.txt")
        AppLogs.prediction_logger = self.extendable_logger("prediction_logs",prediction_log_file_path)
        return AppLogs.prediction_logger
        