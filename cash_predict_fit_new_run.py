#import pandas as pd
from cash_predict import *
import pickle

path_project =  './' 
path_data = 'data/' # поддиректория, с CSV файлом данных и pickle обученной модели


cp_new = CashPredict('new_object', 
                       path_2_project = path_project, 
                       path_4_data = path_data)  



# запускает процесс кросс-валидационного обучения и выбора оптимальной SARIMA модели для каждого терминала
cp_new.fit_models() 