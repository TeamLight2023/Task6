#import pandas as pd
from cash_predict import *
import pickle
import os

path_project =  './' 
path_data = 'data/' # поддиректория, с CSV файлом данных и pickle обученной модели


path_pickle = path_project + path_data
                 
fname = [fn for fn in os.listdir(path_pickle) if '.pkl' in fn][0]

with open(path_pickle + fname, 'rb') as pkl:
    cp_fitted = pickle.load(pkl)



ft = cp_fitted.fit_status # pt_1
print(f'Обученных моделей всего  : {sum([ft[ind] for ind in ft])}'  ) # 1630
print(f'Данных для обучения всего: {len(cp_fitted.tids) }'     ) # 1630









