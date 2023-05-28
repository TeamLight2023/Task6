import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import pickle
import os
from datetime import datetime
import pmdarima as pm


# Этот код в репозитории нашей команды
# При создании нового объекта CashPredict необходимо указать только и имя и объект с данными вид pd.DataFrame(index=DateTime, columns=tids), 
# кол-во дней в данных под тест по умолчанию =30, но может быть изменено при создании объекта
# .fit_models() - запускает процесс поиска наилучших SARIMA моделей, для каждого терминала (tid) ищется наилучшая модель,
#             данный метод может выполняться долго, если в текущем объекте много tid без обученных моделей
# .predict_all() - выводит результаты прогнозирования моделей для всех tid, в случае отсутствия модели для какого-то tid, запускает процесс поиска 
#              наилучшей SARIMA модели для данного tid
# .show_prediction(tid) - показывает на графике исходный ts ряд и прогноз SARIMA модели, также выводит в заголовке графика ошибки rmse отдельно на train и test
#                       нет необходимости запускать .predict_all() до вызова .show_prediction(tid), если prediction не был получен заранее, объект получит наилучшую 
#                       SARIMA модель и сформирует prediction для него
# .predict_out(tid, days_out_of_data) - выводит прогноз для заданного кол-ва дней за пределами временного промежутка имеющихся данных
#                                       нет необходимости запускать .fit_all() (или .predict_all()) до вызова .show_prediction(tid), если prediction не был
#                                       получен заранее, объект получит наилучшую  SARIMA модель и сформирует prediction для него 

class CashPredict:
    def __init__(self, name, 
                 data_df = None, 
                 models_dict = {}, 
                 days_for_test = 30,
                 path_2_project = '/',
                 path_4_data = 'data/',
                 path_4_results = 'results/',
                 csv_in = 'cash_in_terminals.csv',
                 col_cash_in = 'sum_start',
                 col_cash_in_excel = 'остаток на 31.08.2022 (входящий)',
                 col_tids = 'TID',
                 sep_in_csv = ';'
                 ): #tid_df,
        self.name = name
        self.days_for_test = days_for_test
        self.data = data_df
        
        self.models = models_dict
        self.has_predictions = {}
        self.fit_status = {}
        self.errors = {}
        
        self.plot_size = (16, 3)
        
        self.path_2_project = path_2_project
        self.path_4_data = path_4_data 
        self.path_4_results = path_4_results
        self.sep = sep_in_csv
        self.csv_in = csv_in
        self.col_cash_in = col_cash_in
        self.col_cash_in_excel = col_cash_in_excel
        self.col_tids = col_tids
        self.has_data = False
        self.has_prepared = False
        self.prepare_data()

    def read_csv_in(self):
        self.data = pd.read_csv(self.path_2_project + 
                                self.path_4_data + 
                                self.csv_in, 
                                sep = self.sep)
        self.has_data = True
    
    def prepare_data(self):
        if self.has_prepared:
            return
        if not self.has_data:
            self.read_csv_in()
        self.data.rename(columns={self.col_cash_in_excel : self.col_cash_in}, inplace=True)
        self.data.drop(columns=[self.col_cash_in], inplace=True)

        self.data = self.data.transpose()
        self.data.columns = self.data.loc[self.col_tids].values
        self.data.drop(index=self.col_tids, axis=0, inplace=True)
        self.data.index = pd.to_datetime(self.data.index)
        
        self.tids = self.data.columns
        #self.tid_df = tid_df.copy()
        self.predictions = pd.DataFrame(index=self.data.index, columns=self.tids)
        # self.predictions_daily = pd.DataFrame(index=self.data.index)
        self.errors = {tid: {} for tid in self.tids} # словари для tid : {'resid':ts, 'rmse': {'train':, 'test': }, 'lbox':{'train':, 'test':}}
        self.fit_status = {tid: True if tid in self.models else False for tid in self.tids}
        self.tid_f = 0
        self.has_prepared = True

    def fit_model(self, tid, season_order=7, rank=None, **kwargs): # если season_order=1, автоарима отключит CV с сезонной компонентой
        if not self.fit_status[tid]:
            ts_train, _ = self.split(self.data[tid])
            self.models[tid] =  pm.auto_arima(ts_train, seasonal=True, m=season_order, d=rank, **kwargs) # если rank=None, автоарима сама подберет нужный порядок
            self.fit_status[tid] = True
    
    def fit_models(self):
        for tid in self.data.columns.tolist()[self.tid_f:]:
            self.tid_f += 1 # здесь инкрементируется заранее, чтобы не пытаться заново  фитить данные с какой-то ошибкой
            self.fit_model(tid)

    def predict_date(self, tid, date):
        if not self.has_predictions.get(tid, False):
            self.predict(tid)
        return self.data.loc[date, tid], round(self.predictions.loc[date, tid]) # (реальный прирост, прогноз)

    def predict(self, tid):
        if not self.fit_status[tid]:
            self.fit_model(tid)
        model_fitted = self.models[tid]
        prediction_train = model_fitted.predict_in_sample()
        prediction_test = model_fitted.predict(n_periods=self.days_for_test)
        self.predictions[tid] = pd.concat([prediction_train, prediction_test])
        self.has_predictions[tid] = True
        self.calc_error(tid)
        
    def predict_all(self):
        for tid in self.tids:
            if not self.has_predictions.get(tid, False):
                self.predict(tid)
        return self.predictions

    def predict_out(self, tid, days_out_of_data):
        if not self.fit_status[tid]:
            self.fit_model(tid)
        # model_fitted = self.models[tid]
        prediction_out = self.models[tid].predict(n_periods=self.days_for_test + days_out_of_data)
        return prediction_out[-days_out_of_data:]

    def split(self, ts):
        return ts[:-self.days_for_test], ts[-self.days_for_test:] # train, test
       
    def calc_error(self, tid, max_lags=14):
        if not self.has_predictions.get(tid, False):
            self.predict(tid)
        y_hat, y_true = self.predictions[tid], self.data[tid]
        y_hat_train, y_hat_test  = self.split(y_hat)
        y_train, y_test = self.split(y_true)
        resid = y_hat - y_true
        self.errors[tid]['resid'] = resid
        self.errors[tid]['rmse'] = {}
        self.errors[tid]['rmse']['train'] = round(mean_squared_error(y_train, y_hat_train) ** 0.5)
        self.errors[tid]['rmse']['test'] = round(mean_squared_error(y_test, y_hat_test) ** 0.5)
        # ts_to_lbox = self.errors[tid]['resid']#[]
        resid_train, resid_test = self.split(resid)
        lbox_train = sm.stats.diagnostic.acorr_ljungbox(resid_train - np.mean(resid_train), lags=max_lags, return_df=True)
        lbox_test = sm.stats.diagnostic.acorr_ljungbox(resid_test - np.mean(resid_test), lags=max_lags, return_df=True)
        self.errors[tid]['lbox'] = {}
        self.errors[tid]['lbox']['train'] = lbox_train
        self.errors[tid]['lbox']['test'] = lbox_test

    def add_comments(self, comments_dict):
        self.comments = comments_dict
        
    def show_prediction(self, tid):  # , show_acf=True, max_lags=7
        if not self.has_predictions.get(tid, False):
            self.predict(tid)
        ax = self.predictions[tid].plot(figsize=(self.plot_size[0], self.plot_size[1]), grid=True)
        self.data[tid].plot(ax=ax, grid=True)
        ax.set_title(f'Реальность и прогноз для терминала {tid}. RMSE на train: { self.errors[tid]["rmse"]["train"] }, RMSE на test: {self.errors[tid]["rmse"]["test"]}', fontsize=10) 
        ax.tick_params(axis='x', labelsize=0)
        ax.axvline(self.data.index[-self.days_for_test], color='red', linestyle=':')

    def show_acf(self, tid, max_lags=14):
        if not self.has_predictions.get(tid, False):
            self.predict(tid)
        fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(self.plot_size[0], 4 * (self.plot_size[1] + .5)))
        t_acf = 'Автокорреляция остатков на '
        t_pacf = 'Частная автокорреляция остатков на '
        plot_acf(self.split(self.errors[tid]['resid'])[0], lags=list(range(1, max_lags+1)), ax=axs[0])
        axs[0].set_title(t_acf + 'train', fontsize=10)
        plot_pacf(self.split(self.errors[tid]['resid'])[0], lags=list(range(1, max_lags+1)), method='ywm', ax=axs[1]) # 'ywm' make PACF in [-1,1]
        axs[1].set_title(t_pacf + 'train', fontsize=10)
        plot_acf(self.split(self.errors[tid]['resid'])[1], lags=list(range(1, max_lags+1)), ax=axs[2])
        axs[2].set_title(t_acf + 'test', fontsize=10)
        plot_pacf(self.split(self.errors[tid]['resid'])[1], lags=list(range(1, max_lags+1)), method='ywm', ax=axs[3]) # 'ywm' make PACF in [-1,1]
        axs[3].set_title(t_pacf + 'test', fontsize=10)

""" Этот вариант метода делает переобучение, что вообще говоря для прода правильно и лучше использовать именно такой подход,
однако в условиях тестирования нашей концепции, это не очень хорошо:
  - делая прогноз на каждый следующий день с использованием не только train данных, но всех доступных данных к предыдущей для прогноза дате
    мы не сможем воспроизвести прогнозные данные на вчера/позавчера и т.д., поскольку вчерашней, позавчерашней модели уже нет
  - можно конечно сохранять модели за предыдущие дни, но тогда наш объект должен хранить как минимум 1630*30 моделей

    def predict_out(self, tid, days_out_of_data):
        if not self.fit_status[tid]:
            self.fit_model(tid)
        model_fitted = self.models[tid]
        prediction_out = self.models[tid].fit_predict(self.data[tid], n_periods=days_out_of_data)
        self.predictions[tid] = self.models[tid].predict_in_sample()
        return prediction_out
"""