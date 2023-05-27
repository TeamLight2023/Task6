from opt_route_3 import *
import pickle

import datetime
today = datetime.date.today()

res = optroutefinder_lanch_oneday(path_to_times = 'data/times v4.csv',
    path_to_outliers = 'data/outliers_and_data_per_tid_per_date.csv',
    path_to_day_predict = 'data/predict_out_test_2022-12-01.csv',
    path_to_terms_history = 'data/terms_dict_by_date.json',
    path_to_balance='data/balance_2022-11-30.csv',
    max_cash = 1000000,
    p_per_day = 0.02/365,
    days_limit = 14,
    car_cost = 20000,
    work_day = 12*60,
    service_time = 10,
    p_service_cost = 0.0001,
    predict_cash_trust=0.9,
    num_veh=5,
    cycle_time=60,
    koef_priority_0 = 10000000,
    koef_nes_degree=2.6,
    koef_step_func=1,
    koef_costs_without_vehicle=10000,
    outlier_limit=500000,
    koef_outlier = 0,
    save_history=False)

with open(f'results_oneday_{today}.pickle', 'wb') as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)