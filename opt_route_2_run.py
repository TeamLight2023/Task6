from opt_route_2 import *
import pickle

import datetime
today = datetime.date.today()

res = optroutefinder_lanch(path_to_times='data',
    path_to_coords='data',
    path_to_incomes='data',
    path_to_outliers='data',
    path_to_predict='data',
    max_cash = 1000000,
    p_per_day = 0.02/365,
    days_limit = 14,
    car_cost = 20000,
    work_day = 12*60,
    service_time = 10,
    p_service_cost=0.0001,
    predict_cash_trust=0.9, # (0.9)
    num_veh = 5,
    cycle_time=480,
    koef_priority_0 = 10000000,
    koef_nes_degree=2.6, # (2.6, 2.8, 3.4, 3.6)
    koef_step_func=1,
    koef_costs_without_vehicle=10000,
    outlier_limit= 300000, # (300000, 500000)
    koef_outlier = 0.2) # (0, 0.1, 0.2)

with open(f'results_{today}_{res.koef_nes_degree}_{res.predict_cash_trust}_{res.outlier_limit}_{res.koef_outlier}.pickle', 'wb') as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)