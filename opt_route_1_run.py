from opt_route_1 import *
import pickle

import datetime
today = datetime.date.today()

res = optroutefinder_lanch(path_to_times = 'data',
    path_to_coords = 'data',
    path_to_incomes = 'data',
    max_cash = 1000000,
    p_per_day = 0.02/365,
    days_limit = 14,
    car_cost = 20000,
    work_day = 12*60,
    service_time = 10,
    predict_cash_trust=0.9, 
    num_veh=5,
    start_day=1,
    end_day=None,
    cycle_time=300, # от 300 секунд
    koef_priority_0 = 10000000, # 1000000, 5000000
    koef_nes_degree=3.85, # параметры от 1.5 до 4
    koef_step_func=1) # 1,2,3

with open(f'results_{today}_{res.koef_nes_degree}_{res.koef_step_func}.pickle', 'wb') as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)