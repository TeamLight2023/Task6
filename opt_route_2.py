import pandas as pd
import numpy as np
from datetime import timedelta
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
#from geopy.distance import geodesic as GD
import warnings
warnings.filterwarnings('ignore')


class OptRouteFinder(object):

    def __init__(self, 
    path_to_times, # путь к данным о времени в пути
    path_to_coords, # путь к данным о координатах
    path_to_incomes, # путь к данным о приростах фактических (используются для подсчета затрат в конце дня) и определения баланса
    path_to_outliers, # путь к данным об аутлайерах (выбросах в данных), используются данные только за первые 2 мес
    path_to_predict, # путь к прогнозам
    max_cash = 1000000, # лимит в терминале
    p_per_day = 0.02/365, # % на остаток в терминале
    days_limit = 14, # максимально допустимое время, в течение которого терминал можно не обслуживать
    car_cost = 20000, # стоимость одного броневика на день 
    work_day = 12*60, # рабочий день в минутах
    service_time = 10, # среднее время простоя/обслуживания
    p_service_cost=0.0001, # ставка на обслуживание
    predict_cash_trust=0.9, # доверие к прогнозу остатков (day_predict*(2-predict_cash_trust))
    num_veh = 5, # число броневиков
    cycle_time=60, # время подбора оптимального пути броневиков на 1 день
    koef_priority_0 = 10000000, # коэф. штрафа за пропуск обязательных точек для посещения 1)переполнение 2)прошло 13 дней со дня обсуживания в функции штрафа
    koef_nes_degree=3.5, # коэф. при факторе кол-во дней со дня последнего обсуживания в функции штрафа
    koef_step_func=1, # коэф. формирования ступеней для кол-ва дней со дня последнего обсуживания в функции штрафа
    koef_costs_without_vehicle=10000, # коэф. при затратах на обсуживание и % на остаток в функции штрафа
    outlier_limit=300000, # с какой суммы начинаем учитывать терминалы с выбросами в функции штрафов
    koef_outlier = 0.1): # коэф. при параметре терминалы с фактами выбросов/без в функции штрафов
        
        #максимально допустимая сумма денег в терминале 
        self.max_cash = max_cash
        # % на остаток в терминале
        self.p_per_day = p_per_day
        #максимально допустимое время, в течение которого терминал можно не обслуживать 
        self.days_limit = days_limit
        #стоимость одного броневика на день 
        self.car_cost = car_cost
        #рабочий день в минутах
        self.work_hours = work_day
        #среднее время простоя/обслуживания
        self.service_time = service_time
        # ставка на обслуживание
        self.p_service_cost = p_service_cost
        #затраты на броневик за минуту
        self.car_cost_per_minute = self.car_cost/self.work_hours
        #доверие к прогнозу остатков
        self.predict_cash_trust = predict_cash_trust
        #расположение входных данных
        self.path_to_incomes = path_to_incomes
        self.path_to_times = path_to_times
        self.path_to_coords = path_to_coords
        self.path_to_outliers = path_to_outliers
        self.path_to_predict = path_to_predict        
        # число броневиков
        self.num_veh = num_veh
        # время подбора оптимального пути броневиков на 1 день
        self.cycle_time = cycle_time
        
        #параметры для функции потерь
        self.koef_priority_0 = koef_priority_0
        self.koef_nes_degree =  koef_nes_degree
        self.koef_step_func = koef_step_func
        self.koef_costs_without_vehicle = koef_costs_without_vehicle
        
        self.outlier_limit = outlier_limit
        self.koef_outlier = koef_outlier
        

    def read_data(self):
        """Чтение входных данных"""
        self.coord = pd.read_excel(f'{self.path_to_coords}/terminal_data_hackathon v4.xlsx', sheet_name='TIDS', index_col=0)
        self.incomes = pd.read_excel(f'{self.path_to_incomes}/terminal_data_hackathon v4.xlsx', sheet_name='Incomes', index_col='TID')
        self.times = pd.read_csv(f'{self.path_to_times}/times v4.csv')
        
        outs = pd.read_csv(f'{self.path_to_outliers}/outliers_and_data_per_tid_per_date.csv', sep=',', index_col=0)
        outs_train = outs.iloc[:61,[f'_outl_abs_{self.outlier_limit}_sigma_1' in x for x in outs.columns]].sum() # только на трейне
        self.problem_tid_lst = [int(x[:6]) for x in outs_train[outs_train!=0].index]
        
        self.predictions = pd.read_csv(f'{self.path_to_predict}/predictions_all_period.csv', index_col=0).T
        self.predictions.index = self.incomes.index
        
    def prepare_times(self):    
        """Подготовка данных по временным затратам"""
    
        def approx_time_cost(x, top=14):
            """Приблизительное расстояние от/до ближайшей точки через усреднение расстояния до top ближайших точек"""
            return (x.sort_values()[:top]).mean()
    
        self.time_cost = self.times.groupby('Origin_tid').agg({'Total_Time':approx_time_cost}).rename(
            columns={'Total_Time':'approx_time_cost'})/2 + self.times.groupby('Destination_tid').agg({'Total_Time':approx_time_cost}).rename(
            columns={'Total_Time':'approx_time_cost'})/2 + self.service_time
        self.time_cost['approx_time_cost_rub'] = self.time_cost.approx_time_cost*self.car_cost_per_minute 

        self.times['Total_time_plus_service'] = self.times.Total_Time+self.service_time
        self.min_penalty = np.ceil(self.times['Total_time_plus_service'].max()) # минимальный штраф за пропуск точки (иначе в начале алгоритм будет отрабатывать не на 12ч рабочий день, а на часовой=)
        
        times_c = self.times.copy()
        times_c.loc[-1]=[0,0,0,0] # искусстрвенная точка старта
        distance_matrix_df = pd.pivot_table(times_c, values='Total_time_plus_service', index=['Origin_tid'],
                                    columns=['Destination_tid'], aggfunc=np.sum).fillna(0).apply(np.round).applymap(int)
        self.distance_matrix = np.array(distance_matrix_df) # матрица расстояний

    def prepare_report_data(self):
        """Подготовка данных для симуляции"""
        
        self.balance=pd.DataFrame(data=0, index=self.incomes.index, columns = self.incomes.columns[1:])
        self.balance.insert(loc=0, column='остаток на 31.08.2022 (входящий)',
                            value=self.incomes['остаток на 31.08.2022 (входящий)'])
        self.day_df_start = pd.DataFrame(self.time_cost.approx_time_cost_rub*(-1))
        self.day_df_start.index.name ='TID'
        self.day_df_start['outlier'] = 0
        self.day_df_start.loc[self.problem_tid_lst, 'outlier'] = self.koef_outlier
        
        self.all_terms_set = set(self.balance.index)
        
        self.terms_dict ={0:self.all_terms_set}
        self.bags = {0:set()}
        self.day_df_dict={}
        self.route_times={}
        self.route_vehicle={}
        self.time_vehicle = {}
        self.over_balance_limit = {}
        
    def opt_finder(self, d=1, cycle = None):
                                       
        def service_cost(cash):
            """Издержки на обслуживание одного терминала"""
            return max(self.p_service_cost*cash, 100)
        
        def create_data_model(distance_matrix, start_route_point_pos=0, num_veh=self.num_veh):
            """Подготовка данных под задачу оптимизации"""
            data = {}
            data['distance_matrix'] = distance_matrix
            data['num_vehicles'] = num_veh
            data['depot'] = start_route_point_pos
            return data
    
        def distance_callback(from_index, to_index):
            """Определение расстояния между 2 точками"""
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]
        
        def print_solution(data, manager, routing, solution):
            """Вывод решения"""
            print(f'DAY {d}')
            #print(f'Objective: {solution.ObjectiveValue()}')
            
            dropped_nodes = 'Dropped nodes:'
            dropped_nodes_lst=[]
            for node in range(routing.Size()):
                if routing.IsStart(node) or routing.IsEnd(node):
                    continue
                if solution.Value(routing.NextVar(node)) == node:
                    #dropped_nodes += ' {}'.format(manager.IndexToNode(node))
                    dropped_nodes_lst.append(manager.IndexToNode(node)-1)
            
            max_route_distance = 0
            route_vehicle = {}
            time_vehicle = {}
            for vehicle_id in range(data['num_vehicles']):
                route_vehicle[vehicle_id] = []
                index = routing.Start(vehicle_id)
                plan_output = 'Маршрут для броневика {}:\n'.format(vehicle_id)
                route_distance = 0
                while not routing.IsEnd(index):
                    plan_output += ' {} -> '.format(manager.IndexToNode(index))
                    route_vehicle[vehicle_id].append(manager.IndexToNode(index)-1)
                    previous_index = index
                    index = solution.Value(routing.NextVar(index))
                    route_distance += routing.GetArcCostForVehicle(
                        previous_index, index, vehicle_id)
                plan_output += '{}\n'.format(manager.IndexToNode(index))
                plan_output += 'Время в пути: {}min\n'.format(route_distance)
                time_vehicle[vehicle_id] = route_distance + 10
                #print(plan_output)
                max_route_distance = max(route_distance, max_route_distance)
            print('Максимальное время в пути: {}min'.format(max_route_distance))
            dropped_num = 1630-len(dropped_nodes_lst)
            print(f'Точек объезда в день - {dropped_num}')
            return max_route_distance, dropped_nodes_lst, route_vehicle, time_vehicle
            
        if cycle==None:
            cycle = self.balance.shape[1]
        data = create_data_model(self.distance_matrix)
        manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                       data['num_vehicles'],
                                       data['depot'])
        
        # РЕШЕНИЕ
        self.costs_without_vehicle_sum = 0
        while d < cycle:

            day_predict = self.predictions.iloc[:,d-1] # прогноз на день
            day_df = self.day_df_start.copy()
            day_df['pred_balance'] = self.balance.iloc[:,d-1] + day_predict*(2-self.predict_cash_trust) # ожидаемый баланс на конец дня
            day_df['priority'] = (day_df.pred_balance >= self.max_cash).apply(lambda x: 0 if x==True else -1) #приоритет 0 (высший) для точек с ожидаемым переполнением
            
            day_df['cost_per_service'] = - day_df.pred_balance.apply(lambda x: min(x,self.max_cash)).apply(service_cost) # затраты на обслуживание
            day_df['cash_rate_per_day'] = day_df.pred_balance.apply(lambda x: min(x,self.max_cash))*self.p_per_day # % на остаток
            day_df['all_costs'] = day_df.approx_time_cost_rub + day_df.cost_per_service + day_df.cash_rate_per_day # все оценочные (неточные) затраты (впоследствие не используется в функции штрафов)
            day_df['costs_without_vehicle'] = day_df.cost_per_service + day_df.cash_rate_per_day # затраты на обслуживание и % на остаток (используется)
            
            day_df['pred_balance_no_trust_koef'] = self.balance.iloc[:,d-1] + day_predict 
            
            x=self.days_limit-1
            while x > 0:
                day_df.loc[list(set(day_df.index)&set(self.terms_dict[max(d-x,0)])), 'nes_degree'] = x
                x-=1
            day_df['nes_degree'] = day_df['nes_degree'].fillna(self.days_limit) #кол-во дней со дня последнего обслуживания
            
            if d >=self.days_limit:
                not_served = list(day_df[(day_df['nes_degree'] == self.days_limit)].index)
                day_df.loc[not_served, 'priority'] = 0 #приоритет 0 (высший) для точек, не обсуживаемых за последние 13 дней
            
            # ФУНКИЯ ШТРАФА ЗА ПРОПУСК ТОЧКИ
            #day_df['penalty'] = ((day_df.priority + 1)*10000000 + min_penalty*day_df.nes_degree**(-100/day_df.costs_without_vehicle+2)).apply(int)
            day_df['penalty'] = ((day_df.priority + 1 + day_df.outlier)*self.koef_priority_0 -
            (self.koef_costs_without_vehicle/day_df.costs_without_vehicle)*day_df.nes_degree.apply(lambda x: x//self.koef_step_func)**(self.koef_nes_degree)).apply(int)
            
            # ФОРМИРУЕМ МАРШРУТЫ
            routing = pywrapcp.RoutingModel(manager)
            transit_callback_index = routing.RegisterTransitCallback(distance_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
            
            dimension_name = 'Time'
            routing.AddDimension(
            transit_callback_index,
            0,  # нет ожидания перед посещением (время обслуживание заложено в матрицу расстояний/времени
            self.work_hours-self.service_time,  # максимальное время работы броневика 
            #(в матрице расстояний (в минутах) не учитывается время на обслуживание первой точки, поэтому лимит на 10 мин меньше)
            True,  # на начало рабочего дня время не потрачено
            dimension_name)
            distance_dimension = routing.GetDimensionOrDie(dimension_name)
            distance_dimension.SetGlobalSpanCostCoefficient(150)
            
            penalty =  day_df['penalty'] # штраф за пропуск точек
            for node, p in zip(range(1, len(data['distance_matrix'])), penalty):
                routing.AddDisjunction([manager.NodeToIndex(node)], p)
            
            # стратегии алгоритма
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
            search_parameters.time_limit.seconds = self.cycle_time
            # Решение
            solution = routing.SolveWithParameters(search_parameters)
            
            # Решение - вывод
            if solution:
                self.route_times[d], dropped_ponts, self.route_vehicle[d], self.time_vehicle[d] = print_solution(data, manager, routing, solution)
                print(f'{d} день - Готово')
            else:
                print('Решение не найдено!')
                self.route_times[d] = np.nan
            
            # Наступает конец дня. Появляются данные о реальных приростах.
            # Обновляем баланс и смотрим, не переполнился ли какой-то терминал ввиду выброса/неожиданного притока средств
            self.balance.iloc[:,d] = self.incomes.iloc[:,d] # на прирост обновляем везде
            self.balance.iloc[dropped_ponts, d] = self.balance.iloc[dropped_ponts,d-1] + self.incomes.iloc[dropped_ponts,d] # если не заехали, баланс прошлого дня+прирост
            
            self.over_balance_limit[d] = set(self.balance[self.balance.iloc[:, d] >= self.max_cash].index) # точки, где баланс всё же превысил лимит
            self.balance.iloc[:,d] = self.balance.iloc[:,d].apply(lambda x: self.max_cash if x>self.max_cash else x)
            
            dropped_tid = list(day_df.iloc[dropped_ponts].index) # пропущенные точки
            self.terms_dict[d] = self.all_terms_set - set(dropped_tid) # посещенные точки
            visited_tid = list(self.terms_dict[d]) # посещенные точки
            
            # фактические затраты
            day_df['cost_per_service_fact'] = 0
            day_df['cash_rate_per_day_fact'] = 0
            day_df.loc[visited_tid, 'cost_per_service_fact'] = - self.balance.loc[visited_tid].iloc[:,d].apply(service_cost)
            day_df.loc[dropped_tid, 'cash_rate_per_day_fact'] = - self.balance.iloc[dropped_ponts, d]*self.p_per_day
            day_df.loc[dropped_tid, 'cash_rate_per_day_fact_old_version'] = self.balance.loc[visited_tid].iloc[:,d]*self.p_per_day
            self.costs_without_vehicle_sum+=(day_df.cost_per_service_fact.sum() + day_df.cash_rate_per_day_fact.sum())
            
            #ВАЖНО! self.bags - терминалы с приоритетом 0, в которые обязательно заехать, а мы не заехали. ЕСЛИ ЕСТЬ ХОТЯ БЫ ОДИН, останавливам отработку алгоритма на всем периоде.
            self.bags[d] = set(day_df[day_df.priority==0].index) - self.terms_dict[d]
            self.day_df_dict[d]=day_df
            if len(self.bags[d]) > 0:
                print('ПРОПУЩЕНА КРИТИЧЕСКАЯ ТОЧКА')
                break
            d+=1
            
        return self
        
def optroutefinder_lanch(path_to_times = 'data',
    path_to_coords = 'data',
    path_to_incomes = 'data',
    path_to_outliers = 'data',
    path_to_predict = 'data',
    max_cash = 1000000,
    p_per_day = 0.02/365,
    days_limit = 14,
    car_cost = 20000,
    work_day = 12*60,
    service_time = 10,
    p_service_cost = 0.0001,
    predict_cash_trust=0.9,
    num_veh=5,
    start_day=1,
    end_day=None,
    cycle_time=300,
    koef_priority_0 = 10000000,
    koef_nes_degree=3.7,
    koef_step_func=1,
    koef_costs_without_vehicle=10000,
    outlier_limit=300000,
    koef_outlier = 0.1):
    
    finder = OptRouteFinder(path_to_times=path_to_times,
        path_to_coords=path_to_coords,
        path_to_incomes=path_to_incomes,
        path_to_outliers = path_to_outliers,
        path_to_predict = path_to_predict,
        max_cash = max_cash,
        p_per_day = p_per_day,
        days_limit = days_limit,
        car_cost = car_cost,
        work_day = work_day,
        service_time = service_time,
        p_service_cost=p_service_cost,
        predict_cash_trust=predict_cash_trust,
        num_veh=num_veh,
        cycle_time=cycle_time,
        koef_priority_0=koef_priority_0,
        koef_nes_degree=koef_nes_degree,
        koef_step_func=koef_step_func,
        koef_costs_without_vehicle=koef_costs_without_vehicle,
        outlier_limit = outlier_limit,
        koef_outlier=koef_outlier)
    
    finder.read_data()
    finder.prepare_times()
    finder.prepare_report_data()
        
    finder.opt_finder(d=start_day, cycle = end_day)
    return finder 