
# coding: utf-8

# ### SHL github project: uat_shl
# 
# * training module: shl_tm
# 
# * prediction module: shl_pm
# 
# * simulation module: shl_sm
# 
# * misc module: shl_mm
# 
# 
# ### data feeds:
# 
# * historical bidding price, per second, time series
# 
# * live bidding price, per second, time series
# 
# ### parameter lookup table: python dictionary
# 
# * parm_si (seasonality index per second)
# 
# * parm_month (parameter like alpha, beta, gamma, etc. per month)

# In[1]:

import pandas as pd


# In[2]:

# function to fetch Seasonality-Index
def shl_intra_fetch_si(ccyy_mm, time, shl_data_parm_si):
#     return shl_data_parm_si[(shl_data_parm_si['ccyy-mm'] == '2017-09') & (shl_data_parm_si['time'] == '11:29:00')]
    return shl_data_parm_si[(shl_data_parm_si['ccyy-mm'] == ccyy_mm) & (shl_data_parm_si['time'] == time)].iloc[0]['si']


# In[3]:

# function to fetch Dynamic-Increment
def shl_intra_fetch_di(ccyy_mm, shl_data_parm_month):
    return shl_data_parm_month[shl_data_parm_month['ccyy-mm'] == ccyy_mm].iloc[0]['di']


# In[4]:

def shl_intra_fetch_previous_n_sec_time_as_str(shl_data_time_field, n):
    return str((pd.to_datetime(shl_data_time_field, format='%H:%M:%S') - pd.Timedelta(seconds=n)).time())

def shl_intra_fetch_future_n_sec_time_as_str(shl_data_time_field, n):
    return str((pd.to_datetime(shl_data_time_field, format='%H:%M:%S') - pd.Timedelta(seconds=-n)).time())


# In[5]:

def shl_initialize(in_ccyy_mm='2017-07'):
    print()
    print('+-----------------------------------------------+')
    print('| shl_initialize()                              |')
    print('+-----------------------------------------------+')
    print()
    global shl_data_parm_si
    global shl_data_parm_month
    shl_data_parm_si = pd.read_csv('parm_si.csv') 
    shl_data_parm_month = pd.read_csv('parm_month.csv') 

    global shl_global_parm_ccyy_mm 
    shl_global_parm_ccyy_mm = in_ccyy_mm
    
    # create default global base price
    global shl_global_parm_base_price
    shl_global_parm_base_price = 10000000

    global shl_global_parm_dynamic_increment
    shl_global_parm_dynamic_increment = shl_intra_fetch_di(shl_global_parm_ccyy_mm, shl_data_parm_month)

    global shl_global_parm_alpha
    shl_global_parm_alpha = shl_data_parm_month[shl_data_parm_month['ccyy-mm'] == shl_global_parm_ccyy_mm].iloc[0]['alpha']
    global shl_global_parm_beta
    shl_global_parm_beta  = shl_data_parm_month[shl_data_parm_month['ccyy-mm'] == shl_global_parm_ccyy_mm].iloc[0]['beta']
    global shl_global_parm_gamma
    shl_global_parm_gamma = shl_data_parm_month[shl_data_parm_month['ccyy-mm'] == shl_global_parm_ccyy_mm].iloc[0]['gamma']
    global shl_global_parm_sec57_weight
    shl_global_parm_sec57_weight = shl_data_parm_month[shl_data_parm_month['ccyy-mm'] == shl_global_parm_ccyy_mm].iloc[0]['sec57-weight']
    global shl_global_parm_month_weight
    shl_global_parm_month_weight = shl_data_parm_month[shl_data_parm_month['ccyy-mm'] == shl_global_parm_ccyy_mm].iloc[0]['month-weight']
    global shl_global_parm_short_weight
    shl_global_parm_short_weight = shl_data_parm_month[shl_data_parm_month['ccyy-mm'] == shl_global_parm_ccyy_mm].iloc[0]['short-weight']

    # default = 0
    global shl_global_parm_short_weight_ratio
    shl_global_parm_short_weight_ratio = 0
    
    # create default average error between 46~50 seconds:
    global shl_global_parm_short_weight_misc
    shl_global_parm_short_weight_misc = 0

    
    print('shl_global_parm_ccyy_mm           : %s' % shl_global_parm_ccyy_mm)
    print('-------------------------------------------------')
    print('shl_global_parm_alpha             : %0.15f' % shl_global_parm_alpha) # used in forecasting
    print('shl_global_parm_beta              : %0.15f' % shl_global_parm_beta)  # used in forecasting
    print('shl_global_parm_gamma             : %0.15f' % shl_global_parm_gamma) # used in forecasting
    print('shl_global_parm_short_weight      : %f' % shl_global_parm_short_weight) # used in forecasting
    print('shl_global_parm_short_weight_ratio: %f' % shl_global_parm_short_weight) # used in forecasting
    print('shl_global_parm_sec57_weight      : %f' % shl_global_parm_sec57_weight) # used in training a model
    print('shl_global_parm_month_weight      : %f' % shl_global_parm_month_weight) # used in training a model
    print('shl_global_parm_dynamic_increment : %d' % shl_global_parm_dynamic_increment)
    print('-------------------------------------------------')

#     plt.figure(figsize=(6,3)) # plot seasonality index
#     plt.plot(shl_data_parm_si[(shl_data_parm_si['ccyy-mm'] == shl_global_parm_ccyy_mm)]['si'])
    
    global shl_data_pm_1_step
    shl_data_pm_1_step = pd.DataFrame() # initialize dataframe of prediction results
    print()
    print('prediction results dataframe: shl_data_pm_1_step')
    print(shl_data_pm_1_step)

    global shl_data_pm_k_step
    shl_data_pm_k_step = pd.DataFrame() # initialize dataframe of prediction results
    print()
    print('prediction results dataframe: shl_data_pm_k_step')
    print(shl_data_pm_k_step)
    


# In[9]:

def shl_predict_price_1_step(in_current_time, in_current_price):
# 11:29:00~11:29:50

    global shl_data_pm_k_step
    
    global shl_global_parm_short_weight_misc
    if in_current_time < '11:29:50': shl_global_parm_short_weight_misc = 0
    
    global shl_global_parm_short_weight_ratio
    
    global shl_global_parm_base_price 


    print()
    print('+-----------------------------------------------+')
    print('| shl_predict_price()                           |')
    print('+-----------------------------------------------+')
    print()
    print('current_ccyy_mm   : %s' % shl_global_parm_ccyy_mm) # str, format: ccyy-mm
    print('in_current_time   : %s' % in_current_time) # str, format: hh:mm:ss
    print('in_current_price  : %d' % in_current_price) # number, format: integer
    print('-------------------------------------------------')

    
    # capture & calculate 11:29:00 bid price - 1 as base price
    if in_current_time == '11:29:00':
        shl_global_parm_base_price = in_current_price -1 
        print('*INFO* At time [ %s ] Set shl_global_parm_base_price : %d ' % (in_current_time, shl_global_parm_base_price)) # Debug
        
    f_current_datetime = shl_global_parm_ccyy_mm + ' ' + in_current_time
    print('*INFO* f_current_datetime   : %s ' %  f_current_datetime)

    # get Seasonality-Index, for current second
    f_current_si = shl_intra_fetch_si(shl_global_parm_ccyy_mm, in_current_time, shl_data_parm_si)
    print('*INFO* f_current_si         : %0.10f ' %  f_current_si) # Debug
    
    # get Seasonality-Index, for current second + 1
    f_1_step_time = shl_intra_fetch_future_n_sec_time_as_str(in_current_time, 1)
    f_1_step_si = shl_intra_fetch_si(shl_global_parm_ccyy_mm, f_1_step_time, shl_data_parm_si)
    print('*INFO* f_1_step_si         : %0.10f ' %  f_1_step_si) # Debug
    
    # calculate price increment: f_current_price4pm
    f_current_price4pm = in_current_price -  shl_global_parm_base_price
    print('*INFO* f_current_price4pm   : %d ' % f_current_price4pm) # Debug
    
    # calculate seasonality adjusted price increment: f_current_price4pmsi
    f_current_price4pmsi = f_current_price4pm / f_current_si
    print('*INFO* f_current_price4pmsi : %0.10f ' % f_current_price4pmsi) # Debug
    

    if in_current_time == '11:29:00':
#         shl_data_pm_k_step_itr = pd.DataFrame() # initialize prediction dataframe at 11:29:00
        print('---- call prediction function shl_pm ---- %s' % in_current_time)
        f_1_step_pred_les_level = f_current_price4pmsi # special handling for 11:29:00
        f_1_step_pred_les_trend = 0 # special handling for 11:29:00
        f_1_step_pred_les = f_1_step_pred_les_level + f_1_step_pred_les_trend
        f_1_step_pred_adj_misc = 0
        f_1_step_pred_price_inc = (f_1_step_pred_les + f_1_step_pred_adj_misc) * f_1_step_si
        f_1_step_pred_price = f_1_step_pred_price_inc + shl_global_parm_base_price
        f_1_step_pred_price_rounded = round(f_1_step_pred_price/100, 0) * 100
        f_1_step_pred_set_price_rounded = f_1_step_pred_price_rounded + shl_global_parm_dynamic_increment
        
    else:
        print('---- call prediction function shl_pm ---- %s' % in_current_time)
        
#       function to get average forecast error between 46~50 seconds: mean(f_current_step_error)
        if in_current_time == '11:29:50':
            sec50_pred_price_inc = shl_data_pm_k_step[(shl_data_pm_k_step['ccyy-mm'] == shl_global_parm_ccyy_mm)                                                 & (shl_data_pm_k_step['f_1_step_time'] ==in_current_time)].iloc[0]['f_1_step_pred_price_inc']
            sec50_error    = sec50_pred_price_inc - f_current_price4pm
            sec46_49_error = (shl_data_pm_k_step['f_1_step_pred_price_inc'].shift(1)[46:50] - shl_data_pm_k_step['f_current_price4pm'][46:50]).sum()
            print('*INFO* sec50_error    : %f' % sec50_error)
            print('*INFO* sec46_49_error : %f' % sec46_49_error)
            
            shl_global_parm_short_weight_misc = (sec50_error + sec46_49_error) / 5
            print('*INFO* shl_global_parm_short_weight_misc  : %f' % shl_global_parm_short_weight_misc)
            
#       ----------------------------------------------------------------------------------------------------        
#       if in_current_time == '11:29:50':
            shl_global_parm_short_weight_ratio = 1
            print('*INFO* shl_global_parm_short_weight_ratio : %d' % shl_global_parm_short_weight_ratio)
        if in_current_time == '11:29:51':
            shl_global_parm_short_weight_ratio = 2
            print('*INFO* shl_global_parm_short_weight_ratio : %d' % shl_global_parm_short_weight_ratio)        
        if in_current_time == '11:29:52':
            shl_global_parm_short_weight_ratio = 3
            print('*INFO* shl_global_parm_short_weight_ratio : %d' % shl_global_parm_short_weight_ratio)        
        if in_current_time == '11:29:53':
            shl_global_parm_short_weight_ratio = 4
            print('*INFO* shl_global_parm_short_weight_ratio : %d' % shl_global_parm_short_weight_ratio)        
        if in_current_time == '11:29:54':
            shl_global_parm_short_weight_ratio = 5
            print('*INFO* shl_global_parm_short_weight_ratio : %d' % shl_global_parm_short_weight_ratio)        
        if in_current_time == '11:29:55':
            shl_global_parm_short_weight_ratio = 6
            print('*INFO* shl_global_parm_short_weight_ratio : %d' % shl_global_parm_short_weight_ratio)        
        if in_current_time == '11:29:56':
            shl_global_parm_short_weight_ratio = 7
            print('*INFO* shl_global_parm_short_weight_ratio : %d' % shl_global_parm_short_weight_ratio)        
        if in_current_time == '11:29:57':
            shl_global_parm_short_weight_ratio = 8
            print('*INFO* shl_global_parm_short_weight_ratio : %d' % shl_global_parm_short_weight_ratio)        
        if in_current_time == '11:29:58':
            shl_global_parm_short_weight_ratio = 9
            print('*INFO* shl_global_parm_short_weight_ratio : %d' % shl_global_parm_short_weight_ratio)        
        if in_current_time == '11:29:59':
            shl_global_parm_short_weight_ratio = 10
            print('*INFO* shl_global_parm_short_weight_ratio : %d' % shl_global_parm_short_weight_ratio)        
        if in_current_time == '11:29:60':
            shl_global_parm_short_weight_ratio = 11
            print('*INFO* shl_global_parm_short_weight_ratio : %d' % shl_global_parm_short_weight_ratio)        
#       ----------------------------------------------------------------------------------------------------        
        
        previous_pred_les_level = shl_data_pm_k_step[(shl_data_pm_k_step['ccyy-mm'] == shl_global_parm_ccyy_mm)                                             & (shl_data_pm_k_step['f_1_step_time'] ==in_current_time)].iloc[0]['f_1_step_pred_les_level']
        print('     previous_pred_les_level : %f' % previous_pred_les_level)
        
        previous_pred_les_trend = shl_data_pm_k_step[(shl_data_pm_k_step['ccyy-mm'] == shl_global_parm_ccyy_mm)                                             & (shl_data_pm_k_step['f_1_step_time'] ==in_current_time)].iloc[0]['f_1_step_pred_les_trend']
        print('     previous_pred_les_trend : %f' % previous_pred_les_trend)

            
        f_1_step_pred_les_level = shl_global_parm_alpha * f_current_price4pmsi                                     + (1 - shl_global_parm_alpha) * (previous_pred_les_level + previous_pred_les_trend)
        print('     f_1_step_pred_les_level  : %f' % f_1_step_pred_les_level)
        f_1_step_pred_les_trend = shl_global_parm_beta * (f_1_step_pred_les_level - previous_pred_les_level)                                     + (1 - shl_global_parm_beta) * previous_pred_les_trend
        print('     f_1_step_pred_les_trend  : %f' % f_1_step_pred_les_trend)
        
        f_1_step_pred_les = f_1_step_pred_les_level + f_1_step_pred_les_trend
        f_1_step_pred_adj_misc = shl_global_parm_short_weight_misc * shl_global_parm_short_weight * shl_global_parm_short_weight_ratio * shl_global_parm_gamma
        print('     les + misc               : %f' % (f_1_step_pred_adj_misc+f_1_step_pred_les))
        f_1_step_pred_price_inc = (f_1_step_pred_les + f_1_step_pred_adj_misc) * f_1_step_si
        print('     f_1_step_pred_price_inc  : %f' % f_1_step_pred_price_inc)
        print('     f_1_step_si              : %f' % f_1_step_si)
        f_1_step_pred_price = f_1_step_pred_price_inc + shl_global_parm_base_price
        f_1_step_pred_price_rounded = round(f_1_step_pred_price/100, 0) * 100
        f_1_step_pred_set_price_rounded = f_1_step_pred_price_rounded + shl_global_parm_dynamic_increment
   
        
    # write results to shl_pm dataframe
            
    shl_data_pm_k_step_itr_dict = {
                         'ccyy-mm' : shl_global_parm_ccyy_mm
                        ,'f_current_datetime' : f_current_datetime
                        ,'f_current_bid' : in_current_price
                        ,'f_current_price4pm' : f_current_price4pm
                        ,'f_current_si' : f_current_si
                        ,'f_current_price4pmsi' :  f_current_price4pmsi
                        ,'f_1_step_time' : f_1_step_time # predicted values/price for next second: in_current_time + 1 second
                        ,'f_1_step_si' : f_1_step_si
                        ,'f_1_step_pred_les_level' : f_1_step_pred_les_level
                        ,'f_1_step_pred_les_trend' : f_1_step_pred_les_trend
                        ,'f_1_step_pred_les' : f_1_step_pred_les
                        ,'f_1_step_pred_adj_misc' : f_1_step_pred_adj_misc
                        ,'f_1_step_pred_price_inc' : f_1_step_pred_price_inc
                        ,'f_1_step_pred_price' : f_1_step_pred_price
                        ,'f_1_step_pred_price_rounded' : f_1_step_pred_price_rounded
                        ,'f_1_step_pred_set_price_rounded' : f_1_step_pred_set_price_rounded
                        }
#     shl_data_pm_k_step_itr =  shl_data_pm_k_step_itr.append(shl_data_pm_k_step_itr_dict, ignore_index=True)
#     shl_data_pm_k_step     =  shl_data_pm_k_step.append(shl_data_pm_k_step_itr_dict, ignore_index=True)
    return shl_data_pm_k_step_itr_dict

    


# In[10]:

# return_value = {'f_1_step_pred_price_rounded', 'f_1_step_pred_set_price_rounded'}
def shl_predict_price_k_step(in_current_time, in_current_price, in_k_seconds=1, return_value='f_1_step_pred_set_price_rounded'):
    global shl_data_pm_1_step
    
    global shl_data_pm_k_step
    shl_data_pm_k_step = shl_data_pm_1_step.copy() 
    
    shl_data_pm_itr_dict = {}
    
    for k in range(1,in_k_seconds+1):
        print()
        print('==>> Forecasting next %3d second/step... ' % k)
        if k == 1:
            print('     procesing current second/step k : ', k)
            input_price = in_current_price
            input_time  = in_current_time
            shl_data_pm_itr_dict = shl_predict_price_1_step(input_time, input_price)
            shl_data_pm_1_step     =  shl_data_pm_1_step.append(shl_data_pm_itr_dict, ignore_index=True)
        else:
            print('     procesing current second/step k : ', k)
            input_price = shl_data_pm_itr_dict['f_1_step_pred_price']
            input_time  = shl_data_pm_itr_dict['f_1_step_time']
            shl_data_pm_itr_dict = shl_predict_price_1_step(input_time, input_price)

        shl_data_pm_k_step     =  shl_data_pm_k_step.append(shl_data_pm_itr_dict, ignore_index=True)
        
    print('*INFO* RETURNED PREDICTION LIST :', shl_data_pm_k_step[shl_data_pm_k_step['f_1_step_time'] > in_current_time][return_value].tolist())
    return shl_data_pm_k_step[shl_data_pm_k_step['f_1_step_time'] > in_current_time][return_value].tolist()


# ### The End
