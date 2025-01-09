import pandas as pd
import numpy as np

def create_application_features(df_dpi, df_fe):
    pivot_duration = df_dpi.pivot_table(index='abon_id', columns='Application', values='SUM_of_Duration_sec', aggfunc='sum', fill_value=0).reset_index()
    pivot_duration = pivot_duration.add_suffix('_duration')
    pivot_duration.rename(columns={'abon_id_duration': 'abon_id'}, inplace=True)
    df_fe = df_fe.merge(pivot_duration, on='abon_id', how='left')
    df_fe.fillna(0, inplace=True)

    features_2_use = ['target','loc_market_share', 'voice_in_td_cnt_mea_mnt1', 'lt',
       'device_days_usage', 'sms_in_cnt_std_mnt3', 'Balance_uah',
       'all_cnt_std_mnt3', 'days_of_end_last_ppm',
       'conn_out_uniq_cnt_mea_mnt1', 'num_act_days_mea_mnt3',
       'imei_mean_long_days_usage', 'all_cnt_std_mnt1',
       'all_cnt_max_mnt1', 'day_end_gba', 'imei_mean_days_usage',
       'data_3g_tv_cnt_std_mnt1', 'voice_out_tar_dur_min_mnt3',
       '690_duration', '246_duration', 'days_of_last_ppm', '678_duration',
       'num_act_days_mea_mnt1', '850_duration', '175_duration',
       'num_act_days_min_mnt3', '933_duration', '257_duration',
       'loc_is_obl_center', '240_duration', '964_duration',
       '882_duration', '272_duration', '254_duration', '877_duration',
       '897_duration', '381_duration', '2018_duration', '677_duration',
       '1414_duration', '992_duration']
    
    df_fe = df_fe[features_2_use]
   
    return df_fe

