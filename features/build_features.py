import pandas as pd
import numpy as np
from src.features import compute_bin_metrics
from pandas.api.types import is_numeric_dtype
import scorecardpy as sc

def transform_feature_to_woes(feature, target, bin_breaks, is_feature_numeric):
    '''Transform a feature to its WoE representation'''
    
    feature_bin,\
    feature_bin_metrics = compute_bin_metrics\
                                             .woe_bins(feature, 
                                                       target, 
                                                       bin_breaks, 
                                                       is_feature_numeric)
        
    df = feature_bin_metrics.merge(feature_bin, 
                                   how='right', 
                                   left_on=feature_bin.name, 
                                   right_on=feature_bin.name)
    
    woe_values = df['WoE'].rename(f'{feature_bin.name}_woe')
    
    return woe_values

def transform_feature_to_woes_test(feature, target, bin_breaks, feature_test, is_feature_numeric):
    '''Transform a feature to its WoE representation'''
    
    feature_bin_test,\
    feature_bin_metrics = compute_bin_metrics\
                                             .woe_bins_test(feature, 
                                                       target,
                                                       bin_breaks,
                                                       feature_test,
                                                       is_feature_numeric)
        
    df = feature_bin_metrics.merge(feature_bin_test, 
                                   how='right', 
                                   left_on=feature_bin_test.name, 
                                   right_on=feature_bin_test.name)
    
    woe_values = df['WoE'].rename(f'{feature_bin_test.name}_woe')
    
    return woe_values

def make_woe_dataset(df, target, mappings):    
    '''Transform modelling inputs to a WoE dataset'''
    
    feature_woe_list = []

    #Get WoE scores for each feature and add to list
    for feature in df.columns:
        truth_value = is_numeric_dtype(df[feature]) 
        feature_woe_df = transform_feature_to_woes(df[feature], 
                                          target, mappings[feature], 
                                          is_feature_numeric=truth_value)
        
        feature_woe_list.append(feature_woe_df)
   
    woe_df = pd.concat(feature_woe_list, axis=1)
    
    return woe_df

def make_woe_dataset_test(df_train, target, df_test, mappings):    
    '''Transform modelling inputs to a WoE dataset'''
    
    feature_woe_list = []

    #Get WoE scores for each feature and add to list
    for feature in df_train.columns:
        truth_value = is_numeric_dtype(df_train[feature]) 
        feature_woe_df = transform_feature_to_woes_test(df_train[feature], 
                                          target, mappings[feature],
                                          df_test[feature],
                                          is_feature_numeric=truth_value)
        
        feature_woe_list.append(feature_woe_df)
   
    woe_df = pd.concat(feature_woe_list, axis=1)
    
    return woe_df

def get_woe_data(data, feature_name_list, target_name, loan_dict):
    '''Transform data to WoE data based on binning schema'''
    
    loans = data.loc[:, feature_name_list + [target_name]]
    #Bin data based on binning schema
    loan_bins = sc.woebin(loans, y=target_name, breaks_list=loan_dict)
    #Transform dataframe to WoE dataframe
    woe_data = sc.woebin_ply(loans, loan_bins)
    
    woe_vars = [f'{var}_woe' for var in feature_name_list] #WoE feature names
    
    return woe_data[woe_vars], woe_data[target_name], loan_bins