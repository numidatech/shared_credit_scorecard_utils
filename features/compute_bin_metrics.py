import pandas as pd
import numpy as np
import portion


def map_categorical_features(categorical_group):
    '''Function creates a name to call the list of groups and maps the name to the list of groups'''
    
    categorical_map = dict()
    
    #Check if categorical_group is a list of lists
    if any(isinstance(sub_list,list) for sub_list in categorical_group):
        for categories in categorical_group:
            categorical_map[categories[0] + '_and_co'] = categories #Name category_list after first item in list
    
    else:
        #Convert list to list of lists with each element transformed to a list
        element_list_of_lists = [[element] for element in categorical_group]

        for category in categorical_group:
            categorical_map[category] = [category]
        
    
    return categorical_map
  
def map_numerical_features(bin_breaks):
  '''Creates intervals and associated descriptors, and maps descriptors to 
  intervals in a dictionary
  
  Parameters
  ----------
  bin_breaks: numerical list
  
  Returns
  -------
  Dictionary mapping descriptors (string type) to intervals (pd.Interval type)
  '''
  numerical_map = {}
  bin_breaks = sorted(bin_breaks)
  n_breaks = len(bin_breaks)

  if n_breaks == 1:
    numerical_map[
                  str(pd.Interval(
                      left=float('-inf'), right=bin_breaks[0], closed='neither'
                      ))
                  ] = pd.Interval(
                      left=float('-inf'), right=bin_breaks[0], closed='neither'
                      )

    numerical_map[
                  str(pd.Interval(
                      left=bin_breaks[0], right=float('inf'), closed='left'
                      ))
                  ] = pd.Interval(
                      left=bin_breaks[0], right=float('inf'), closed='left'
                      )

  if n_breaks > 1:
    for idx, bound in enumerate(bin_breaks):
      if idx == 0:
        numerical_map[
                      str(pd.Interval(
                          left=float('-inf'), right=bound, closed='neither'
                          ))
                      ] = pd.Interval(
                          left=float('-inf'), right=bound, closed='neither'
                          )

        numerical_map[
                      str(pd.Interval(
                          left=bound, right=bin_breaks[idx+1], closed='left'
                          ))
                      ] = pd.Interval(
                          left=bound, right=bin_breaks[idx+1], closed='left'
                          )

      elif idx == n_breaks-1:
        numerical_map[
                      str(pd.Interval(
                          left=bound, right=float('+inf'), closed='left'
                          ))
                      ] = pd.Interval(
                          left=bound, right=float('+inf'), closed='left'
                          )
        
      else:
        numerical_map[
                      str(pd.Interval(
                          left=bound, right=bin_breaks[idx+1], closed='left'
                          ))
                      ] = pd.Interval(
                          left=bound, right=bin_breaks[idx+1], closed='left'
                          )
     
  return numerical_map

def assign_bin(x, feature_map):
  '''Returns the respective bin for a value by searching a dictionary

  Parameters
  ---------
  x: Value requiring a bin.
      Value types - int, float, string
  
  feature_map: Dictionary, mapping bins to values within the bins. 

  Returns
  ------
  The bin (str type) for the respective value
  '''

  if pd.isnull(x):
    return 'missing'
  else:
    for bin_name, bin in feature_map.items():
      if x in bin:
        return bin_name
        
def transform_to_bin(feature_series, feature_map):
  '''Converts a feature to its bin representation

  Parameters
  ----------
  feature: pd.series representing a feature

  feature_map: Dictionary that maps a bin to its values

  Returns
  -------
  pd.series of bins 
  '''
  bin = feature_series.apply(assign_bin, feature_map=feature_map)
  
  return bin.rename(f'{feature_series.name}_bin')

def calc_feature_metrics(feature_bin, target, is_feature_numeric=True):
  '''Generate statistics for the binned feature

  Parameters
  ---------
  feature_bin: pd.Series of bins

  target: Dependent variable - binary

  numerical: bool that controls code execution based on whether the feature 
  is numerical or categorical

  Returns
  -------
  A dataframe of calculated metrics for the feature based on binned values.
  Includes Weight of Evidence and Information Value
  '''
  df_temp = pd.concat([feature_bin, target], axis=1)
  
  df = df_temp.groupby(df_temp.columns[0], as_index=False).agg(
      n_obs=(df_temp.columns.values[1], 'count'),
      bad_rate=(df_temp.columns.values[1], 'mean'))
  df['prop_n'] = df['n_obs'] / df['n_obs'].sum()
  df['n_good'] = df['n_obs'] * (1-df['bad_rate'])
  df['n_bad'] = df['n_obs'] * df['bad_rate']
  df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
  df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
  df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
  df['IV'] = ((df['prop_n_good'] - df['prop_n_bad']) * df['WoE']).sum()

  #If the feature is numerical we sort by interval in ascending order. However,
  #we must first convert the 'missing' descriptor with np.nan, and get the 
  #interval from the string version as a new feature.
  if is_feature_numeric:
    def sort_helper(x):
      '''Helper function that swaps 'missing' for np.nan and string intervals 
      for their interval type to assist with sorting'''

      if x == 'missing':
        return np.nan
      else:
        return portion.from_string(x, conv=float)

    df['sorting_var'] = df.iloc[:, 0].apply(sort_helper)
    return df.sort_values(by='sorting_var', na_position='first', 
                          ignore_index=True).drop(columns = ['sorting_var'])

  #If feature is categorical, we sort dataframe by Weight of Evidence
  else:
    return df.sort_values(by='WoE', ignore_index=True)

def calc_feature_metrics_test(feature_bin, target, feature_bin_test, is_feature_numeric=True):
    '''Generate statistics for grouped characteristic '''
    df_temp = pd.concat([feature_bin, target], axis=1)
    df_train = df_temp.groupby(df_temp.columns[0], as_index=False)\
                      .agg(bad_rate=(df_temp.columns.values[1], 'mean'))
    
    feat_series_test = feature_bin_test.value_counts(dropna=False)
    
    df_test = pd.DataFrame(feat_series_test).reset_index()
    df_test = df_test.rename(columns={'index': df_test.columns[1], df_test.columns[1]: 'n_obs'})
    
    df_test = pd.merge(df_test, df_train, on=df_test.columns[0], how='left')
    
    df_test['prop_n'] = df_test['n_obs'] / df_test['n_obs'].sum()
    df_test['n_good'] = df_test['n_obs'] * (1-df_test['bad_rate'])
    df_test['n_bad'] = df_test['n_obs'] * df_test['bad_rate']
    df_test['prop_n_good'] = df_test['n_good'] / df_test['n_good'].sum()
    df_test['prop_n_bad'] = df_test['n_bad'] / df_test['n_bad'].sum()
    df_test['WoE'] = np.log(df_test['prop_n_good'] / df_test['prop_n_bad'])
    
    #If the feature is numerical we sort by interval in ascending order. However,
    #we must first convert the 'missing' descriptor with np.nan, and get the 
    #interval from the string version as a new feature.
    if is_feature_numeric:
        def sort_helper(x):
            '''Helper function that swaps 'missing' for np.nan and string intervals 
            for their interval type to assist with sorting'''
            if x == 'missing':
                return np.nan
            else:
                return portion.from_string(x, conv=float)
            
        df_test['sorting_var'] = df_test.iloc[:, 0].apply(sort_helper)
        
        return df_test.sort_values(by='sorting_var', na_position='first', 
                          ignore_index=True).drop(columns = ['sorting_var'])
    #If feature is categorical, we sort dataframe by Weight of Evidence
    else:
        return df_test.sort_values(by='WoE', ignore_index=True)
    
    return df_test


def woe_bins(feature, target, bin_breaks, is_feature_numeric=True):
  '''Wrapper function to create bin metrics of feature'''
  
  if is_feature_numeric:
    feature_map = map_numerical_features(bin_breaks)
  else:
    feature_map = map_categorical_features(bin_breaks)

  feature_bin = transform_to_bin(feature, feature_map) 
  feature_bin_metrics = calc_feature_metrics(feature_bin=feature_bin, 
                                         target=target, 
                                         is_feature_numeric=is_feature_numeric)
  
  return feature_bin, feature_bin_metrics

def woe_bins_test(feature, target, bin_breaks, feature_test, is_feature_numeric=True):
    '''Wrapper function to create bin metrics of feature'''
    
    if is_feature_numeric:
        feature_map = map_numerical_features(bin_breaks)
    else:
        feature_map = map_categorical_features(bin_breaks)
        
    feature_bin = transform_to_bin(feature, feature_map)
    feature_bin_test = transform_to_bin(feature_test, feature_map)
    feature_bin_test_metrics = calc_feature_metrics_test(feature_bin=feature_bin, 
                                                         target=target,
                                                         feature_bin_test=feature_bin_test,
                                                         is_feature_numeric=is_feature_numeric)
    return feature_bin_test, feature_bin_test_metrics    