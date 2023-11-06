import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from src.features.compute_bin_metrics import woe_bins, map_categorical_features, map_numerical_features, assign_bin

def get_woe_scores(metrics_df):
    '''Broadcast feature name and get bin and WoE scores from metrics dataframe'''
    
    df = metrics_df.iloc[:, [0, 8]]
    dfc = df.copy()
    feature_name = df.columns[0][:-4]
    dfc.loc[:, 'feature'] = feature_name
    dfc = dfc.iloc[:, [2, 0, 1]]
    dfc = dfc.rename(columns={dfc.columns[1]: 'bin'})

    return dfc

def make_woe_card(df, target, mappings):    
    '''Transform modelling inputs to a WoE dataset'''
    
    feature_woe_list = []

    #Get WoE scores for each feature and add to list
    for feature in df.columns:
        truth_value = is_numeric_dtype(df[feature]) 
        
        #Get the metrics dataframe for the feature
        _, metrics_df = woe_bins(df[feature], 
                                 target, 
                                 mappings[feature], 
                                 is_feature_numeric=truth_value)
        
        #Select and re-order relevant columns
        feature_woe_scores = get_woe_scores(metrics_df)
        
        feature_woe_list.append(feature_woe_scores)
   
    woe_card = pd.concat(feature_woe_list, axis=0)
    
    return woe_card

def map_all_feat_bins_to_labels(feature_df, feature_breaks):
    '''Creates a dictionary of dictionaries that identifies each 
    feature, with their bin_labels and bin intervals/bin categories
    that can be searched
    
    Parameters
    ----------
    feature_df (pd.DataFrame): Dataframe of features
    feature_breaks (dict): Dictionary of bin breaks for each feature
    
    Return
    ------
    (dict) consisting of all bin labels and intervals/bin categories
    identified by the feature
    
    '''
    ref_dict = {}
    
    for feature in feature_df.columns:
        truth_value = is_numeric_dtype(feature_df[feature])
        
        if truth_value:
            ref_dict[feature] = map_numerical_features(feature_breaks[feature])
        
        else:
            ref_dict[feature] = map_categorical_features(feature_breaks[feature])
            
    return ref_dict

def create_credit_scorecard(feature_names,
                            model, woe_card, 
                            target_score=500, 
                            target_odds=50, 
                            pts_double_odds=20):
    '''Create a credit scorecard'''
    
    factor = pts_double_odds / np.log(2)
    offset = target_score - (factor * np.log(target_odds))
    
    feature_coef = pd.DataFrame({'feature': list(feature_names), 
                                   'coef': list(model.coef_[0])})
    
    feature_woe_coef = pd.merge(woe_card, feature_coef,
                              how='left', on='feature')
    
    credit_scorecard = feature_woe_coef.copy()
    n = len(feature_names)
    credit_scorecard.loc[:, 'points'] = -(credit_scorecard['coef'] \
                                         * credit_scorecard['WoE'] + model.intercept_/n)\
                                         * factor
    
    base = []
    base.insert(0, {'feature': 'basepoints', 'bin': np.nan, 'WoE': np.nan,
                    'coef': np.nan, 'points': round(offset, 0)})
    
    base_and_credit_scorecard = pd.concat([pd.DataFrame(base), 
                                           credit_scorecard],
                                          ignore_index=True)
    
    base_and_credit_scorecard[['points']] = base_and_credit_scorecard[['points']].round()
    
 
    return base_and_credit_scorecard

def compute_credit_score(df, scorecard, maps):
    '''Calculate the credit score of candidates'''
    new_df = df.copy()
    new_df = new_df.reset_index(drop=True)
    
    row_scores = []
    
    #Iterate through each row
    for idx, series in new_df.iterrows():
        feature_bin = [] #Stores the feature name and bin names for each value in the row
        
        #Iterate through each feature and its value for the row
        for name, value in series.items():
            bin_name = assign_bin(value, maps[name]) #Search the dictionary for 
                                                     #the value and return the bin name
                
            identifier = (name, bin_name) #Combine the feature_name and bin_name as a tuple
            
            feature_bin.append(identifier) #Add the feature / bin name combo to a list, 
                                           #representing value for the row
        
        #For each row, convert the feature / bin name combo values from a list
        #of tuples to a dataframe. Change the names of the columsn to feature and bin.
        
        #The scorecard dataframe also has column names identified as feature and bin.
        
        #Merging the row dataframe with the scorecard dataframe on the feature and bin
        #columns selects only relevant rows in the the scorecard.
        
        #The values of the score column can then be summed up to yield the credit score
        
        row = pd.DataFrame(feature_bin)
        row = row.rename(columns={row.columns[0]:'feature', 
                                 row.columns[1]: 'bin'})
        
        row_score_df = pd.merge(row, scorecard, on=['feature', 'bin'], how='left')
        
        row_score = round(row_score_df['points'].sum() + scorecard.loc[0, 'points'].astype('int'), 0)
        
        #Store the points for each row in a list
        row_scores.append(row_score)
        
    new_df.loc[:, 'score'] = row_scores
        
    return new_df

def convert_scorecard_to_df(scorecard_dict):
    scorecard_list = []
    for key, value in scorecard_dict.items():
        scorecard_list.append(scorecard_dict[key])
    
    scorecard_list = [df.set_index(['variable', 'bin', 'points']) 
                      for df in scorecard_list]
        
    scorecard_df = pd.concat(scorecard_list, axis=1)
    
    return scorecard_df.reset_index()