import pandas as pd
import numpy as np

def make_policy_data(grade_letters, grade_varname,
                     policy_per_grade, policy_varname):
    '''Create dataset that maps grades to policy per grade'''
    
    grade_with_policy = zip(grade_letters, policy_per_grade)
    
    grade_policy_df = pd.DataFrame(grade_with_policy, 
                                   columns=[grade_varname, policy_varname])
    
    return grade_policy_df


def apply_policy_filter(df, grade_letters, grade_varname,
                        policy_per_grade, policy_varname,
                        suffixes=['_old', '_new']):
    '''Create new dataframe that includes policy implications'''
    
    grade_policy_df = make_policy_data(grade_letters, grade_varname,
                                       policy_per_grade, policy_varname)
    
    df_with_policy = pd.merge(df, grade_policy_df,
                              on=grade_varname,
                              how='left',
                              suffixes=suffixes)
   
    return df_with_policy