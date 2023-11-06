import pandas as pd
import numpy as np
import copy


def to_intervals(risk_setting):
    '''Take a list of numbers and convert to tuples of intervals bounded by +/- infinity'''
    
    interval_list = []    
    config = sorted(risk_setting)
    
    config.insert(0, float('-inf'))
    config.append(float('inf'))
    
    for i in range(1, len(config)):
        interval = [config[i-1], config[i]]
        interval_list.append(interval)
    
    return interval_list

def make_grade_interval_map(risk_intervals, grade_letters):
    '''Create a dictionary that maps a grade to score interval'''
    
    grade_interval_map = {}

    descending_letters = sorted(grade_letters, reverse=True)
    grade_with_interval = zip(descending_letters, 
                              risk_intervals)

    for grade, interval in grade_with_interval:
        grade_interval_map[grade] = pd.Interval(left=interval[0], 
                                                right=interval[-1],
                                                closed='right')
        
    return grade_interval_map

def get_grade(score, grading_scheme):
    '''Get score grade'''
    
    for grade in grading_scheme:
        if score in grading_scheme[grade]:
            return grade
        
def grade_scores(scores, grading_scheme):
    '''Return a series of grades'''
    
    grades = scores.apply(get_grade, 
                          grading_scheme=grading_scheme)
    
    return grades

def apply_grade_filter(df, score_feature, risk_setting, 
                       grade_letters=['D', 'C', 'B', 'A'], grade_varname='grade'):
    '''Creates new dataframe that includes grades based on scores, 
    risk_setting, and grade_letters'''
    
    grade_df = df.copy(deep=True)
    
    scores = grade_df[score_feature]
    risk_intervals = to_intervals(risk_setting)
    grade_map = make_grade_interval_map(risk_intervals, grade_letters)
    grades = grade_scores(scores, grade_map)
    
    grade_df[grade_varname] = grades
    
    return grade_df