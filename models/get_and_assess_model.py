from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_model_and_performance(model, X_train, X_test, y_train, y_test):
    '''Fit a model to data and get its AUC performance
    
    Parameters
    ----------
    model (base_estimator): A model architecture with paramters
    X_train (pandas dataframe): Features of training set
    X_test (pandas dataframe): Features of test set
    y_train (pandas series): Target of training set
    y_test (pandas series): Target of test set
    
    Return
    ------
    model: Pre-trained model
    auc_scores train (pandas series): List of cross-validation AUC scores of the training set
    auc_score test )pandas series: AUC score for the test set
    '''
    
    base_estimator = clone(model)
    
    model.fit(X_train, y_train)
    y_train_pred_proba = model.predict_proba(X_train)[:,1]
    y_test_pred_proba = model.predict_proba(X_test)[:,1]
    
    auc_scores = cross_val_score(base_estimator, X_train, y_train, cv=4, scoring='roc_auc')
    
    test_perf = roc_auc_score(y_test, y_test_pred_proba)
    
    return (model, auc_scores, test_perf, y_test_pred_proba)

def make_gains_table(df,
                     label_name,
                     edges=[float('-inf'), 340, 385, 435, float('inf')],
                     with_approved_col=False):
    '''Segment values into bins, counts and proportions'''
    def cumulative_sum(df, column_name):
            return  df.loc[::-1, column_name].cumsum()[::-1]
        
    def cumulative_perc(df, column_name, base_column_name='count'):
            return ((df.loc[::-1, column_name].cumsum()
                     / df[base_column_name].sum()) 
                    * 100).round(2)[::-1]
    
    score_df = df.copy()

    score_df['risk_bin'] = pd.cut(x=df['score'], 
                                  bins=edges, 
                                  right=True)
    
    if with_approved_col:
        bin_aggregates = score_df.groupby('risk_bin', as_index=False).agg(
            count=('score', 'count'), 
            bads=(label_name, 'sum'),
            count_approved=('approved', 'sum'))

        bin_aggregates['count_perc'] = round((bin_aggregates['count'] 
                                             / bin_aggregates['count'].sum()) * 100, 2)
        bin_aggregates['goods'] = bin_aggregates['count'] - bin_aggregates['bads']
        bin_aggregates['count_rejected'] = bin_aggregates['count'] - bin_aggregates['count_approved']

        bin_aggregates['interval_goods_perc'] = round((bin_aggregates['goods'] 
                                                       / bin_aggregates['count']) * 100, 2)
        bin_aggregates['interval_bads_perc'] = round((bin_aggregates['bads'] 
                                                     / bin_aggregates['count']) * 100, 2)
        bin_aggregates['interval_approval_perc'] = ((bin_aggregates['count_approved']
                                                     / bin_aggregates['count']) * 100).round(2)

        bin_aggregates['goods_distr_perc'] = ((bin_aggregates['goods'] 
                                               / bin_aggregates['goods'].sum())
                                              * 100).round(2)

        bin_aggregates['bads_distr_perc'] = ((bin_aggregates['bads'] 
                                              / bin_aggregates['bads'].sum())
                                              * 100).round(2)


        bin_aggregates['cum_count'] = cumulative_sum(bin_aggregates, 'count')
        bin_aggregates['cum_goods'] = cumulative_sum(bin_aggregates, 'goods')
        bin_aggregates['cum_bads'] = cumulative_sum(bin_aggregates, 'bads')
        bin_aggregates['cum_count_approved'] = cumulative_sum(bin_aggregates, 'count_approved')
        bin_aggregates['cum_count_rejected'] = cumulative_sum(bin_aggregates, 'count_rejected')

        bin_aggregates['cum_count_perc'] = cumulative_perc(bin_aggregates, 'count')
        bin_aggregates['cum_goods_perc'] = cumulative_perc(bin_aggregates, 'goods')
        bin_aggregates['cum_bads_perc'] = cumulative_perc(bin_aggregates, 'bads')
        bin_aggregates['cum_approved_perc'] = cumulative_perc(bin_aggregates, 'count_approved')
        bin_aggregates['cum_rejected_perc'] = cumulative_perc(bin_aggregates, 'count_rejected')



        final_df = bin_aggregates[['risk_bin', 
                                   'count', 'count_perc', 
                                   'cum_count', 'cum_count_perc',
                                   'goods', 'goods_distr_perc', 
                                   'interval_goods_perc',
                                   'cum_goods', 'cum_goods_perc',
                                   'bads', 'bads_distr_perc',
                                   'interval_bads_perc',
                                   'cum_bads', 'cum_bads_perc',
                                   'count_approved', 'interval_approval_perc',
                                   'cum_count_approved', 'cum_approved_perc',
                                   'count_rejected', 'cum_count_rejected', 'cum_rejected_perc',
                                  ]]
    
        return final_df
    else:
        bin_aggregates = score_df.groupby('risk_bin', as_index=False).agg(count=('score', 'count'), 
                                                                           bads=(label_name, 'sum'))
        bin_aggregates['count_perc'] = round((bin_aggregates['count'] 
                                             / bin_aggregates['count'].sum()) * 100, 2)
        bin_aggregates['goods'] = bin_aggregates['count'] - bin_aggregates['bads']

        bin_aggregates['interval_goods_perc'] = round((bin_aggregates['goods'] 
                                                       / bin_aggregates['count']) * 100, 2)
        bin_aggregates['interval_bads_perc'] = round((bin_aggregates['bads'] 
                                                     / bin_aggregates['count']) * 100, 2)

        bin_aggregates['goods_distr_perc'] = ((bin_aggregates['goods'] 
                                               / bin_aggregates['goods'].sum())
                                              * 100).round(2)

        bin_aggregates['bads_distr_perc'] = ((bin_aggregates['bads'] 
                                              / bin_aggregates['bads'].sum())
                                              * 100).round(2)



        bin_aggregates['cum_count'] = cumulative_sum(bin_aggregates, 'count')
        bin_aggregates['cum_goods'] = cumulative_sum(bin_aggregates, 'goods')
        bin_aggregates['cum_bads'] = cumulative_sum(bin_aggregates, 'bads')

        bin_aggregates['cum_count_perc'] = cumulative_perc(bin_aggregates, 'count')
        bin_aggregates['cum_goods_perc'] = cumulative_perc(bin_aggregates, 'goods')
        bin_aggregates['cum_bads_perc'] = cumulative_perc(bin_aggregates, 'bads')

        final_df = bin_aggregates[['risk_bin', 
                                   'count', 'count_perc', 
                                   'cum_count', 'cum_count_perc',
                                   'goods', 'goods_distr_perc', 
                                   'interval_goods_perc',
                                   'cum_goods', 'cum_goods_perc',
                                   'bads', 'bads_distr_perc',
                                   'interval_bads_perc',
                                   'cum_bads', 'cum_bads_perc'
                                  ]]
    
        return final_df
    
def compare_distr(agg1, agg2, series_name, distr_names=['distr1', 'distr2'], rotation=45, figsize=(10,6)):
    
    distr1, distr2 = distr_names
    
    agg1['distribution'] = distr1
    agg2['distribution'] = distr2
    compare_df = pd.concat([agg1, agg2])
    
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(f'{series_name} for {distr1} and {distr2}')
    
    d = sns.barplot(data=compare_df, x='risk_bin', y=series_name, hue='distribution')

    #Label bars
    d.bar_label(d.containers[0], d.containers[0].datavalues)
    d.bar_label(d.containers[1], d.containers[1].datavalues)

    d.set_xlabel(None)
    d.set_ylabel(None)
    d.set_yticks([])
    
    plt.legend(loc="upper center",prop = {'size': 12}, bbox_to_anchor=(1.2, 1))
    
    plt.xticks(rotation=rotation)
    
    plt.show();
    
def score_to_good_prob(score, pdo=50, target_score=600, target_good_odds=19/1):
    '''Convert score to good probability'''

    factor = pdo/np.log(2)
    offset = target_score - (factor * np.log(target_good_odds))
    odds = np.exp((score - offset)/factor)
    good_prob = odds / (odds + 1)

    return good_prob.round(6)

def good_prob_to_score(prob, pdo=50, target_score=600, target_good_odds=19/1):
    '''Convert score to good probability'''
    
    factor = pdo/np.log(2)
    offset = target_score - (factor * np.log(target_good_odds))
    odds = prob / (1-prob)
    score = offset + (np.log(odds) * factor)
    
    return int(score)

def compare_num_distrs(distr1, distr2, varname, distr_names=['distr1', 'distr2']):
    
    distr1_name, distr2_name = distr_names
    
    
    fig, ax = plt.subplots(figsize=(13, 6))
    
    plt.title(f'{varname} distributions for {distr1_name} and {distr2_name} data')
    plt.ylabel('proportion')

    g1=sns.histplot(data=distr1, x=varname, stat='density', alpha=0.1)
    g2=sns.histplot(data=distr2, x=varname, stat='density',alpha=0.1, color='magenta')
    
    fig.legend(labels=[f'{distr1_name}', f'{distr2_name}'])
    
    plt.show();
    

def combine_data(accept_df, reject_df, variables, performance_labels = ['loan_id', 'approved', 'dpd_15', 'score']):
    
    accept_df['approved'] = 1
    
    reject_df['loan_id'] = np.array([f'r_{i}' for i in range(reject_df.shape[0])])

    
    relevant_columns = variables + performance_labels
    
    
    combined = pd.concat([accept_df[relevant_columns], reject_df[relevant_columns]]).reset_index(drop=True)
    
    return combined


def compare_totals(agg1, agg2, series_name, distr_names=['distr1', 'distr2']):
    
    distr1, distr2 = distr_names
    
    total_agg1 = round(agg1[series_name][0], 4)
    total_agg2 = round(agg2[series_name][0], 4)
    
    print(f'{series_name} for {distr1}: {total_agg1} ')
    print(f'{series_name} for {distr2}: {total_agg2} ')
    
def get_payments(loans_df, fees_payments_df, 
                   edges=[float('-inf'), 340, 385, 435, float('inf')],
                   interest_basis_points = [0, 1600, 1300, 1000],
                   new_scorecard=False):
    
    
    basis_point_factor = 10_000

    loans = loans_df.copy(deep=True)
    fees_payments = fees_payments_df.copy(deep=True)
    
    loans['loan_id'] =  loans['loan_id'].astype(str)
    fees_payments['loan_id'] = fees_payments['loan_id'].astype(str)
    
    df = loans.merge(fees_payments, on='loan_id', how='left')
    
    df['interest'] = df['principal'] \
                   * (df['monthly_fee_basis_points'] / basis_point_factor)
    
    fee_columns = ['principal', 'total_paid', 'monthly_fee_basis_points',
                   'daily_penalty_basis_points', 'fixed_fee_amount']
    
    payment_columns = ['amount_paid', 
                       'principal_paid',
                       'interest_paid',
                       'fees_paid',
                       'penalties_paid', 
                       'unallocated_paid']
    
    dpd_payment_columns = ['amount_paid_15_dpd', 
                           'principal_paid_15_dpd',
                           'interest_paid_15_dpd',
                           'fees_paid_15_dpd',
                           'penalties_paid_15_dpd', 
                           'unallocated_paid_15_dpd']
    
    #Need to determine the new loan fees and payments given new scores
    if new_scorecard:
        #Loan_ids with scores in the first bin will not be given a loan - hence zeros
        #for their fee and payment details
        df.loc[df['score'].between(edges[0], edges[1], inclusive='right'), 
               fee_columns + payment_columns + dpd_payment_columns] = 0
    
        #For other loans outside the first bin we assume that the principal, penalties 
        #and fixed fee amounts remain the same even with new scores. 
        #However, we change the interest rates to match their new score and interest rate regime
        for i in range(len(interest_basis_points)):
            df.loc[df['score'].between(edges[i], edges[i+1], inclusive='right'), 
                         'monthly_fee_basis_points'] = interest_basis_points[i]
            
            df['interest'] = df['principal'] \
                           * (df['monthly_fee_basis_points'] / basis_point_factor)

       
        
        #We assume the same repayment behavior for defaulters and non-defaulters
        #This means that defaulters would only pay what they eventually paid and we don't change anything
        #and non-defaulters would pay the new interest even though the interest rate may have changed
        
        #Assuming non-defaulters pay their respective interests in full regardless of rate,
        #this means the dpd and regular amounts are the sampe
        df.loc[df['dpd_15'] == 0, 
               ['interest_paid', 
                'interest_paid_15_dpd']] =  df['principal'] \
                                         * (df['monthly_fee_basis_points'] / basis_point_factor)
        
        #We assume that the new amount paid, total paid and amount paid dpd 
        #reflects the interest rate change for non-defaulters
        df.loc[df['dpd_15'] == 0, 
                ['amount_paid', 
                 'amount_paid_15_dpd',
                 'total_paid']]  = df['principal_paid'] \
                                 + df['interest_paid'] \
                                 + df['fees_paid'] \
                                 + df['penalties_paid'] \
                                 + df['unallocated_paid']
       
    
    
    relevant_cols = ['loan_id', 'approved','dpd_15', 'score'] \
                  + fee_columns + ['interest'] + payment_columns + dpd_payment_columns
    
    return df[relevant_cols]


# def get_bin_yields(df, edges=[float('-inf'), 340, 385, 435, float('inf')]):
    
#     millions_factor = 1000_000

#     df['risk_bin'] = pd.cut(x=df['score'], 
#                             bins=edges, 
#                             right=True)

#     bins = df.groupby('risk_bin', as_index=False)\
#              .agg(total_principal_owed_millions=('principal', 'sum'),
#                   total_principal_paid_15_dpd_millions=('principal_paid_15_dpd', 'sum'),
#                   total_principal_paid_millions=('principal_paid', 'sum'),
#                   total_interest_owed_millions=('interest', 'sum'),
#                   total_interest_paid_15_dpd_millions=('interest_paid_15_dpd', 'sum'),
#                   total_interest_paid_millions=('interest_paid', 'sum'))
    
#     bins = bins.assign(total_owed_millions = bins['total_principal_owed_millions'] \
#                                            + bins['total_interest_owed_millions'],
#                        total_paid_15_dpd_millions = bins['total_principal_paid_15_dpd_millions'] \
#                                                   + bins['total_interest_paid_15_dpd_millions'],
#                       total_paid_millions = bins['total_principal_paid_millions'] \
#                                                   + bins['total_interest_paid_millions'])
    
#     bins = bins.assign(yield_15_dpd = (bins['total_paid_15_dpd_millions']\
#                                     / bins['total_principal_owed_millions']).round(4),
#                        all_yield = (bins['total_paid_millions']\
#                                     / bins['total_principal_owed_millions']).round(4),
#                        net_yield_15_dpd_millions = bins['total_paid_15_dpd_millions'] \
#                                                  - bins['total_principal_owed_millions'],
                      
#                        net_yield_millions = bins['total_paid_millions'] \
#                                                  - bins['total_principal_owed_millions'])
    
#     million_columns = ['total_principal_owed_millions', 'total_principal_paid_15_dpd_millions',
#                        'total_principal_paid_millions',
#                        'total_interest_owed_millions','total_interest_paid_15_dpd_millions', 
#                        'total_interest_paid_millions',
#                        'total_owed_millions','total_paid_15_dpd_millions', 'total_paid_millions',
#                        'net_yield_15_dpd_millions', 'net_yield_millions']
    
#     bins[million_columns] = (bins[million_columns] / millions_factor).round(2)
    
    
#     bins = bins.assign(total_yield_15_dpd=(bins['total_paid_15_dpd_millions'].sum()\
#                                          / bins['total_principal_owed_millions'].sum()).round(4),
#                        total_yield=(bins['total_paid_millions'].sum()\
#                                          / bins['total_principal_owed_millions'].sum()).round(4),
#                        total_net_yield_15_dpd_millions=bins['total_paid_15_dpd_millions'].sum()
#                                                       - bins['total_principal_owed_millions'].sum(),
#                       total_net_yield_millions=bins['total_paid_millions'].sum()
#                                                       - bins['total_principal_owed_millions'].sum())
    
#     return bins

def get_bin_yields(df, edges=[float('-inf'), 340, 385, 435, float('inf')], 
                   all_payments=True):
    
    millions_factor = 1000_000

    df['risk_bin'] = pd.cut(x=df['score'], 
                            bins=edges, 
                            right=True)

    bins = df.groupby('risk_bin', as_index=False)\
             .agg(total_principal_owed_millions=('principal', 'sum'),
                  total_principal_paid_15_dpd_millions=('principal_paid_15_dpd', 'sum'),
                  total_principal_paid_millions=('principal_paid', 'sum'),
                  total_interest_owed_millions=('interest', 'sum'),
                  total_interest_paid_15_dpd_millions=('interest_paid_15_dpd', 'sum'),
                  total_interest_paid_millions=('interest_paid', 'sum'))
    
    bins = bins.assign(total_owed_millions = bins['total_principal_owed_millions'] \
                                           + bins['total_interest_owed_millions'],
                       total_paid_15_dpd_millions = bins['total_principal_paid_15_dpd_millions'] \
                                                  + bins['total_interest_paid_15_dpd_millions'],
                      total_paid_millions = bins['total_principal_paid_millions'] \
                                                  + bins['total_interest_paid_millions'])
    
    bins = bins.assign(yield_15_dpd = (bins['total_paid_15_dpd_millions']\
                                    / bins['total_principal_owed_millions']).round(4),
                       
                       bin_yield = (bins['total_paid_millions']\
                                    / bins['total_principal_owed_millions']).round(4),
                       
                       net_yield_15_dpd_millions = bins['total_paid_15_dpd_millions'] \
                                                 - bins['total_principal_owed_millions'],
                      
                       net_yield_millions = bins['total_paid_millions'] \
                                                 - bins['total_principal_owed_millions'],
                       
                       recovery_rate_15_dpd = (bins['total_paid_15_dpd_millions']\
                                               / bins['total_owed_millions']).round(4),
                      
                       recovery_rate = (bins['total_paid_millions']\
                                        / bins['total_owed_millions']).round(4))
    
    million_columns = ['total_principal_owed_millions', 'total_principal_paid_15_dpd_millions',
                       'total_principal_paid_millions',
                       'total_interest_owed_millions','total_interest_paid_15_dpd_millions', 
                       'total_interest_paid_millions',
                       'total_owed_millions','total_paid_15_dpd_millions', 'total_paid_millions',
                       'net_yield_15_dpd_millions', 'net_yield_millions']
    
    bins[million_columns] = (bins[million_columns] / millions_factor).round(2)
    
    
    bins = bins.assign(total_yield_15_dpd=(bins['total_paid_15_dpd_millions'].sum()\
                                         / bins['total_principal_owed_millions'].sum()).round(4),
                       total_yield=(bins['total_paid_millions'].sum()\
                                         / bins['total_principal_owed_millions'].sum()).round(4),
                       total_net_yield_15_dpd_millions=bins['total_paid_15_dpd_millions'].sum()
                                                      - bins['total_principal_owed_millions'].sum(),
                      total_net_yield_millions=bins['total_paid_millions'].sum()
                                                      - bins['total_principal_owed_millions'].sum())
    
    if all_payments:
        relevant_columns = [column for column in bins.columns if '15' not in column.split('_')]
          
    else:
        relevant_columns = ['risk_bin'] + [column for column in bins.columns if 'owed' in column.split('_') 
                            or '15' in column.split('_')]
    
    return bins.loc[:, relevant_columns]
