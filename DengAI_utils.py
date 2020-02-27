#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:15:21 2020

@author: arodriguez
"""

def season(input_data):
        input_data = int(input_data)
        if (input_data >=1) & (input_data <=3): # Winter        
            output = 4;        
        if (input_data >=4) & (input_data <=6): # Spring        
            output = 1;
        if (input_data >=7) & (input_data <=9): # Summer        
            output = 2;
        if (input_data >=10) & (input_data <=12): # Autumn/Fall        
            output = 3;
        return output
    
    
    
    
def context_extractor(data_structure):
    alt_data_structure = data_structure[:];
    
    col_names = [i for i in data_structure.columns  if i not in ['total_cases','total_cases_LOG','diff','pos_neg']]
    lag = 1;
    col_names_lag = [i+'_'+str(lag) for i in col_names]
    
    alt_data_structure[col_names_lag] = city_data[col_names]
    alt_data_structure['weekofyear_1'] = alt_data_structure['weekofyear_1'].shift(1)
            
    return alt_data_structure



# We study each feature independently and how it relates with the rest
def compute_correlation(city_input_data):
    correlation = city_input_data.corr()
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    
    fig, ax = plt.subplots(figsize=(15,15)) 
    ax = sns.heatmap(
        correlation, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True,
        ax = ax
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    );
    plt.title('Correlation matrix')
    return correlation