
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn import preprocessing
import re
import pickle

class FrequencyFeature:
    
    """
    This class computes Frequency scores for vocabulary items

    Attributes:
        new_freqdist_dict: a dictionary which maps vocabulary items to frequencies in the BNC
    """
    
    def __init__(self):
        """
        The constructor for the FrequencyFeature class.
        """
        with open(r'obj/combined_bnc_freqdist_dict_lower.pkl', 'rb') as f:
            bnc_freqdist_dict=pickle.load(f)
        self.new_freqdist_dict=bnc_freqdist_dict
            

    @staticmethod
    def get_freqscore_table(text_untagged_items, freq_dist_dict):
        """
        Calculates normalised Frequency  scores for vocab items using the frequency dictionary

        Params:
            text_untagged_items(list): list of vocabulary items
            freq_dist_dict: dictionary which maps vocabulary items to frequencies in the BNC
        
        Returns: pandas dataframe of vocabulary items and their normalised frequency scores
        """
        base_table = pd.DataFrame(text_untagged_items)
        base_table.columns=['word_in_text']
        base_table['freq']=''
        for index, row in base_table.iterrows():
            if row['word_in_text'] in freq_dist_dict:
                row['freq'] = freq_dist_dict[row['word_in_text']]
            else:
                row['freq'] = 0
        base_table['freq_smoothed']=base_table['freq'].astype(float)+1
        
        total_dict_instances=0
        for entry in freq_dist_dict:
            total_dict_instances+=freq_dist_dict[entry]
        total_instances_smoothed=total_dict_instances+(len(base_table))
        
        base_table['freq_proportion']=base_table['freq_smoothed'] / total_instances_smoothed
        base_table['freq_log']=np.log(base_table['freq_proportion'])
        x = base_table['freq_log'].values.astype(float) #returns a numpy array
        x = x.reshape(-1,1)
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        base_table['freq_normalised_score'] = (1-x_scaled)
        base_table=base_table[['word_in_text', 'freq_normalised_score']]
        return base_table

