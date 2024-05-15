
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
from sklearn import preprocessing
import pickle


class POSFrequencyFeature:
    
    """
    This class computes Frequency scores for vocabulary items incorporating POS tags into the frequency distr & score

    Attributes:
        bnc_freqdist_pos_dict: a dictionary which maps pos-tagged vocabulary items to pos-tagged frequencies in the BNC
    """
    
    #open text with plain text reader, return its tokenized sentences, and count of tokenized words
    def __init__(self):
        """
        The constructor for the POSFrequencyFeature class.
        """
        with open(r'C:/Users/rowena/Documents/MSC/Project/obj/combined_tagged_bnc_freqdict_lower.pkl', 'rb') as f:
            bnc_freqdist_pos_dict=pickle.load(f)
        self.bnc_freqdist_pos_dict=bnc_freqdist_pos_dict      

    @staticmethod
    def get_pos_freqscore_table(text_tagged_items, freq_dist_dict):
        """
        Calculates normalised Frequency  scores for vocab items using the POS-taggedfrequency dictionary

        Params:
            text_tagged_items(list): list of vocabulary items with their NLTK POS tags
            freq_dist_dict: dictionary which maps vocabulary items with POS tags to frequencies in the BNC
        
        Returns: pandas dataframe of vocabulary items and their normalised POS frequency scores
        """
        base_table = pd.DataFrame(text_tagged_items)
        base_table['tuple_tagged_wordintext']=''
        for index, row in base_table.iterrows():
            row['tuple_tagged_wordintext']=(row[0], row[1])
            
        base_table['freq']=''
        for index, row in base_table.iterrows():
            if row['tuple_tagged_wordintext'] in freq_dist_dict:
                row['freq'] = freq_dist_dict[row['tuple_tagged_wordintext']]
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
        base_table['freq_pos_normalised_score'] = (1-x_scaled)
        
        base_table.rename(columns={0:'word_in_text', 1:'POS_tag'}, inplace=True)
        base_table=base_table[['word_in_text', 'POS_tag','tuple_tagged_wordintext', 'freq_pos_normalised_score']]
        
        return base_table

