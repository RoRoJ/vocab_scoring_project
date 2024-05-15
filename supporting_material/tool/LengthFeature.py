
# coding: utf-8

# In[64]:


import numpy as np
import pandas as pd

class LengthFeature:
    
    """
    This class computes Length scores for vocabulary items

    Attributes:
        length_score_dict: a dictionary which maps vocabulary items to length scores
    """
    
    def __init__(self):
        """
        The constructor initialises the length score dictionary
        """
        self.length_score_dict=pd.read_excel('files/lengthscores.xlsx', header=None, index_col=0).to_dict()[1]
       
    @staticmethod
    def get_lengthscore_table(text_untagged_items, length_score_dict):
        """
        Assigns length scores for vocab items using the length dictionary

        Params:
            text_untagged_items (list): list of vocabulary items
            length_score_dict (dictionary): dictionary which maps lengths to scores
        
        Returns: pandas dataframe of vocabulary items and their length scores
        """
        base_table = pd.DataFrame(text_untagged_items)
        base_table.columns=['vocab_item']
        base_table['length']=''
        base_table['length_score']=''
        for index, row in base_table.iterrows():
            if type(row['vocab_item'])!=tuple:
                row['length'] = len(row['vocab_item'])
            elif type(row['vocab_item'])==tuple:
                length=0
                for item in row['vocab_item']:
                    length+=len(item)
                row['length'] = length
            if row['length']>91:
                row['length_score']=1.0
            else:
                row['length_score']=length_score_dict[row['length']]
                
        return base_table

