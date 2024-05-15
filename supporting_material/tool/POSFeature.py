
# coding: utf-8

# In[29]:


import pandas as pd

class POSFeature:
    
    """
    This class computes POS scores for vocabulary items

    Attributes:
        pos_score_dict: a dictionary which maps POS tags to scores
    """
    
    def __init__(self):
        """
        The constructor for the FrequencyFeature class.
        """
        self.pos_score_dict=pd.read_excel('files/posscores.xlsx', header=0,usecols=[0,1],index_col='POS').to_dict()['Normalised Score']
       
    @staticmethod
    def get_pos_score_table(text_maptagged_items, pos_score_dict):
        """
        Assigns normalised POS scores for vocab items using the pos score dictionary

        Params:
            text maptagged_items(list): list of broad pos tagged vocab items
            pos_score_dict (dictionary): dictionary which maps broad pos tags to scores
        
        Returns: pandas dataframe of vocabulary items and their POS tag scores
        """
        base_table = pd.DataFrame(text_maptagged_items)
        base_table.columns=['vocab_item','POS_Mapped']
        base_table['POS_Score']=''

        for index, row in base_table.iterrows():
            row['POS_Score']=pos_score_dict[row['POS_Mapped']]
                
        return base_table

