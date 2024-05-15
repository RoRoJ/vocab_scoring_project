
# coding: utf-8

# In[8]:


import pandas as pd
from TextItems import TextItems
from FrequencyFeature import FrequencyFeature
from L1SimilarityFeature import L1SimilarityFeature
from LengthFeature import LengthFeature
from POSFeature import POSFeature
from POSFrequencyFeature import POSFrequencyFeature


# In[9]:


class TextScorer:
    
    """
    This class represents a pandas dataframe holding vocabulary items from a text and their feature scores

    Attributes:
        text_sentences: list of sentence tokens from the text
        text_untagged_items: list of vocabulary items from the text
        text_tagged_items: list of NLTK POS-tagged vocabulary items from the text
        text_maptagged_items: list of broader POS-tagged vocabulary items from the text
        master_table: pandas dataframe holding vocab items, tags and feature scores
    """
    
    def __init__(self, textfilepath):
        """
        The constructor for the TextScorer class.
        
        Parameters:
           textfilepath (string): The filepath for the text file  
        """
        #Calls the TextItems class to create vocabulary items from the text
        self.text_sentences = TextItems(textfilepath).text_sentences
        self.text_untagged_items = TextItems.get_text_items(self.text_sentences, return_option='untagged', case_option='all_lower', exclude_nondict_option='yes')
        self.text_tagged_items = TextItems.get_text_items(self.text_sentences, return_option='tagged', case_option='all_lower', exclude_nondict_option='yes')
        self.text_maptagged_items = TextItems.map_text_tags(self.text_tagged_items)
        
        #create table to hold text items, postags and mapped postags
        table = pd.merge(pd.DataFrame(self.text_tagged_items), pd.DataFrame(self.text_maptagged_items), left_index=True, right_index=True)
        table.drop(['0_y'], axis=1, inplace=True)
        table.columns=['word_in_text', 'pos_tag', 'mapped_tag']
        self.master_table = table
    
    def add_freq_scores(self):
        """
        Method to call FrequencyFeature class and acquire Frequency scores
        Updates the class attribute master_table which holds all frequency scores, and returns it
        """
        self.freqdistdict=FrequencyFeature().new_freqdist_dict
        freq_scored_text=FrequencyFeature.get_freqscore_table(self.text_untagged_items, self.freqdistdict)
        updated_table = self.master_table.join(freq_scored_text['freq_normalised_score'])
        self.master_table = updated_table.rename(columns = {'freq_normalised_score':'freq_score'})
        return self.master_table
   
    def add_l1sim_scores(self):
        """
        Method to call L1SimilarityFeature class and acquire L1Similarity scores
        Updates the class attribute master_table which holds all frequency scores, and returns it
        """
        self.frenchdictdict_simple=L1SimilarityFeature().french_dict_simple
        self.frenchdictdict_pos=L1SimilarityFeature().french_dict_pos
        
        french_scored_text=L1SimilarityFeature.get_frenchscore_table_pos(self.text_maptagged_items, self.frenchdictdict_pos, self.frenchdictdict_simple)
        self.french_detail_table = french_scored_text
        
        self.master_table = self.master_table.join(french_scored_text['french_score'])
        self.master_table = self.master_table.rename(columns = {'french_score':'l1sim_score'})
        return self.master_table
    
    def add_length_scores(self):
        """
        Method to call LengthFeature class and acquire Length scores
        Updates the class attribute master_table which holds all frequency scores, and returns it
        """
        self.lengthdict=LengthFeature().length_score_dict
        length_scored_text=LengthFeature.get_lengthscore_table(self.text_untagged_items, self.lengthdict)
        self.master_table = self.master_table.join(length_scored_text['length_score'])
        return self.master_table
    
    def add_pos_scores(self):
        """
        Method to call POSFeature class and acquire POS scores
        Updates the class attribute master_table which holds all frequency scores, and returns it
        """
        self.posdict=POSFeature().pos_score_dict
        pos_scored_text=POSFeature.get_pos_score_table(self.text_maptagged_items, self.posdict)
        self.master_table = self.master_table.join(pos_scored_text['POS_Score'])
        return self.master_table
    
    def add_posfreq_scores(self):
        """
        Method to call POSFeature class and acquire POS scores
        Updates the class attribute master_table which holds all frequency scores, and returns it
        """
        self.freq_dist_dict_postagged=POSFrequencyFeature().bnc_freqdist_pos_dict
        pos_freqscored_text=POSFrequencyFeature.get_pos_freqscore_table(self.text_tagged_items, self.freq_dist_dict_postagged)        
        self.master_table = self.master_table.join(pos_freqscored_text['freq_pos_normalised_score'])
        return self.master_table
    
    def reset_master_table(self):
        """
        Method to call reset the master table by overwriting it with the constructor version
        """
        table = pd.merge(pd.DataFrame(self.text_tagged_items), pd.DataFrame(self.text_maptagged_items), left_index=True, right_index=True)
        table.drop(['0_y'], axis=1, inplace=True)
        table.columns=['word_in_text', 'pos_tag', 'mapped_tag']
        self.master_table = table
        return self.master_table
    
    def add_all_scores(self):
        """
        Method to call all the other add_xx_scores methods and add scores all in one go
        Updates the class attribute master_table and returns it
        """
        TextScorer.add_freq_scores(self)
        TextScorer.add_l1sim_scores(self)
        TextScorer.add_length_scores(self)
        TextScorer.add_pos_scores(self)
        TextScorer.add_posfreq_scores(self)
        return self.master_table

