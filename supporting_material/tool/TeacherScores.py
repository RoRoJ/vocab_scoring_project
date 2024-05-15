
# coding: utf-8

# In[7]:


import pandas as pd

class TeacherScores:
    
    """
    This class maps the aggregate teacher difficulty scores for each vocabulary item onto a dataframe holding said items
    Attributes:
        scored_sw_tokens_dict: a dictionary which maps vocabulary items to aggregate teacher difficulty scores
    """
    
    #open text with plain text reader, return its tokenized sentences, and count of tokenized words
    def __init__(self, text):
        """
        The constructor for the class
        Params:
            text(string): specifies the sample text name to map teacher scores for
        """
        sheetname = text+'_responses'
        scored_sw_tokens_dict = pd.read_excel('files/teacher_scores.xlsx', sheet_name=sheetname, usecols='A,B', index_col=0, header=0).to_dict()
        self.scored_sw_tokens_dict = scored_sw_tokens_dict['Score']
        
        #deals with MWEs in the excel file by transforming them into tuples to match object format in TextItems
        teacher_scores_dict=[]
        for key in self.scored_sw_tokens_dict:
            if ' ' in key:
                new_key = key.split(' ')
                teacher_scores_dict.append(((tuple(new_key)), self.scored_sw_tokens_dict[key]))
            else:
                teacher_scores_dict.append((key, self.scored_sw_tokens_dict[key]))                 
        self.teacher_scores_dict= dict(teacher_scores_dict)
        
    def add_teacher_scores(self, table):
        """
        Method takes a pandas dataframe of vocabulary items and maps them to teacher scores
        Params: 
            table (pd dataframe): The dataframe of vocabulary items to look up teacher scores for
        """
        table['teacher_score']=''
        for i, row in table.iterrows():
            if row['word_in_text'] in self.teacher_scores_dict:
                table.at[i,'teacher_score'] = self.teacher_scores_dict[row['word_in_text']]
            else:
                table.at[i,'teacher_score'] =0
        return table

