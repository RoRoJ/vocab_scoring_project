
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import nltk
import re
import pickle

class L1SimilarityFeature:
    
    """
    This class computes L1 Similarity Feature scores for vocabulary items
    Currently configured to French only
    Other languages possible with the addition of more dictionaries

    Attributes:
        french_dict_simple: a dictionary which maps english vocabulary items to french translations
                            and their Levenshtein scores
        french_dict_pos: a more sophisticated version of the above, incorporating pos tags into the mapping
    """
    
    #open text with plain text reader, return its tokenized sentences, and count of tokenized words
    def __init__(self):
        
        """
        The constructor for the class. Initiates the dictionary objects.
        """
        
        #new, simple dictionary (parentheses text removed)
        with open(r'C:obj/french_dict_simple.pkl', 'rb') as f:
            french_dict=pickle.load(f)
        french_dict_lower=[]
        for key in french_dict:
            if type(key)!=tuple:
                french_dict_lower.append((key.lower(), french_dict[key]))
            if type(key)==tuple:
                lower_tuple=[]
                for component in key:
                    lower_tuple.append(component.lower())
                french_dict_lower.append((tuple(lower_tuple), french_dict[key]))
        self.french_dict_simple=dict(french_dict_lower)
        
        #new, POS complex dictionary (parentheses text removed)
        with open(r'C:obj/french_dict_pos.pkl', 'rb') as f:
            french_dict=pickle.load(f)  
        
        tag_dict=pd.read_excel('files/tag_mapping_forfrenchdict.xlsx', sheet_name='tag_mapping_frdict', usecols='A,B', index_col=0, header=0).to_dict()
        tag_dict = tag_dict['MAPTAG']
            
        french_dict_lower=[]
        for key in french_dict:
            #check whether word part of key is single word or tuple
            if type(key[0])!=tuple:
                #if it's single word
                french_dict_lower.append(((key[0].lower(),tag_dict[key[1]]), french_dict[key]))
            if type(key[0])==tuple:
                lower_tuple=[]
                for component in key[0]:
                    lower_tuple.append(component.lower())
                french_dict_lower.append( ((tuple(lower_tuple), tag_dict[key[1]]), french_dict[key]) )
        self.french_dict_pos=dict(french_dict_lower)
    
    @staticmethod
    def get_frenchscore_table_simple(text_untagged_items, french_dict):
        """
        Calculates normalised L1 Similarity scores for vocab items using the French dictionary

        Params:
            text_untagged_items(list): list of vocabulary items
            french_dict: dictionary which maps vocabulary items to frequencies in the BNC
        
        Returns: pandas dataframe of vocabulary items and their L1 Similarity scores
        """
        base_table = pd.DataFrame(text_untagged_items)
        base_table.columns=['word_in_text']
        base_table['score'] = base_table['word_in_text'].map(dict(french_dict))

        porter_stemmer = nltk.stem.porter.PorterStemmer()
        stemmer2 = nltk.SnowballStemmer("english", ignore_stopwords=False)
        lemmatizer = nltk.wordnet.WordNetLemmatizer()

        base_table['stemmed']=''
        for index, row in base_table.iterrows():
            if pd.isna(row['score']) and type(row['word_in_text'])!=tuple:
                row['stemmed']= porter_stemmer.stem(row['word_in_text'])
        base_table['stemmed_score'] = base_table['stemmed'].map(dict(french_dict))
        for index, row in base_table.iterrows():
            if pd.isna(row['score']):
                row['score']=row['stemmed_score']

        base_table['capitalised']=''
        for index, row in base_table.iterrows():
            if pd.isna(row['score']) and type(row['word_in_text'])!=tuple:
                row['capitalised']=row['word_in_text'].capitalize()
        base_table['capitalised_score'] = base_table['capitalised'].map(dict(french_dict))
        for index, row in base_table.iterrows():
            if pd.isna(row['score']):
                row['score']=row['capitalised_score']
    
        base_table['lemmatized']=''
        for index, row in base_table.iterrows():
            if pd.isna(row['score']) and type(row['word_in_text'])!=tuple:
                row['lemmatized'] = lemmatizer.lemmatize(row['word_in_text'], 'v')
        base_table['lemmatized_score'] = base_table['lemmatized'].map(dict(french_dict))
        for index, row in base_table.iterrows():
            if pd.isna(row['score']):
                row['score']=row['lemmatized_score']
    
        for index, row in base_table.iterrows():
            if pd.isna(row['score']) and type(row['word_in_text'])==tuple:
                row['score']=('unfound_mwes', 1.0)
            if type(row['word_in_text'])!=tuple:
                if pd.isna(row['score']) and bool(re.search('^[A-z]+$', row['word_in_text']))==False:
                    row['score']=('punct/number', 1.0)

        for index, row in base_table.iterrows():
            if pd.isna(row['score']) and type(row['word_in_text'])==tuple:
                row['score']=('unfound_mwes', 1.0)
            if type(row['word_in_text'])!=tuple:
                if pd.isna(row['score']) and bool(re.search('^[A-z]+$', row['word_in_text']))==False:
                    row['score']=('punct/number', 1.0)
        
        for index, row in base_table.iterrows():
            if pd.isna(row['score']):
                row['score']=('unfound single word', 1.0)
        
        base_table['french_score']=''
        for index, row in base_table.iterrows():
            row['french_score']=row['score'][1]

        return base_table
    
    @staticmethod
    def get_frenchscore_table_pos(text_maptagged_items, french_dict_pos, french_dict_simple):
        """
        Calculates normalised L1 Similarity scores for vocab items using the French dictionary
        Enhances previous method's performnace by incorporating POS tags
        Params:
            text_maptagged_items(list): list of vocabulary items with broad POS tags
            french_dict_pos: dictionary which maps vocab items + pos tags to French translations
            french_dict_simple: dictionary which maps vocab items alone to French translations
        
        Returns: pandas dataframe of vocabulary items and their L1 Similarity Scores
        """
        
        
        base_table = pd.DataFrame(text_maptagged_items)
        base_table.columns=['word_in_text', 'tag']
        base_table['word_and_tag']=''
        for index, row in base_table.iterrows():
            row['word_and_tag']=(row['word_in_text'], row['tag'])
        
        #first attempt to find a score by matching wholesale word+tag to an entry in french_dict with pos tags
        base_table['score'] = base_table['word_and_tag'].map(dict(french_dict_pos))

        porter_stemmer = nltk.stem.porter.PorterStemmer()
        stemmer2 = nltk.SnowballStemmer("english", ignore_stopwords=False)
        lemmatizer = nltk.wordnet.WordNetLemmatizer()

        base_table['stemmed']=''
        base_table['stemmed_and_tag']=''
        base_table['lemmatized']=''
        base_table['lemmatized_and_tag']=''
        
        #second and third attempts to find a score by matching stemmed then lemmatised word+tag to an entry in french_dict with pos tags
        for index, row in base_table.iterrows():
            if pd.isna(row['score']) and type(row['word_in_text'])!=tuple:
                row['stemmed']= porter_stemmer.stem(row['word_in_text'])
                row['stemmed_and_tag']=(row['stemmed'], row['tag'])
                if row['tag']=='VERB':
                    row['lemmatized'] = lemmatizer.lemmatize(row['word_in_text'], 'v')
                elif (row['tag']=='NOUN') or (row['tag']=='PRP_NOUN'):
                    row['lemmatized'] = lemmatizer.lemmatize(row['word_in_text'], 'n')
                elif (row['tag']=='ADJ_ADV') and (row['word_in_text'].endswith('ly')):
                    row['lemmatized'] = row['word_in_text'][:-2]
                row['lemmatized_and_tag']=(row['lemmatized'], row['tag'])
                
        base_table['stemmed_tagscore'] = base_table['stemmed_and_tag'].map(dict(french_dict_pos))
        for index, row in base_table.iterrows():
            if pd.isna(row['score']):
                row['score']=row['stemmed_tagscore']
    
        base_table['lemmatized_tagscore'] = base_table['lemmatized_and_tag'].map(dict(french_dict_pos))
        for index, row in base_table.iterrows():
            if pd.isna(row['score']):
                row['score']=row['lemmatized_tagscore']
        
        for index, row in base_table.iterrows():
            if pd.isna(row['score']):
                #fourth attempt to find a score by matching word without tag to an entry in french_dict_simple
                row['score']=(french_dict_simple.get(row['word_in_text']))
            if pd.isna(row['score']):
                #fifth attempt to find a score by matching stemmed word without tag to an entry in french_dict_simple
                row['score']=french_dict_simple.get(row['stemmed'])
            if pd.isna(row['score']):
                #sixth attempt to find a score by matching lemmatized word without tag to an entry in french_dict_simple
                row['score']=french_dict_simple.get(row['lemmatized'])
        
    
        for index, row in base_table.iterrows():
            if pd.isna(row['score']) and type(row['word_in_text'])==tuple:
                row['score']=('unfound_mwes', 1.0)
            if type(row['word_in_text'])!=tuple:
                if pd.isna(row['score']) and bool(re.search('^[A-z]+$', row['word_in_text']))==False:
                    row['score']=('punct/number', 1.0)
      
        for index, row in base_table.iterrows():
            if pd.isna(row['score']):
                row['score']=('unfound single word', 1.0)
        
        base_table['french_score']=''
        for index, row in base_table.iterrows():
            row['french_score']=row['score'][1]

        return base_table

