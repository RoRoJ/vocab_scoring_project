
# coding: utf-8

# In[1]:


import nltk
import pandas as pd
import pickle
import re
from nltk.tokenize.moses import MosesDetokenizer
from nltk.corpus import wordnet

class TextItems:
    
    """
    This class facilitates the extraction of vocabulary items from a text file

    Attributes:
        text_sentences: list of sentence tokens from the text
    """
    
    #open text with plain text reader, return its tokenized sentences, and count of tokenized words
    def __init__(self, filename):
        """
        The constructor for the Textitems class.
        
        Parameters:
           filename (string): The filepath for the text file  
        """
        with open(filename, 'r', encoding="utf8") as myfile:
            text_plain = myfile.read()
            text_plain = text_plain.replace('\ufeff', '')
            text_plain = text_plain.replace('\n', ' ')
            self.text_sentences=nltk.tokenize.sent_tokenize(text_plain)

    @staticmethod
    def ngram_index(words, ngram):
        """
        Function to compute the position of an n-gram in a sentence.
        https://stackoverflow.com/questions/33393402/how-to-find-position-of-an-ngram-in-a-sentence
        
        Parameters:
           words (list): A list of word tokens(constituting a sentence)
           ngram (list): A list of word tokens (constituting an ngram potentially within the sentence)
        
        Returns:
            An integer corresponding to the starting index  of the ngram in the (list of) words
        """
        return list(nltk.ngrams(words, len(ngram))).index(tuple(ngram))
    
    #takes a sentence-tokenized text and master list of mwexps as args: must be in form [[mwe1],[mwe2],[mwn]]
    #returns one of two types of entity list:
    #1. All mwexpss and (mutually exclusive) single tokens found in the text
    #2. The above, with POS tags

    @staticmethod
    def get_text_items(text_sentences, return_option, case_option, exclude_nondict_option):
        
        """
        Function to extract vocabulary items and POS tags from a list of sentences
        
        Parameters:
            text_sentences (list): A list of sentence tokens
            return_option (string): Determines whether POS tags should be returned with the vocab items
            case_option (string): Determines whether the sentence tokens should be converted to lower case
            exclude_nondict_option (string): Determines whether non-dictionary items should be returned
        
        Returns:
            A list comprising single word tokens and tuple tokens of MWEs found within the text OR
            A list of tuples (comprising the above with POS tags)
        
        """
        
        #load list of mwes
        with open(r'obj/mwes_list.pkl', 'rb') as f:
            mwes_list= pickle.load(f)
        
        #transform mwes in the list to lower case
        if case_option=='all_lower':
            mwes_list_lower=[]
            mwes_list_entry_lower=[]
            for item in mwes_list:
                for word in item:
                    mwes_list_entry_lower.append(word.lower())
                mwes_list_lower.append(mwes_list_entry_lower)
                mwes_list_entry_lower=[]
            mwes_list=mwes_list_lower
      
        #initialise detokenizer and lemmatizer
        detokenizer = MosesDetokenizer()
        lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    
        #count the number of tokens in the whole text
        tokens_in_text=0
        for sentence in text_sentences:
            tokenized = nltk.tokenize.word_tokenize(sentence)
            tokens_in_text+=len(tokenized)
    
        #create a text index, with a boolean value corresponding to each individual token
        text_index = [True] * tokens_in_text

        found_ngrams=[] #this holds the mwexprs/ngrams found in the text
        pos_document=[] #this holds the pos-tagged tokens in the text
        token_counter=0 #this accumulates the number of tokens processed in each loop 
    
        #loops through each sentence in the text
        for i in range(len(text_sentences)):
            tokenized = nltk.tokenize.word_tokenize(text_sentences[i]) #tokenises the current sentence
            tokenized_for_inf = nltk.tokenize.word_tokenize(text_sentences[i]) #copy of above for an infinitive search
            tokenized_tagged = nltk.pos_tag(tokenized)  #tokenises and POS-tags the current sentence
    
            if case_option=='all_lower':
                tokenized = [item.lower() for item in tokenized]
                tokenized_for_inf = [item.lower() for item in tokenized_for_inf]
                tokenized_tagged = [(item[0].lower(), item[1]) for item in tokenized_tagged]
        
            pos_document.append(tokenized_tagged)
    
            #run through the tokenized sentence. If a verb is detected, change the verb to the 
            #infinitive in the tokenized_for_inf sentence
            for i in range(len(tokenized_for_inf)):
                if tokenized_tagged[i][1][0]=='V':
                    tokenized_for_inf[i]=lemmatizer.lemmatize(tokenized[i], 'v')
    
            #create two joined strings of the sentence, one in its original form, one with the verbs in the infinitve
            joined_string_orig=detokenizer.detokenize(tokenized, return_str=True)
            joined_string_inf=detokenizer.detokenize(tokenized_for_inf, return_str=True)
    
            #loop through the master list of multiword expressions
            for element in mwes_list:
                joined=' '.join(element) #join the mwe into a string with spaces
        
                #for speed improvements, check whether the joined mwe is in the joined original string. Skip rest if not.
                if joined in joined_string_orig:
                    sentence_ngrams=list(nltk.ngrams(tokenized, len(element)))
                    sentence_ngrams_index=list(nltk.ngrams(text_index[(token_counter):(token_counter+len(tokenized))], len(element)))
                    #this 2nd variable copies the current state of the text_index for this set of ngrams
            
                    #...check each mwe found in the joined sentence against each ngram in the tokenised sentence
                    for n in range(len(sentence_ngrams)):
                        #only let it match if at least one of the tokens in the ngram hasn't been used in another mwe match
                        if tuple(element)==sentence_ngrams[n] and (True in sentence_ngrams_index[n]):
                            found_ngrams.append(sentence_ngrams[n]) #append the found mwe ngram to the list
                            ngram_length=len(sentence_ngrams[n])
                            for q in range(ngram_length): #this loop sets the booleans in the text index that correspond to the ngram, to False
                                text_index[q+token_counter+(TextItems.ngram_index(tokenized,sentence_ngrams[n]))]=False
                            break #once an mwe is found in the sentence ngrams, the rest of its ngrams aren't checked
                    
                #for speed improvements, check whether the joined mwe is in the joined verbs-in-infinitve string. Skip rest if not.
                elif joined in joined_string_inf:
                    sentence_ngrams=list(nltk.ngrams(tokenized_for_inf, len(element))) 
                    sentence_ngrams_index=list(nltk.ngrams(text_index[(token_counter):(token_counter+len(tokenized))], len(element)))           
            
                    for n in range(len(sentence_ngrams)): 
                        if (tuple(element)==sentence_ngrams[n]) and (True in sentence_ngrams_index[n]):
                            found_ngrams.append(sentence_ngrams[n])
                            ngram_length=len(sentence_ngrams[n])
                            for q in range(ngram_length):
                                text_index[q+token_counter+(TextItems.ngram_index(tokenized_for_inf,sentence_ngrams[n]))]=False
                            break
                    
            token_counter+=len(tokenized) #increment the token counter by the length of this sentence's tokens
    
        #turns the list of sublists of tuples that is the current pos_doc, into a flat list of tuples
        flat_pos_doc = [item for sublist in pos_document for item in sublist]
        if len(flat_pos_doc)!=len(text_index):
                print('Error when flattening and index-checking entities')
    
        #appends valid single word tokens (not with corresponding False ie used by a MWES) to entity list
        entities=[]
        for i in range(len(flat_pos_doc)):
            if text_index[i]==True and re.search('^d*[A-z]', flat_pos_doc[i][0]) and flat_pos_doc[i][0]!='[' and flat_pos_doc[i][0]!=']':
                entities.append(flat_pos_doc[i])
        #appends found mwes to entity list
        for item in found_ngrams:
            entities.append((item, 'MWE'))
        
        #excludes words that aren't either an MWE, a stopword, hyphenated or in Wordnet
        if exclude_nondict_option=='yes':
            extended_stopwords=["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "whichever", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]
            entities_2=[]
            for item in entities:
                if item[0]=='s':
                    pass
                elif item[1]=="MWE":
                    entities_2.append(item)
                elif '-' in item[0]:
                    entities_2.append(item)
                elif(wordnet.synsets(item[0])) or (item[0] in extended_stopwords):
                    entities_2.append(item)
            entities=entities_2
    
        if return_option=='tagged':
            return [item for item in entities]
    
        elif return_option=='untagged':
            return [item[0] for item in entities]
        
    @staticmethod
    def map_text_tags(text_tagged_items):
        
        """
        Function to map NLTK POS tags to broader POS tags
        
        Parameters:
            text_tagged_items(list): A list of tuples (vocabitem, NLTKPOStag)
        
        Returns:
            A list of tuples (vocabitem, broadPOStag)
        
        """
        
        #read list of tag mappings from excel:
        tag_dict = pd.read_excel('files/tag_mapping.xlsx', usecols='A,B', index_col=0, header=0).to_dict()
        tag_dict = tag_dict['MY CAT']
        #perform the mapping:
        text_maptagged_entities=[]
        for item in text_tagged_items:
            mapped_item=(item[0], tag_dict[item[1]])
            text_maptagged_entities.append(mapped_item)
        return text_maptagged_entities

