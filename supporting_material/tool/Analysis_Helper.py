
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
from pydoc import help
from scipy.stats.stats import pearsonr


# In[1]:


class Analysis_Helper:
    
    """
    This class provides a helper method to analyse feature scores against teacher (target) scores

    """
    
    @staticmethod
    def drop_dups_get_errorrate(scoredtable, featurescorecolumnname, textforgraphtitle='None', graph='no'):
        
        """
        Method prints mean squared error rate, Pearson correlation co-efficient and pvalue, and scattergraph
        of the specified feature against teacher (target) scores
        Params:
            scored_table: a pandas dataframe containing vocab items for a text with a column for each feature score
            featurescorecolumnname: specifies the feature to be analysed
            textforgraphtitle: text name for graph title
            graph: 'yes' to print the graph, 'no' to skip the graph print
        """
    
        table=scoredtable.copy()
        print('Analysis of', textforgraphtitle, featurescorecolumnname, 'against teacher scores:')
        #convert the numeric columns to float types
        table[featurescorecolumnname].astype(float)
        table['teacher_score']=table.teacher_score.astype(float)
    
        #drop any duplicate rows
        table.drop_duplicates(subset='word_in_text', keep='first', inplace=True)
    
        #get mean square error rate
        number_rows=table.shape[0]
        table['diff_squared']=(table['teacher_score']-table[featurescorecolumnname])**2
        msqer=table['diff_squared'].sum() / number_rows
        print('Mean squared error rate is', round(msqer,2))
    
        #get pearson correlation coefficient
        p_coeff, pval=pearsonr(table['teacher_score'], table[featurescorecolumnname])
        print('Pearson correlation co-efficient: ',round(p_coeff,2), 'and two-tailed pvalue:', round(pval,2))
    
        if graph=='yes':
           
            #make basic graph and set axis labels    
            ax = table.set_index('teacher_score')[featurescorecolumnname].plot(style='o', figsize=(10,7), grid=True, title=('Teacher Scores / ' + textforgraphtitle))
            ax.set_xlabel("teacher_score")
            ax.set_ylabel(featurescorecolumnname)
    
                #label scatter points #https://stackoverflow.com/questions/15910019/annotate-data-points-while-plotting-from-pandas dataframe
            #for i, point in table.iterrows():
                #ax.text(point['teacher_score'], point[featurescorecolumnname], str(point['word_in_text']))
    
            #add trend line https://stackoverflow.com/questions/41635448/how-can-i-draw-scatter-trend-line-on-matplot-python-pandas
            x=table['teacher_score']
            y=table[featurescorecolumnname]
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(x,p(x),"r--")
    
            print(plt.show())
        print('\n')

