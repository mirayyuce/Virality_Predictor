import pandas as pd
import numpy as np
import os

import Utils as utils

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet as wn
nltk.download('wordnet')

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample


PATH = '~/Virality_Predictor/datasets/'
NUM_CLASSES = 15

stop_words = set(stopwords.words("english"))

class LemmaTokenizer(object):
    ''' Lemmatizes the text, discards stopwords, punctuation and digits
    Args:
        df: Articles
    Returns:
        df: Articles
    '''
    def __init__(self):
        
        self.wnl = WordNetLemmatizer()
        self.tags = {
                'N': wn.NOUN,
                'V': wn.VERB,
                'R': wn.ADV,
                'J': wn.ADJ,
            }

    def __call__(self, articles, lang):
        if lang == 'en':
            stop_words = set(stopwords.words("english"))
        elif lang == 'pt':
            stop_words = set(stopwords.words("portuguese"))

        # For each document: Tokenize the text, prepare pos-tags.
        for row_idx, text in articles.items():
            
            tokenized_text = word_tokenize(text)
            words_tags = pos_tag(tokenized_text)
            cleaned_text = []

            # For each word-token pair, check if it is a real word, and lemmatize it.
            for word_idx, word_tag in enumerate(words_tags):
                tag = word_tag[1]
                word = word_tag[0].lower()
                
                # Continue if punctuation or digit
                if not word.isalpha(): 
                    continue
                    
                # Continue if stopword
                elif word in stop_words:
                    continue
                
                # If tag not found, consider the word a NOUN
                tag = self.tags.get(tag[0], wn.NOUN)
                
                if not tag:
                    lemma = word
                else:
                    lemma = self.wnl.lemmatize(word, tag)

                cleaned_text.append(lemma)
            
            articles[row_idx] = (' ').join(cleaned_text)
            
        return articles


def clean_all_text(load, df, name, lang):
    ''' Calls the text preprocesser. Loads or dumps the cleaned text
    Args:
        df: Articles
        name: Name of the .csv file
        load: If true, it loads from datasets
        lang: Language of documents. Necessary for preprocessing
    Returns:
        df: Articles
    '''
    preprocess_lematizer = LemmaTokenizer()
    clean_file = os.path.join(PATH, 'cleaned_' + name + "_text.csv")
    
    
    if load:
        clean_df = pd.read_csv(clean_file, index_col=0)['all_text']

    else:   
        clean_df = preprocess_lematizer(df['all_text'], lang)
        clean_df.to_csv(clean_file, index=True)  
        
    return clean_df
    

def get_train_test_datasets(df):
    ''' Create train and test data
    Args:
        df: Articles
    Returns:
        articles_train, labels_train, articles_test, labels_test
    '''
    df_train, df_test = train_test_split(df, test_size = 0.20, random_state = 20)
    
    articles_train = df_train[['all_text','virality']]
    articles_test = df_test[['all_text','virality']]

    labels_train = articles_train['virality']
    labels_test = articles_test['virality']
    
    print('Dataset shapes: Train data',len(articles_train), ', Test data', len(articles_test),
          ', Train labels ',len(labels_train),', Test labels',len(labels_test))

    return articles_train, labels_train, articles_test, labels_test

def categorize_virality(virality_values):
    ''' Normalizes and categorizes virality values per article
    Args:
        virality_values: Sum of eventType based on event weight
    Returns:
        virality_categorized: Normalized and categorized virality
    '''
    num_bins = NUM_CLASSES
    norm_virality = virality_values / np.linalg.norm(virality_values)
    linspace = np.linspace(0.0, 1.0, num_bins)
    virality_categorized = np.digitize(norm_virality, linspace)
    return virality_categorized

def calculate_virality(articles_df, user_df):
    ''' Creates virality column for articles
    Args:
        articles_df: Articles
        user_df: Users
    Returns:
        articles_df: Articles with virality class labels
    '''
    virality_df = user_df[['contentId', 'virality']].groupby(['contentId'], sort=True).sum().apply(categorize_virality).reset_index()
    articles_df = pd.merge(articles_df, virality_df, on = 'contentId', how='inner')
    return articles_df


def hyperparameter_search(parameters, pipeline, train_data, train_labels):
    ''' Hyperparameter search using 3 fold cross validation
    Args:
        parameters: Parameters for search
        pipeline: sklearn pipeline object
        train_data: train_data
        train_labels: train_labels
    Returns:
        None
    '''

    grid_search = GridSearchCV(pipeline, parameters, cv=3, n_jobs=-1)
    
    grid_search = grid_search.fit(train_data, train_labels)
    
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, grid_search.best_params_[param_name]))
        
    print(grid_search.cv_results_)
    print(grid_search.best_score_)
    print(grid_search.scorer_)

def upsample_minority_classes(df):
    ''' Upsamples minority classes in Articles
    Args:
        df: Articles
        
    Returns:
        df_upsampled: Articles, classes balanced
    '''
    majority_size = max(list(df.groupby(["virality"]).size()))
    classes_df = [df[df.virality==1]]

    for cls in reversed(range(1,NUM_CLASSES+1)):
        minority_df = df[df.virality==cls]
        if not minority_df.empty:
            # Number of new samples to resample
            num_new_samples=majority_size-len(minority_df)
            # Upsample minority class
            
            minority_df_upsampled = resample(minority_df, 
                                            replace=True,     # sample with replacement
                                            n_samples=num_new_samples,    # to match majority class
                                            random_state=50)
            classes_df.append(minority_df_upsampled)
        df_upsampled = pd.concat(classes_df)
    return df_upsampled
