import pandas as pd
import numpy as np
import os
import re

from sklearn import preprocessing
PATH = '~/Desktop/Projects/Virality_Predictor/datasets/'

def split_url(url_string):
    ''' Splits url into domain name. Near perfect.
    Args:
        url_string: Url of each article
    Returns:
        Domain name 
    '''
    ex = r"(?:http(?:s)?:\/\/(?:www.)?)(.+)(?:\.\w+\/)"
    return re.findall(ex, url_string)[0]


def filter_articles_in_lang(df, langs):
    ''' Returns articles in selected language
    Args:
        langs: Selected language or languages
        df: Articles dataframe
    Returns:
        new_df
    '''
    if len(langs) == 1:
        new_df = df[df['lang'] == langs[0]]    
    else:
        new_df = df[df.lang.isin(langs)]
    return new_df


def articles_bag_of_words(df):
    ''' Combines domain_name, title and text columns into a new one called all_text
    Args:
        df: Articles dataframe
    Returns:
        df
    '''
    df['all_text'] = df['domain_name'] + " " + df['title'] + " " + df['text']
    return df


def normalize_virality_users(df):
    ''' Normalizes virality values per article
    Args:
        df: User data
    Returns:
        df_normalized: Normalized virality dataframe
    '''
    # Create x, where x the 'scores' column's values as floats
    x = df[['virality']].values.astype(float)

    # Create a minimum and maximum processor object
    min_max_scaler = preprocessing.MinMaxScaler()

    # Create an object to transform the data to fit minmax processor
    x_scaled = min_max_scaler.fit_transform(x)

    # Run the normalizer on the dataframe
    df_normalized = pd.DataFrame(x_scaled)
    return df_normalized


def load_shared_articles(file_name, lang):
    ''' Loads the articles data
    Args:
        file_name: Articles file name, .csv
        lang: Desired article language, str
    Returns:
        df
    '''
    articles_file = os.path.join(PATH, file_name)
    original_df = pd.read_csv(articles_file)
    
    # Get only 'shared' articles
    df = original_df[original_df['eventType'] == 'CONTENT SHARED']
    
    # Domain name could be a useful feature
    df['domain_name'] = df['url'].apply(split_url)
    
    # Filter articles in language
    df = filter_articles_in_lang(df, lang)
    
    # Create bag of words using domain name, title, and text
    df = articles_bag_of_words(df)
    
    # Drop columns that not used in this work
    df.drop(columns = ['url','timestamp','eventType','authorPersonId','authorSessionId','authorUserAgent','authorRegion','authorCountry'], inplace = True)
    df.set_index('contentId', inplace=True)
    return df

def load_shared_articles_with_weekday(file_name, lang):
    ''' Loads the articles data. Adds information about weekday activity
    Args:
        file_name: Articles file name, .csv
        lang: Desired article language, str
    Returns:
        df
    '''
    articles_file = os.path.join(PATH, file_name)
    original_df = pd.read_csv(articles_file)
    
    # Get only 'shared' articles
    df = original_df[original_df['eventType'] == 'CONTENT SHARED']
    
    # Domain name could be a useful feature
    df['domain_name'] = df['url'].apply(split_url)
    
    # Convert timestamp to date_time
    df['date_time'] =pd.to_datetime(df['timestamp'], unit='s')

    # Add a new column to see whether an article was shared on weekdays or not
    df['weekday'] = ((pd.DatetimeIndex(df['date_time']).dayofweek) // 5 == 1).astype(float)

    # Filter articles in English
    df = filter_articles_in_lang(df, lang)
    
    # Create bag of words using domain name, title, and text
    df = articles_bag_of_words(df)
    
    # Drop columns that not used in this work
    df.drop(columns = ['url','timestamp','eventType','authorPersonId','authorSessionId','authorUserAgent','authorRegion','authorCountry'], inplace = True)
    df.set_index('contentId', inplace=True)
    return df


def load_user_interactions_with_weekday(file_name, articles, to_normalize=False):
    ''' Loads the users data, calculates virality per person, per article. Adds information about weekday activity
    Args:
        file_name: Articles file name, .csv
        articles: Indices of articles not removed, Series
        normalize: Normalizes virality values. Boolean
        
    Returns:
        df
    '''
    users_file = os.path.join(PATH, 'users_interactions.csv')
    original_df = pd.read_csv(users_file)
    
    # Change eventType strings to virality values based on event weights
    article_importance = {"VIEW": 1, "LIKE": 4, "COMMENT CREATED": 10, "FOLLOW": 25, "BOOKMARK": 100}
    original_df["virality"] = original_df["eventType"].apply(lambda x: article_importance[x])
    original_df.drop(columns = ['eventType'])

    # Convert timestamp to date_time
    original_df['date_time'] =pd.to_datetime(original_df['timestamp'], unit='s')

    # Add a new column to see whether the users are active in weekdays or not
    original_df['weekday'] = ((pd.DatetimeIndex(original_df['date_time']).dayofweek) // 5 == 0).astype(float)
    
    if to_normalize:
        df = original_df.groupby(['personId', 'contentId'])['virality'].sum().apply(normalize_virality).reset_index()
    else:
        df = original_df.groupby(['personId', 'contentId'])['virality'].sum().reset_index()
        
    df = df[['virality', 'contentId', 'personId']]
    df = pd.merge(df, articles, on = 'contentId', how='inner')
    
    return df


def load_user_interactions(file_name, articles, to_normalize=False):
    ''' Loads the users data, calculates virality per person, per article
    Args:
        file_name: Articles file name, .csv
        articles: Articles df
        normalize: Normalizes virality values. Boolean
        
    Returns:
        df
    '''
    users_file = os.path.join(PATH, 'users_interactions.csv')
    original_df = pd.read_csv(users_file)
    
    # Change eventType to virality values based on event weights
    article_importance = {"VIEW": 1, "LIKE": 4, "COMMENT CREATED": 10, "FOLLOW": 25, "BOOKMARK": 100}
    original_df["virality"] = original_df["eventType"].apply(lambda x: article_importance[x])
    original_df.drop(columns = ['eventType'])

    if to_normalize:
        df = original_df.groupby(['personId', 'contentId'])['virality'].sum().apply(np.log10).reset_index()
    else:
        df = original_df.groupby(['personId', 'contentId'])['virality'].sum().reset_index()
        
    df = df[['virality', 'contentId', 'personId']]
    df = pd.merge(df, articles, on = 'contentId', how='inner')
    
    return df



def filter_user_interactions(df, min_value):
    """Discards users based on a minimum number of interactions

    Args:
        df: Users data
        min_value: minimum number of interactions to keep
    Returns:
        df
    """
    # Number of users before filtering
    num_users_before = df['personId'].nunique()

    # Users with enough interactions
    filter_users = df['personId'].value_counts() > min_value
    filter_users = filter_users[filter_users].index.tolist()

    # New dataframe with only selected users
    df = df[(df['personId'].isin(filter_users))]
    df = df[['personId', 'contentId', 'virality']]

    # Number of users after filtering
    num_users_after = df['personId'].nunique()

    print('Number of users discarded: ', num_users_before - num_users_after)
    return df




