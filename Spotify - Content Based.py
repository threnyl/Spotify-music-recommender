#!/usr/bin/env python
# coding: utf-8

# In[51]:


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


# # Preprocessing Genre

# In[52]:


def genre_preprocess(df):
    '''
    Preprocess the genre data
    '''
    df['genres_list'] = df['track_genre'].apply(lambda x: x.split(","))
    return df.drop(['track_genre'], axis=1)


# In[53]:


spotify_df = pd.read_csv('/Users/threnylaird/Documents/personal project/spotify/spotify_subset.csv', index_col=0)
date_df = pd.read_csv('/Users/threnylaird/Documents/personal project/spotify/playlist_date.csv', index_col=0)

spotify_df = genre_preprocess(spotify_df)


# In[59]:


spotify_df[spotify_df['track_id'].isin(date_df.track_id)]


# # Sentiment Analysis
# Analysis for text 

# In[60]:


def get_subjectivity(text):
    '''
    Get subjectivity using TextBlob
    '''
    return TextBlob(text).sentiment.subjectivity


def get_polarity(text):
    '''
    Get polarity using TextBlob
    '''
    return TextBlob(text).sentiment.polarity


def get_analysis(score, task='polarity'):
    '''
    Categorize polarity & subjectivity score
    '''
    if task == 'subjectivity':    
        if score < 1/3:
            return 'low'
        elif score > 2/3:
            return 'high'
        else:
            return 'medium'
    else:
        if score < 0:
            return 'negative'
        elif score == 0:
            return 'neutral'
        else:
            return 'positive'


def sentiment_analysis(df, text_col):
    '''
    Perform sentiment analysis on text
    '''
    df['subjectivity'] = df[text_col].apply(get_subjectivity).apply(lambda x: get_analysis(x,'subjectivity'))
    df['polarity'] = df[text_col].apply(get_polarity).apply(get_analysis)
    return df


# # One hot encoding

# In[61]:


def ohe_prep(df, column, new_name): 
    ''' 
    Create One Hot Encoded features of a specific column
    ---
    Input: 
    df (pandas dataframe): Spotify Dataframe
    column (str): Column to be processed
    new_name (str): new column name to be used
        
    Output: 
    tf_df: One-hot encoded features 
    '''
    tf_df = pd.get_dummies(df[column])
    tf_df = tf_df.add_prefix(new_name + '|')
    #feature_names = tf_df.columns
    #tf_df.columns = [new_name + "|" + str(i) for i in feature_names]
    tf_df.reset_index(drop = True, inplace = True)    
    return tf_df


# In[66]:


def create_features_set(df, float_cols):
    '''
    Process spotify df to create a final set of features that will be used to generate recommendations
    ---
    Input: 
    df (pandas dataframe): Spotify Dataframe
    float_cols (list(str)): List of float columns that will be scaled
            
    Output: 
    final (pandas dataframe): Final set of features 
    '''
    
    # Tfidf genre lists
    tfidf = TfidfVectorizer()
    tfidf_matrix =  tfidf.fit_transform(df['genres_list'].apply(lambda x: ' '.join(x)))
    genre_df = pd.DataFrame(tfidf_matrix.toarray())
    genre_df.columns = ['genre' + '|' + i for i in tfidf.get_feature_names()]
    if 'genre|unknown' in genre_df.columns: # drop unknown genre
        genre_df.drop(columns='genre|unknown', inplace=True) 
    genre_df.reset_index(drop = True, inplace=True)
    
    # Sentiment analysis
    df = sentiment_analysis(df, 'track_name')

    # One-hot Encoding
    subject_ohe = ohe_prep(df, 'subjectivity','subject') * 0.3
    polar_ohe = ohe_prep(df, 'polarity','polar') * 0.3
    key_ohe = ohe_prep(df, 'key','key') * 0.5
    mode_ohe = ohe_prep(df, 'mode','mode') * 0.5
    time_signt_ohe = ohe_prep(df, 'time_signature', 'time_sgnt') * 0.2

    # Normalization
    # Scale popularity columns
    pop = df[['track_pop']].reset_index(drop = True)
    scaler = MinMaxScaler()
    pop_scaled = pd.DataFrame(scaler.fit_transform(pop), columns = pop.columns) * 0.5
    
    # Scale audio columns
    floats = df[float_cols].reset_index(drop = True)
    scaler = MinMaxScaler()
    floats_scaled = pd.DataFrame(scaler.fit_transform(floats), columns = floats.columns) * 0.5

    # Concanenate all features
    final = pd.concat([genre_df, floats_scaled, pop_scaled, subject_ohe, polar_ohe, key_ohe, mode_ohe, time_signt_ohe], axis = 1)
    
    # Add song id
    final['track_id'] = df['track_id'].values
    
    return final


# In[67]:


float_cols = spotify_df.dtypes[spotify_df.dtypes == 'float64'].index.values
complete_features_df = create_features_set(spotify_df, float_cols=float_cols)


# In[117]:


complete_features_df


# In[70]:


def generate_playlist_vector(features_df, date_df):
    '''
    Summarize a user's playlist into a single vector
    ---
    Input: 
    complete_feature_set (pandas dataframe): Dataframe which includes all of the features for the spotify songs
    playlist_df (pandas dataframe): playlist dataframe
        
    Output: 
    complete_feature_set_playlist_final (pandas series): single vector feature that summarizes the playlist
    complete_feature_set_nonplaylist (pandas dataframe): 
    '''
    # Find playlist's and nonplaylist's songs
    playlist_df = features_df[features_df['track_id'].isin(date_df['track_id'].values)]
    nonplaylist_df = features_df[~features_df['track_id'].isin(date_df['track_id'].values)]
    
    playlist_df = playlist_df.merge(date_df, on = 'track_id', how = 'inner')
    playlist_df['date_added'] = pd.to_datetime(playlist_df['date_added'])#.dt.to_pydatetime()
    playlist_df.sort_values(by='date_added', ascending=False, inplace=True)
    
    # Create weight based on how recent the tracks are added to playlist
    oldest_date = playlist_df['date_added'].iloc[-1]
    playlist_df['months_from_oldest'] = ((playlist_df['date_added'] - oldest_date).dt.days/30) + 10 # adding 10 so the min value will not be 0
    playlist_df['weight'] = playlist_df['months_from_oldest'].apply(lambda x: x/playlist_df['months_from_oldest'].iloc[0])
    
    # Multiply each row with their correspondent weight
    playlist_df.update(playlist_df.iloc[:,:-4].mul(playlist_df.weight, axis=0))
    #print(playlist_df)
    
    # Drop columns 
    playlist_df = playlist_df.drop(columns=['track_id','date_added', 'months_from_oldest', 'weight'], axis=1)
    
    return playlist_df.sum(axis = 0), nonplaylist_df.reset_index(drop=True)
 


# In[71]:


playlist_vector, nonplaylist_df = generate_playlist_vector(complete_features_df, date_df)


# In[103]:


df1 = playlist_vector.filter(regex = '^genre')#.to_frame()
df1[df1>0]


# In[ ]:


nonplaylist_df


# In[124]:


filter_genre = ['genre|' in x for x in nonplaylist_df.columns]
genre_cols = nonplaylist_df.columns[filter_genre]
genre_cols


# In[125]:


def generate_playlist_recom(spotify_df, playlist_vector, nonplaylist_df, genre_cols):
    '''
    Generated recommendation based on songs in aspecific playlist.
    ---
    Input: 
    df (pandas dataframe): spotify dataframe
    features (pandas series): summarized playlist feature (single vector)
    nonplaylist_features (pandas dataframe): feature set of songs that are not in the selected playlist
        
    Output: 
    non_playlist_df_top_40: Top 40 recommendations for that playlist
    '''
    
    # Find cosine similarity between the playlist and the complete song set
    nonplaylist_df_top10 = nonplaylist_df.copy()
    nonplaylist_df_top10['sim'] = cosine_similarity(nonplaylist_df_top10.drop('track_id', axis = 1).values, 
                                                    playlist_vector.values.reshape(1, -1))[:,0]
    nonplaylist_df_top10.sort_values('sim', ascending = False, inplace=True)
    
    return spotify_df[spotify_df['track_id'].isin(nonplaylist_df_top10[:10]['track_id'])].reset_index(drop=True)


# In[126]:


#playlist_recom = generate_playlist_recom(spotify_df, playlist_vector, nonplaylist_df)
playlist_recom = generate_playlist_recom(spotify_df, df1, nonplaylist_df, genre_cols)
playlist_recom


# In[ ]:




