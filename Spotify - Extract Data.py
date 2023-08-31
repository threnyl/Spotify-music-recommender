#!/usr/bin/env python
# coding: utf-8

# In[1]:


import spotipy 
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np


# In[2]:


def drop_duplicates(df):
    '''
    Drop song duplicates
    '''
    df['artists_song'] = df.apply(lambda row: row['artists']+row['track_name'],axis = 1)
    return df.drop_duplicates('artists_song').drop(['artists_song'], axis=1).reset_index(drop=True)


# In[3]:


dat = pd.read_csv('/Users/threnylaird/Documents/personal project/spotify/spotify_subset.csv', index_col=0)
dat


# In[3]:


# Read in the raw file and save it in dataframe
dat = pd.read_csv('/Users/threnylaird/Documents/personal project/spotify/spotify_dataset.csv', index_col=0)

# Check how many unique genres in the original dataset
print(dat.nunique(axis=0)['track_genre'])

# Get 10,000 tracks randomly
dat = dat.sample(n=10000,replace=False).reset_index(drop=True)

# Drop duplicate songs
dat = drop_duplicates(dat)

# Check how many songs remain in the dataframe
print(dat.shape[0])

# Check if the dataframe contains all 114 genres from original dataset
print(dat.nunique(axis=0)['track_genre'])

# See how many tracks for each genre
pd.set_option('display.max_rows', None)
print(dat.groupby('track_genre').count().iloc[:,0])
pd.set_option('display.max_rows', 10)

'''
In this dataset, each track is categorized into a single genre
''' 

# Change popularity column to track_pop
dat.rename(columns = {'popularity':'track_pop'}, inplace = True)

# Select useful columns
dat = dat[['track_id', 'track_name', 'artists', 'track_genre', 'track_pop', 'danceability', 
           'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 
           'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']]


# In[7]:


dat


# In[4]:


def get_items_data(items, sp):
    '''
    Get tracks' data from dictionary
    '''
    # Lists of features to get
    tracks_id = [] #Id
    tracks_title = [] #Song title
    tracks_artists = [] #Artist
    tracks_pop = [] #Track's popularity
    tracks_added = [] #Date of the track added to playlist
    artists_id = [] #Artists' id
    tracks_genre = [] #Artists' genre
    
    for track in items:
        if track['track'] is None:
            continue
        else:
            tracks_id.append(track['track']['id'])
            tracks_title.append(track['track']['name'])
            tracks_artists.append(track['track']['artists'][0]['name'])
            tracks_pop.append(track['track']['popularity'])
            tracks_added.append(track['added_at'][:10])
            artists_id.append(track['track']['artists'][0]['id'])
    
    '''
    We will get genres from the artists's data
    However, artists can have multiple genres and they are stored in lists
    Therefore, the artists' genres might not be the best genre representation for the tracks
    '''
    
    # Spotipy has query limits of 50 artists at a time, so split the list into 2 halves     
    for arr in np.array_split(artists_id, 2):
        for artist in sp.artists(arr)['artists']:
            # If genres' list is empty, put 'unknown'
            if artist['genres']:
                tracks_genre.append(artist['genres'])
            else:
                tracks_genre.append(['unknown'])
    
    # Extract audio features of the tracks
    playlist_aud_features = sp.audio_features(tracks_id)
    
    # Convert date added from string to datetime type
    tracks_added = pd.to_datetime(tracks_added)
    
    # Put every feature to one dataframe except date added
    df_features = pd.DataFrame(data=playlist_aud_features, columns=playlist_aud_features[0].keys())
    df_features['track_id'] = tracks_id
    df_features['track_name'] = tracks_title
    df_features['artists'] = tracks_artists
    df_features['track_pop'] = tracks_pop
    # Convert genres into string
    df_features['track_genre'] = [','.join(x) for x in tracks_genre] 
    df_features = df_features[['track_id', 'track_name', 'artists', 'track_genre', 'track_pop', 
                               'danceability', 'energy', 'key', 'loudness', 'mode', 
                               'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
                               'valence', 'tempo', 'time_signature']]
    
    # Create seperate dataframe for date added
    df_date_added = pd.DataFrame(list(zip(tracks_id, tracks_added)), columns=['track_id', 'date_added'])
    
    return df_features.reset_index(drop=True), df_date_added.reset_index(drop=True)
    

def get_playlist_data(url, df):

    # Authentication - without user
    with open('/Users/threnylaird/Documents/personal project/spotify/secret.txt') as f:
        secret_ls = f.readlines()
        cid = secret_ls[0][:-1]
        secret = secret_ls[1]
        
    client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
    sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)
    
    
    # Get playlist uri
    playlist_uri = url.split('/')[-1].split('?')[0]
    playlist_tracks_data = sp.playlist_tracks(playlist_uri)
    
    # Get feature and date dataframe 
    feature_df, date_df = get_items_data(playlist_tracks_data['items'], sp)
    
    # Concatenate feature dataframe to our dataframe
    df = pd.concat([df, feature_df], axis=0)
    
    # Spotipy has 100 tracks query limits, so iterate until the last track
    while playlist_tracks_data['next']:
        playlist_tracks_data = sp.next(playlist_tracks_data)
        feature_df, date_next = get_items_data(playlist_tracks_data['items'], sp)
        df = pd.concat([df, feature_df], axis=0)
        date_df = pd.concat([date_df, date_next], axis=0)
    
    return df.reset_index(drop=True), date_df.reset_index(drop=True)


# In[14]:


playlist_link = 'https://open.spotify.com/playlist/5pW6m6RZ8jGueVIck7vGU4?si=4ca08d5c0928498d'
dat, date_df = get_playlist_data(playlist_link, dat)
dat = drop_duplicates(dat)


# In[15]:


dat


# In[16]:


date_df


# In[17]:


with open('/Users/threnylaird/Documents/personal project/spotify/spotify_subset.csv', 'w') as csv_file:
    dat.to_csv(path_or_buf=csv_file, index=True)

with open('/Users/threnylaird/Documents/personal project/spotify/playlist_date.csv', 'w') as csv_file:
    date_df.to_csv(path_or_buf=csv_file, index=True)

