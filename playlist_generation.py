import csv
import os
import itertools
import ast
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
import pickle
from sklearn import metrics
import numpy as np

# similarity based (noaudio)
# popular rec (pop)
# greatest artist (sagh)
# collocation artist (cagh) - similarity based (noaudio - artist-level)

# What's missing in here.
# similarity based (audio) - should be matched with mixset sequencing
# audio based EDM artist classifier (feature) similarity based vs Rachel's shortest path

# read sequencing.csv 
playlists = {}
with open('sequencing.csv', mode='r') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        data = dict(row)
        playlist_id = data['id']
        tracks_in_playlist = ast.literal_eval(data['track_id'])
        playlists[playlist_id] = tracks_in_playlist

print(len(playlists))

tracks = playlists.values()
seed_list = [x[0] for x in tracks]
tracks = list(itertools.chain(*tracks))
tracks = list(set(tracks))
print(len(playlists))
print(len(tracks))
print(seed_list)
print(len(seed_list))


# total songs / binary matrix
df = pd.concat([pd.Series(v, name=k).astype(str) for k, v in playlists.items()], axis=1)
df = pd.get_dummies(df.stack()).sum(level=1).clip_upper(1)
print(df.shape)

# replace playlist index to the first item id of playlists
org_list = df.index.tolist()
df = df.rename(index=dict(zip(org_list,seed_list)))

print(df.head())
# for cagh, artist level playlist sim mat
# co-occur hammming distance similarity matrix (songXsong)
jac_sim = 1 - pairwise_distances(df.T, metric = "hamming")
# optionally convert it to a DataFrame
jac_sim = pd.DataFrame(jac_sim, index=df.columns, columns=df.columns)
print(jac_sim.shape)
print(jac_sim.head)
store = pd.HDFStore('jac_sim.h5')
store['jac_sim'] = jac_sim
store.close()

# greatest artist pop_seed
# read track_sub, find matching artist - artist-track dictionary - set used artist's tracks to 1 similarity score followed by popularity ranking
artist_to_track = {}
track_to_artist = {}
with open('track_subsub.csv', mode='r') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        data = dict(row)
        track_id = data['track_id']
        artist_id = data['artist_id']
        artist_to_track.setdefault(artist_id, []).append(track_id)
        track_to_artist[track_id] = artist_id

# sim mat based ranking for each seed (up to 300, 50 interval?) / df: ground truth
# AUC and mAP
jac_sim = pd.read_hdf('jac_sim.h5')

# set diagonal term to 0 to avoid self recommendation
jac_sim.values[[np.arange(jac_sim.shape[0])]*2] = 0

# set groundtruth self to 0 
for i,row in df.iterrows():
    df.set_value(i, i, 0)
print(df.head())

# popularity based
pops = df.sum(axis=0) # np.max(pop_seed) = 32
pop_seed = pops.values

auc_noaudio = []
auc_pop = []
auc_sagh = []
for iter in range(len(seed_list)):
    # ground truth
    gt_seed = df.iloc[iter,:].values
    
    # jaccard similarity based 
    noaudio_seed = jac_sim.loc[seed_list[iter],:].values

    # greatest artist pop seed
    seed_artist_tracks = artist_to_track[track_to_artist[seed_list[iter]]]
    sagh = pops.copy()
    for iter2 in range(len(seed_artist_tracks)):
        if seed_artist_tracks[iter2] in sagh.index:
            sagh.loc[seed_artist_tracks[iter2]] = 100 * sagh.loc[seed_artist_tracks[iter2]] # 100 * previous values
        else:
            continue
    sagh_seed = sagh.values

    auc_noaudio_seed = metrics.roc_auc_score(gt_seed,noaudio_seed)
    auc_noaudio.append(auc_noaudio_seed)
    print('auc_noaudio: ', np.mean(np.asarray(auc_noaudio)))

    auc_pop_seed = metrics.roc_auc_score(gt_seed,pop_seed)
    auc_pop.append(auc_pop_seed)
    print('auc_pop: ', np.mean(np.asarray(auc_pop)))

    auc_sagh_seed = metrics.roc_auc_score(gt_seed,sagh_seed)
    auc_sagh.append(auc_sagh_seed)
    print('auc_sagh: ', np.mean(np.asarray(auc_sagh)))


