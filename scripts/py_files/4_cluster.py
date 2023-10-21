#!/usr/bin/env python
# coding: utf-8

import os
import csv
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util


# # Read Encoded Data

all_posts_embed = np.load('../data/merged_q/all_posts_max_len_40_embed.npy')
all_replies_embed = np.load('../data/merged_q/all_replies_max_len_40_embed.npy')

print('all_posts_embed shape:', all_posts_embed.shape)
print('all_replies_embed shape:', all_replies_embed.shape)


# # Clustering Thresholds

thresholds = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]


# # Cluster the Posts

all_posts_clusters = {}

for threshold in thresholds:
    print('Clustering with threshold = {}...'.format(threshold))
    start_time = time.time()
    all_posts_clusters[threshold] = util.community_detection(all_posts_embed,
                                                             min_community_size = 2,
                                                             threshold = threshold)
    print('Done after {:.2f} sec'.format(time.time() - start_time))

for threshold in thresholds:
    print('----- Threshold = {} -----'.format(threshold))
    num_posts_clustered = 0
    for c in all_posts_clusters[threshold]:
        num_posts_clustered += len(c)
    print('Num clusters = {}'.format(len(all_posts_clusters[threshold])))
    print('Num posts clustered = {}'.format(num_posts_clustered))
    print('Remaining num posts = {}'.format(all_posts_embed.shape[0] - num_posts_clustered))
    print('Total num post nodes = {}'.format(all_posts_embed.shape[0] - num_posts_clustered + len(all_posts_clusters[threshold])))
    print()

with open('../data/merged_q/all_posts_max_len_40_clusters.pickle', 'wb') as f:
    pickle.dump(all_posts_clusters, f)


# ### More

with open('../data/merged_q/all_posts_max_len_40_clusters.pickle', 'rb') as f:
    all_posts_clusters = pickle.load(f)

for threshold in thresholds:
    print('Clustering with threshold = {}...'.format(threshold))
    start_time = time.time()
    all_posts_clusters[threshold] = util.community_detection(all_posts_embed,
                                                             min_community_size = 2,
                                                             threshold = threshold)
    print('Done after {:.2f} sec'.format(time.time() - start_time))

for threshold in thresholds:
    print('----- Threshold = {} -----'.format(threshold))
    num_posts_clustered = 0
    for c in all_posts_clusters[threshold]:
        num_posts_clustered += len(c)
    print('Num clusters = {}'.format(len(all_posts_clusters[threshold])))
    print('Num posts clustered = {}'.format(num_posts_clustered))
    print('Remaining num posts = {}'.format(all_posts_embed.shape[0] - num_posts_clustered))
    print('Total num post nodes = {}'.format(all_posts_embed.shape[0] - num_posts_clustered + len(all_posts_clusters[threshold])))
    print()

with open('../data/merged_q/all_posts_max_len_40_clusters.pickle', 'wb') as f:
    pickle.dump(all_posts_clusters, f)


# # Cluster the Replies (1/2)

N = all_replies_embed.shape[0]

all_replies_embed_1 = all_replies_embed[:N//2,:]
print('all_replies_embed_1 shape:', all_replies_embed_1.shape)

all_replies_clusters_1 = {}

for threshold in thresholds:
    print('Clustering with threshold = {}...'.format(threshold))
    start_time = time.time()
    all_replies_clusters_1[threshold] = util.community_detection(all_replies_embed_1,
                                                                 min_community_size = 2,
                                                                 threshold = threshold)
    print('Done after {:.2f} sec'.format(time.time() - start_time))

for threshold in thresholds:
    print('----- Threshold = {} -----'.format(threshold))
    num_replies_clustered = 0
    for c in all_replies_clusters_1[threshold]:
        num_replies_clustered += len(c)
    print('Num clusters = {}'.format(len(all_replies_clusters_1[threshold])))
    print('Num replies clustered = {}'.format(num_replies_clustered))
    print('Remaining num replies = {}'.format(all_replies_embed_1.shape[0] - num_replies_clustered))
    print('Total num replies nodes = {}'.format(all_replies_embed_1.shape[0] - num_replies_clustered + len(all_replies_clusters_1[threshold])))
    print()

with open('../data/merged_q/all_replies_max_len_40_clusters/all_replies_clusters_1.pickle', 'wb') as f:
    pickle.dump(all_replies_clusters_1, f)


# ### More

N = all_replies_embed.shape[0]
all_replies_embed_1 = all_replies_embed[:N//2,:]
print('all_replies_embed_1 shape:', all_replies_embed_1.shape)

with open('../data/merged_q/all_replies_max_len_40_clusters/all_replies_clusters_1.pickle', 'rb') as f:
    all_replies_clusters_1 = pickle.load(f)

for threshold in thresholds:
    print('Clustering with threshold = {}...'.format(threshold))
    start_time = time.time()
    all_replies_clusters_1[threshold] = util.community_detection(all_replies_embed_1,
                                                                 min_community_size = 2,
                                                                 threshold = threshold)
    print('Done after {:.2f} sec'.format(time.time() - start_time))

for threshold in thresholds:
    print('----- Threshold = {} -----'.format(threshold))
    num_replies_clustered = 0
    for c in all_replies_clusters_1[threshold]:
        num_replies_clustered += len(c)
    print('Num clusters = {}'.format(len(all_replies_clusters_1[threshold])))
    print('Num replies clustered = {}'.format(num_replies_clustered))
    print('Remaining num replies = {}'.format(all_replies_embed_1.shape[0] - num_replies_clustered))
    print('Total num replies nodes = {}'.format(all_replies_embed_1.shape[0] - num_replies_clustered + len(all_replies_clusters_1[threshold])))
    print()

with open('../data/merged_q/all_replies_max_len_40_clusters/all_replies_clusters_1.pickle', 'wb') as f:
    pickle.dump(all_replies_clusters_1, f)


# # Cluster the Replies (2/2)

N = all_replies_embed.shape[0]

all_replies_embed_2 = all_replies_embed[N//2:,:]
print('all_replies_embed_2 shape:', all_replies_embed_2.shape)

all_replies_clusters_2 = {}

for threshold in thresholds:
    print('Clustering with threshold = {}...'.format(threshold))
    start_time = time.time()
    all_replies_clusters_2[threshold] = util.community_detection(all_replies_embed_2,
                                                                 min_community_size = 2,
                                                                 threshold = threshold)
    print('Done after {:.2f} sec'.format(time.time() - start_time))

for threshold in thresholds:
    print('----- Threshold = {} -----'.format(threshold))
    num_replies_clustered = 0
    for c in all_replies_clusters_2[threshold]:
        num_replies_clustered += len(c)
    print('Num clusters = {}'.format(len(all_replies_clusters_2[threshold])))
    print('Num replies clustered = {}'.format(num_replies_clustered))
    print('Remaining num replies = {}'.format(all_replies_embed_2.shape[0] - num_replies_clustered))
    print('Total num replies nodes = {}'.format(all_replies_embed_2.shape[0] - num_replies_clustered + len(all_replies_clusters_2[threshold])))
    print()

with open('../data/merged_q/all_replies_max_len_40_clusters/all_replies_clusters_2.pickle', 'wb') as f:
    pickle.dump(all_replies_clusters_2, f)


# ### More

N = all_replies_embed.shape[0]
all_replies_embed_2 = all_replies_embed[N//2:,:]
print('all_replies_embed_2 shape:', all_replies_embed_2.shape)

with open('../data/merged_q/all_replies_max_len_40_clusters/all_replies_clusters_2.pickle', 'rb') as f:
    all_replies_clusters_2 = pickle.load(f)

for threshold in thresholds:
    print('Clustering with threshold = {}...'.format(threshold))
    start_time = time.time()
    all_replies_clusters_2[threshold] = util.community_detection(all_replies_embed_2,
                                                                 min_community_size = 2,
                                                                 threshold = threshold)
    print('Done after {:.2f} sec'.format(time.time() - start_time))

for threshold in thresholds:
    print('----- Threshold = {} -----'.format(threshold))
    num_replies_clustered = 0
    for c in all_replies_clusters_2[threshold]:
        num_replies_clustered += len(c)
    print('Num clusters = {}'.format(len(all_replies_clusters_2[threshold])))
    print('Num replies clustered = {}'.format(num_replies_clustered))
    print('Remaining num replies = {}'.format(all_replies_embed_2.shape[0] - num_replies_clustered))
    print('Total num replies nodes = {}'.format(all_replies_embed_2.shape[0] - num_replies_clustered + len(all_replies_clusters_2[threshold])))
    print()

with open('../data/merged_q/all_replies_max_len_40_clusters/all_replies_clusters_2.pickle', 'wb') as f:
    pickle.dump(all_replies_clusters_2, f)


# # Cluster the Replies (Centroids)

all_replies_clusters = {}

for threshold in thresholds:
    print('----- Threshold = {} -----'.format(threshold))
    start_time = time.time()
    replies_embed = []
    for c in all_replies_clusters_1[threshold]:
        replies_embed.append(all_replies_embed_1[c[0]])
    for c in all_replies_clusters_2[threshold]:
        replies_embed.append(all_replies_embed_2[c[0]])
    replies_embed = np.array(replies_embed)
    print('replies_embed shape:', replies_embed.shape)
    all_replies_clusters[threshold] = util.community_detection(replies_embed,
                                                               min_community_size = 1,
                                                               threshold = threshold)
    print('Num clusters:', len(all_replies_clusters[threshold]))
    print('Done after {:.2f} sec'.format(time.time() - start_time))
    print()

all_replies_clusters_combine_1 = {}

N_1 = all_replies_embed_1.shape[0]

for threshold in thresholds:
    clusters_1 = all_replies_clusters_1[threshold]
    clusters_2 = all_replies_clusters_2[threshold]
    clusters = []
    for c in all_replies_clusters[threshold]:
        cluster = []
        for i in c:
            if i < len(clusters_1):
                cluster += clusters_1[i]
            else:
                cluster += [j+N_1 for j in clusters_2[i-len(clusters_1)]]
        clusters.append(cluster)
    all_replies_clusters_combine_1[threshold] = clusters
    print('Threshold = {}, num clusters = {}'.format(threshold, len(clusters)))

with open('../data/merged_q/all_replies_max_len_40_clusters/all_replies_clusters_combine_1_centroid.pickle', 'wb') as f:
    pickle.dump(all_replies_clusters_combine_1, f)


# # Cluster the Replies (Mean)

all_replies_clusters = {}

for threshold in thresholds:
    print('----- Threshold = {} -----'.format(threshold))
    start_time = time.time()
    replies_embed = []
    for c in all_replies_clusters_1[threshold]:
        replies_embed.append(np.mean(all_replies_embed_1[c], axis = 0))
    for c in all_replies_clusters_2[threshold]:
        replies_embed.append(np.mean(all_replies_embed_2[c], axis = 0))
    replies_embed = np.array(replies_embed)
    print('replies_embed shape:', replies_embed.shape)
    all_replies_clusters[threshold] = util.community_detection(replies_embed,
                                                               min_community_size = 1,
                                                               threshold = threshold)
    print('Num clusters:', len(all_replies_clusters[threshold]))
    print('Done after {:.2f} sec'.format(time.time() - start_time))
    print()

all_replies_clusters_combine_1 = {}

N_1 = all_replies_embed_1.shape[0]

for threshold in thresholds:
    clusters_1 = all_replies_clusters_1[threshold]
    clusters_2 = all_replies_clusters_2[threshold]
    clusters = []
    for c in all_replies_clusters[threshold]:
        cluster = []
        for i in c:
            if i < len(clusters_1):
                cluster += clusters_1[i]
            else:
                cluster += [j+N_1 for j in clusters_2[i-len(clusters_1)]]
        clusters.append(cluster)
    all_replies_clusters_combine_1[threshold] = clusters
    print('Threshold = {}, num clusters = {}'.format(threshold, len(clusters)))

with open('../data/merged_q/all_replies_max_len_40_clusters/all_replies_clusters_combine_1_mean.pickle', 'wb') as f:
    pickle.dump(all_replies_clusters_combine_1, f)


# # Check

all_replies_df = pd.read_csv('../data/merged_q/all_replies_max_len_40.csv')

with open('../data/merged_q/all_replies_max_len_40_clusters/all_replies_clusters_combine_1_mean.pickle', 'rb') as f:
    all_replies_clusters_combine_1_mean = pickle.load(f)

for threshold in thresholds:
    mean_sample_df = []
    clusters = all_replies_clusters_combine_1_mean[threshold]
    all_idx = [i for i in range(len(clusters)) if len(clusters[i]) >= 10]
    idx = np.random.choice(all_idx, 10, replace = False)
    for i in idx:
        cluster = clusters[i]
        rows = np.random.choice(cluster, 10, replace = False)
        mean_sample_df.append(all_replies_df.iloc[rows])
    mean_sample_df = pd.concat(mean_sample_df)
    mean_sample_df.to_csv('../data/merged_q/all_replies_max_len_40_clusters/samples/mean_sample_df_{:.2f}.csv'.format(threshold), index = False)

with open('../data/merged_q/all_replies_max_len_40_clusters/all_replies_clusters_combine_1_centroid.pickle', 'rb') as f:
    all_replies_clusters_combine_1_centroid = pickle.load(f)

for threshold in thresholds:
    centroid_sample_df = []
    clusters = all_replies_clusters_combine_1_centroid[threshold]
    all_idx = [i for i in range(len(clusters)) if len(clusters[i]) >= 10]
    idx = np.random.choice(all_idx, 10, replace = False)
    for i in idx:
        cluster = clusters[i]
        rows = np.random.choice(cluster, 10, replace = False)
        centroid_sample_df.append(all_replies_df.iloc[rows])
    centroid_sample_df = pd.concat(centroid_sample_df)
    centroid_sample_df.to_csv('../data/merged_q/all_replies_max_len_40_clusters/samples/centroid_sample_df_{:.2f}.csv'.format(threshold), index = False)

all_posts_df = pd.read_csv('../data/merged_q/all_posts_max_len_40.csv')

with open('../data/merged_q/all_posts_max_len_40_clusters/all_posts_clusters.pickle', 'rb') as f:
    all_posts_clusters = pickle.load(f)

for threshold in thresholds:
    sample_df = []
    clusters = all_posts_clusters[threshold]
    all_idx = [i for i in range(len(clusters)) if len(clusters[i]) >= 10]
    idx = np.random.choice(all_idx, 10, replace = False)
    for i in idx:
        cluster = clusters[i]
        rows = np.random.choice(cluster, 10, replace = False)
        sample_df.append(all_posts_df.iloc[rows])
    sample_df = pd.concat(sample_df)
    sample_df.to_csv('../data/merged_q/all_posts_max_len_40_clusters/samples/sample_df_{:.2f}.csv'.format(threshold), index = False)
