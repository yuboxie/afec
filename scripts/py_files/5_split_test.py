#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


all_posts_df = pd.read_csv('../data/merged_q/all_posts_max_len_40.csv')
all_replies_df = pd.read_csv('../data/merged_q/all_replies_max_len_40.csv')
print('all_posts_df shape = {}'.format(all_posts_df.shape))
print('all_replies_df shape = {}'.format(all_replies_df.shape))

all_posts_embed = np.load('../data/merged_q/all_posts_max_len_40_embed.npy')
all_replies_embed = np.load('../data/merged_q/all_replies_max_len_40_embed.npy')
print('all_posts_embed shape = {}'.format(all_posts_embed.shape))
print('all_replies_embed shape = {}'.format(all_replies_embed.shape))


# ## Select out ED Testing Data

ed_test_df = pd.read_csv('../data/ed/raw/test.csv')

ed_test_ids = [x + '_uttr:1' for x in ed_test_df['conv_id'].tolist()]
ed_test_posts_df = all_posts_df[all_posts_df['id'].isin(ed_test_ids)]

ed_test_posts_ids = ed_test_posts_df['id'].tolist()
print(ed_test_posts_ids[:10])
print('len(ed_test_posts_ids) = {}'.format(len(ed_test_posts_ids)))


# ## Select out 10% of Reddit

all_reddit_posts_df = all_posts_df[~all_posts_df['id'].str.startswith('hit:')]
all_reddit_posts_df

N = all_reddit_posts_df.shape[0]
N_test = N // 10
test_indices = np.random.choice(N, N_test, replace = False)
print('len(test_indices) = {}'.format(len(test_indices)))

reddit_test_posts_df = all_reddit_posts_df.iloc[test_indices]
reddit_test_posts_df

reddit_test_posts_ids = reddit_test_posts_df['id'].tolist()
print(reddit_test_posts_ids[:10])
print('len(reddit_test_posts_ids) = {}'.format(len(reddit_test_posts_ids)))


# ## Combine

rows = test_indices.tolist() + list(range(151405, 152680))
print('total number of testing points: {}'.format(len(rows)))

np.save('../data/test/posts_indices.npy', np.array(rows))


# ## Create Test DF

test_posts_df = all_posts_df.iloc[rows]

test_posts_df.to_csv('../data/test/posts.csv', index = False)

test_replies_df = pd.DataFrame()

for post_id in tqdm(test_posts_df['id'].tolist()):
    df = all_replies_df[all_replies_df['parent_id'] == post_id]
    assert df.shape[0] > 0
    test_replies_df = pd.concat([test_replies_df, df])

test_replies_df.to_csv('../data/test/replies.csv', index = False)

all_replies_ids = all_replies_df['id'].tolist()
test_replies_ids = []
for reply_id in tqdm(test_replies_df['id'].tolist()):
    test_replies_ids.append(all_replies_ids.index(reply_id))

print(len(test_replies_ids), len(set(test_replies_ids)))

np.save('../data/test/replies_indices.npy', np.array(test_replies_ids))


# ## Sort

post_indices = np.load('../data/test/post_indices.npy')
post_indices = np.sort(post_indices)
print(post_indices.shape)

test_posts_df = all_posts_df.iloc[post_indices]

np.save('../data/test/posts_indices.npy', post_indices)
test_posts_df.to_csv('../data/test/posts.csv', index = False)


# -----

reply_indices = np.load('../data/test/reply_indices.npy')
reply_indices = np.sort(reply_indices)
print(reply_indices.shape)

test_replies_df = all_replies_df.iloc[reply_indices]

np.save('../data/test/replies_indices.npy', reply_indices)
test_replies_df.to_csv('../data/test/replies.csv', index = False)


# # Generate Test CSV with 41 Labels

all_posts_df = pd.read_csv('../data/merged_q/all_posts_max_len_40_labeled.csv')
all_replies_df = pd.read_csv('../data/merged_q/all_replies_max_len_40_labeled.csv')
print('all_posts_df shape = {}'.format(all_posts_df.shape))
print('all_replies_df shape = {}'.format(all_replies_df.shape))

post_indices = np.load('../data/test/posts_indices.npy')
print(post_indices.shape)

test_posts_df = all_posts_df.iloc[post_indices]

test_posts_df.to_csv('../data/test/posts_labeled_41.csv', index = False)


# ---

reply_indices = np.load('../data/test/replies_indices.npy')
print(reply_indices.shape)

test_replies_df = all_replies_df.iloc[reply_indices]

test_replies_df.to_csv('../data/test/replies_labeled_41.csv', index = False)
