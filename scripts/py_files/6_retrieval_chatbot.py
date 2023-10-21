#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


# # Read Data

all_posts_df = pd.read_csv('../data/merged_q/all_posts_max_len_40_labeled.csv')
all_replies_df = pd.read_csv('../data/merged_q/all_replies_max_len_40_labeled.csv')
print('all_posts_df shape = {}'.format(all_posts_df.shape))
print('all_replies_df shape = {}'.format(all_replies_df.shape))

all_posts_embed = np.load('../data/merged_q/all_posts_max_len_40_embed.npy')
all_replies_embed = np.load('../data/merged_q/all_replies_max_len_40_embed.npy')
print('all_posts_embed shape = {}'.format(all_posts_embed.shape))
print('all_replies_embed shape = {}'.format(all_replies_embed.shape))

with open('../data/merged_q/all_posts_max_len_40_clusters/all_posts_clusters.pickle', 'rb') as f:
    all_posts_clusters = pickle.load(f)
print('Num of post clusters with threshold 0.85 = {}'.format(len(all_posts_clusters[0.85])))

with open('../data/merged_q/all_replies_max_len_40_clusters/all_replies_clusters_combine_1_centroid.pickle', 'rb') as f:
    all_replies_clusters = pickle.load(f)
print('Num of reply clusters with threshold 0.80 = {}'.format(len(all_replies_clusters[0.8])))

test_posts_indices = np.load('../data/test/posts_indices.npy')
test_posts_df = pd.read_csv('../data/test/posts_labeled_41.csv')
assert test_posts_indices.shape[0] == test_posts_df.shape[0]

test_replies_indices = np.load('../data/test/replies_indices.npy')
test_replies_df = pd.read_csv('../data/test/replies_labeled_41.csv')
assert test_replies_indices.shape[0] == test_replies_df.shape[0]


# # Preprocess

test_posts_embed = all_posts_embed[test_posts_indices]
print(test_posts_embed.shape)

other_posts_indices = set(range(all_posts_df.shape[0])) - set(test_posts_indices)
print(len(other_posts_indices))

minion_posts_indices = set()
for c in all_posts_clusters[0.85]:
    minion_posts_indices = minion_posts_indices.union(set(c[1:]))
print(len(minion_posts_indices))

other_posts_indices_c = np.sort(list(other_posts_indices - minion_posts_indices))
print(other_posts_indices_c.shape)

other_posts_embed = all_posts_embed[other_posts_indices_c]
print(other_posts_embed.shape)

all_posts_clusters_dict = {}
for c in all_posts_clusters[0.85]:
    all_posts_clusters_dict[c[0]] = c

reply_degrees = {reply_id: 1 for reply_id in all_replies_df['id'].tolist()}
for c in tqdm(all_replies_clusters[0.8]):
    assert len(c) >= 2
    for i in c:
        reply_degrees[all_replies_df.iloc[i]['id']] = len(c)


# # Calculate Similarity

# cos_sim = cosine_similarity(test_posts_embed, other_posts_embed)
# np.save('../data/test/result/posts_cos_sim.npy', cos_sim)
cos_sim = np.load('../data/test/result/posts_cos_sim.npy')

# cos_sim_top_idx = np.argsort(cos_sim, axis = 1)[:,-1]
# np.save('../data/test/result/posts_cos_sim_top_idx.npy', cos_sim_top_idx)
cos_sim_top_idx = np.load('../data/test/result/posts_cos_sim_top_idx.npy')

# cos_sim_top_idx_orig = other_posts_indices_c[cos_sim_top_idx]
# np.save('../data/test/result/posts_cos_sim_top_idx_orig.npy', cos_sim_top_idx_orig)
cos_sim_top_idx_orig = np.load('../data/test/result/posts_cos_sim_top_idx_orig.npy')


# # Generate All Responses

print(test_posts_df.iloc[3]['text'])
print(all_posts_df.iloc[cos_sim_top_idx_orig[3]]['text'])

result_dict = {}
result_cols = ['post_id', 'post', 'most_sim_post_id', 'most_sim_post_emo', 'most_sim_post', 'cos_sim',
               'highest_degree_id', 'highest_degree', 'random_id', 'random',
               'intent_id', 'intent_emo', 'intent',
               'follow_id', 'follow_emo', 'follow']
for col in result_cols:
    result_dict[col] = []

emp_intents = ['agreeing', 'acknowledging', 'encouraging', 'consoling',
               'sympathizing', 'suggesting', 'questioning', 'wishing']

emo_groups = [['prepared', 'confident', 'proud'], ['content', 'hopeful', 'anticipating'],
              ['joyful', 'excited'], ['caring'], ['faithful', 'trusting', 'grateful'],
              ['jealous', 'annoyed', 'angry', 'furious'], ['terrified', 'afraid', 'anxious', 'apprehensive'],
              ['disgusted'], ['ashamed', 'guilty', 'embarrassed'],
              ['devastated', 'sad', 'disappointed', 'nostalgic', 'lonely'],
              ['surprised'], ['impressed'], ['sentimental'], ['neutral'], ['agreeing', 'acknowledging'],
              ['encouraging'], ['consoling', 'sympathizing'], ['suggesting'], ['questioning'], ['wishing']]
emo_group_dict = {}
for g in emo_groups:
    for emo in g:
        emo_group_dict[emo] = [x for x in g if x != emo]
emo_group_dict

for i in tqdm(range(test_posts_df.shape[0])):
    post_id = test_posts_df.iloc[i]['id']
    post = test_posts_df.iloc[i]['text']
    most_sim_post_idx = cos_sim_top_idx_orig[i]
    most_sim_post_id = all_posts_df.iloc[most_sim_post_idx]['id']
    most_sim_post_emo = all_posts_df.iloc[most_sim_post_idx]['emotion']
    most_sim_post = all_posts_df.iloc[most_sim_post_idx]['text']
    cos_sim_val = cos_sim[i,cos_sim_top_idx[i]]
    
    result_dict['post_id'].append(post_id)
    result_dict['post'].append(post)
    result_dict['most_sim_post_id'].append(most_sim_post_id)
    result_dict['most_sim_post_emo'].append(most_sim_post_emo)
    result_dict['most_sim_post'].append(most_sim_post)
    result_dict['cos_sim'].append(cos_sim_val)

    if most_sim_post_idx in all_posts_clusters_dict:
        posts_idx = all_posts_clusters_dict[most_sim_post_idx]
    else:
        posts_idx = [most_sim_post_idx]

    posts_ids = all_posts_df.iloc[posts_idx]['id'].tolist()
    cand_replies_df = all_replies_df[all_replies_df['parent_id'].isin(posts_ids)].copy()

    cand_replies_df['degree'] = [reply_degrees[cand_replies_df.iloc[j]['id']] for j in range(cand_replies_df.shape[0])]
    
    # Highest degree and random
    cand_replies_df = cand_replies_df.sort_values(by = 'degree', ascending = False)
    
    d = cand_replies_df.iloc[0]['degree']
    j = 1
    while j < cand_replies_df.shape[0]:
        if cand_replies_df.iloc[j]['degree'] < d:
            break
        j += 1

    chosen_idx_hd = np.random.choice(j, 1, replace = False)[0]
    chosen_idx_rand = np.random.choice(cand_replies_df.shape[0], 1, replace = False)[0]

    result_dict['highest_degree_id'].append(cand_replies_df.iloc[chosen_idx_hd]['id'])
    result_dict['highest_degree'].append(cand_replies_df.iloc[chosen_idx_hd]['text'])
    result_dict['random_id'].append(cand_replies_df.iloc[chosen_idx_rand]['id'])
    result_dict['random'].append(cand_replies_df.iloc[chosen_idx_rand]['text'])
    
    # Empathetic intents and follow emotions
    cand_replies_intent_df = cand_replies_df[cand_replies_df['emotion'].isin(emp_intents)]
    if cand_replies_intent_df.shape[0] != 0:
        chosen_idx = np.random.choice(cand_replies_intent_df.shape[0], 1, replace = False)[0]
        reply_intent = cand_replies_intent_df.iloc[chosen_idx]['text']
        emo_intent = cand_replies_intent_df.iloc[chosen_idx]['emotion']
        id_intent = cand_replies_intent_df.iloc[chosen_idx]['id']
    else:
        chosen_idx = np.random.choice(cand_replies_df.shape[0], 1, replace = False)[0]
        reply_intent = cand_replies_df.iloc[chosen_idx]['text']
        emo_intent = cand_replies_df.iloc[chosen_idx]['emotion']
        id_intent = cand_replies_df.iloc[chosen_idx]['id']
    
    cand_replies_same_emo_df = cand_replies_df[cand_replies_df['emotion'] == most_sim_post_emo]
    cand_replies_sim_emo_df = cand_replies_df[cand_replies_df['emotion'].isin(emo_group_dict[most_sim_post_emo])]
    if cand_replies_same_emo_df.shape[0] != 0:
        chosen_idx = np.random.choice(cand_replies_same_emo_df.shape[0], 1, replace = False)[0]
        reply_follow = cand_replies_same_emo_df.iloc[chosen_idx]['text']
        emo_follow = cand_replies_same_emo_df.iloc[chosen_idx]['emotion']
        id_follow = cand_replies_same_emo_df.iloc[chosen_idx]['id']
    elif cand_replies_sim_emo_df.shape[0] != 0:
        chosen_idx = np.random.choice(cand_replies_sim_emo_df.shape[0], 1, replace = False)[0]
        reply_follow = cand_replies_sim_emo_df.iloc[chosen_idx]['text']
        emo_follow = cand_replies_sim_emo_df.iloc[chosen_idx]['emotion']
        id_follow = cand_replies_sim_emo_df.iloc[chosen_idx]['id']
    else:
        chosen_idx = np.random.choice(cand_replies_df.shape[0], 1, replace = False)[0]
        reply_follow = cand_replies_df.iloc[chosen_idx]['text']
        emo_follow = cand_replies_df.iloc[chosen_idx]['emotion']
        id_follow = cand_replies_df.iloc[chosen_idx]['id']

    result_dict['intent_id'].append(id_intent)
    result_dict['intent_emo'].append(emo_intent)
    result_dict['intent'].append(reply_intent)
    result_dict['follow_id'].append(id_follow)
    result_dict['follow_emo'].append(emo_follow)
    result_dict['follow'].append(reply_follow)

result_df = pd.DataFrame(result_dict)

result_df.to_csv('../data/test/result/all/retrieval_result.csv', index = False)


# # Generate Sample Responses

# N = 100
# sample_indices = np.random.choice(len(test_posts_indices), N, replace = False)
# np.save('../data/test/posts_indices_sample_idx.npy', sample_indices)
sample_indices = np.load('../data/test/posts_indices_sample_idx.npy')

print(test_posts_df.iloc[sample_indices[3]]['text'])
print(all_posts_df.iloc[cos_sim_top_idx_orig[sample_indices[3]]]['text'])

sample_result_dict = {}
sample_result_cols = ['post_id', 'post', 'most_sim_post_id', 'most_sim_post', 'cos_sim',
                      'highest_degree', 'hd_len', 'random', 'r_len']
for col in sample_result_cols:
    sample_result_dict[col] = []


# ## Rule 1&2: Pick Replies with Highest Degree & Pick Randomly

for i in tqdm(sample_indices):
    post_id = test_posts_df.iloc[i]['id']
    post = test_posts_df.iloc[i]['text']
    most_sim_post_idx = cos_sim_top_idx_orig[i]
    most_sim_post_id = all_posts_df.iloc[most_sim_post_idx]['id']
    most_sim_post = all_posts_df.iloc[most_sim_post_idx]['text']
    cos_sim_val = cos_sim[i,cos_sim_top_idx[i]]
    
    sample_result_dict['post_id'].append(post_id)
    sample_result_dict['post'].append(post)
    sample_result_dict['most_sim_post_id'].append(most_sim_post_id)
    sample_result_dict['most_sim_post'].append(most_sim_post)
    sample_result_dict['cos_sim'].append(cos_sim_val)

    if most_sim_post_idx in all_posts_clusters_dict:
        posts_idx = all_posts_clusters_dict[most_sim_post_idx]
    else:
        posts_idx = [most_sim_post_idx]

    posts_ids = all_posts_df.iloc[posts_idx]['id'].tolist()
    cand_replies_df = all_replies_df[all_replies_df['parent_id'].isin(posts_ids)].copy()

    cand_replies_df['degree'] = [reply_degrees[cand_replies_df.iloc[j]['id']] for j in range(cand_replies_df.shape[0])]
    
    cand_replies_df = cand_replies_df.sort_values(by = 'degree', ascending = False)
    
    d = cand_replies_df.iloc[0]['degree']
    j = 1
    while j < cand_replies_df.shape[0]:
        if cand_replies_df.iloc[j]['degree'] < d:
            break
        j += 1

    chosen_idx = np.random.choice(j, 1, replace = False)[0]
    chosen_idx_rand = np.random.choice(cand_replies_df.shape[0], 1, replace = False)[0]

    sample_result_dict['highest_degree'].append(cand_replies_df.iloc[chosen_idx]['text'])
    sample_result_dict['hd_len'].append(cand_replies_df.iloc[chosen_idx]['length'])
    sample_result_dict['random'].append(cand_replies_df.iloc[chosen_idx_rand]['text'])
    sample_result_dict['r_len'].append(cand_replies_df.iloc[chosen_idx_rand]['length'])


sample_result_df = pd.DataFrame(sample_result_dict)
sample_result_df.to_csv('../data/test/result/retrieval_result_sample.csv', index = False)


# ## Rule 3&4: Based on Emotion/Intent Labels

emp_intents = ['agreeing', 'acknowledging', 'encouraging', 'consoling',
               'sympathizing', 'suggesting', 'questioning', 'wishing']

emo_groups = [['prepared', 'confident', 'proud'], ['content', 'hopeful', 'anticipating'],
              ['joyful', 'excited'], ['caring'], ['faithful', 'trusting', 'grateful'],
              ['jealous', 'annoyed', 'angry', 'furious'], ['terrified', 'afraid', 'anxious', 'apprehensive'],
              ['disgusted'], ['ashamed', 'guilty', 'embarrassed'],
              ['devastated', 'sad', 'disappointed', 'nostalgic', 'lonely'],
              ['surprised'], ['impressed'], ['sentimental'], ['neutral'], ['agreeing', 'acknowledging'],
              ['encouraging'], ['consoling', 'sympathizing'], ['suggesting'], ['questioning'], ['wishing']]
emo_group_dict = {}
for g in emo_groups:
    for emo in g:
        emo_group_dict[emo] = [x for x in g if x != emo]

sample_result_dict = {}
sample_result_cols = ['post_id', 'post', 'most_sim_post_id', 'most_sim_post', 'most_sim_post_emo', 'cos_sim',
                      'intent_emo', 'intent', 'follow_emo', 'follow']
for col in sample_result_cols:
    sample_result_dict[col] = []

for i in tqdm(sample_indices):
    post_id = test_posts_df.iloc[i]['id']
    post = test_posts_df.iloc[i]['text']
    most_sim_post_idx = cos_sim_top_idx_orig[i]
    most_sim_post_id = all_posts_df.iloc[most_sim_post_idx]['id']
    most_sim_post = all_posts_df.iloc[most_sim_post_idx]['text']
    most_sim_post_emo = all_posts_df.iloc[most_sim_post_idx]['emotion']
    cos_sim_val = cos_sim[i,cos_sim_top_idx[i]]
    
    sample_result_dict['post_id'].append(post_id)
    sample_result_dict['post'].append(post)
    sample_result_dict['most_sim_post_id'].append(most_sim_post_id)
    sample_result_dict['most_sim_post'].append(most_sim_post)
    sample_result_dict['most_sim_post_emo'].append(most_sim_post_emo)
    sample_result_dict['cos_sim'].append(cos_sim_val)

    # Find other posts in the same cluster as the most similar post
    if most_sim_post_idx in all_posts_clusters_dict:
        posts_idx = all_posts_clusters_dict[most_sim_post_idx]
    else:
        posts_idx = [most_sim_post_idx]

    posts_ids = all_posts_df.iloc[posts_idx]['id'].tolist()
    cand_replies_df = all_replies_df[all_replies_df['parent_id'].isin(posts_ids)].copy()

    cand_replies_intent_df = cand_replies_df[cand_replies_df['emotion'].isin(emp_intents)]
    if cand_replies_intent_df.shape[0] != 0:
        chosen_idx = np.random.choice(cand_replies_intent_df.shape[0], 1, replace = False)[0]
        reply_intent = cand_replies_intent_df.iloc[chosen_idx]['text']
        emo_intent = cand_replies_intent_df.iloc[chosen_idx]['emotion']
    else:
        chosen_idx = np.random.choice(cand_replies_df.shape[0], 1, replace = False)[0]
        reply_intent = cand_replies_df.iloc[chosen_idx]['text']
        emo_intent = cand_replies_df.iloc[chosen_idx]['emotion']
    
    cand_replies_same_emo_df = cand_replies_df[cand_replies_df['emotion'] == most_sim_post_emo].copy()
    cand_replies_sim_emo_df = cand_replies_df[cand_replies_df['emotion'].isin(emo_group_dict[most_sim_post_emo])].copy()
    if cand_replies_same_emo_df.shape[0] != 0:
        chosen_idx = np.random.choice(cand_replies_same_emo_df.shape[0], 1, replace = False)[0]
        reply_follow = cand_replies_same_emo_df.iloc[chosen_idx]['text']
        emo_follow = cand_replies_same_emo_df.iloc[chosen_idx]['emotion']
    elif cand_replies_sim_emo_df.shape[0] != 0:
        chosen_idx = np.random.choice(cand_replies_sim_emo_df.shape[0], 1, replace = False)[0]
        reply_follow = cand_replies_sim_emo_df.iloc[chosen_idx]['text']
        emo_follow = cand_replies_sim_emo_df.iloc[chosen_idx]['emotion']
    else:
        chosen_idx = np.random.choice(cand_replies_df.shape[0], 1, replace = False)[0]
        reply_follow = cand_replies_df.iloc[chosen_idx]['text']
        emo_follow = cand_replies_df.iloc[chosen_idx]['emotion']

    sample_result_dict['intent_emo'].append(emo_intent)
    sample_result_dict['intent'].append(reply_intent)
    sample_result_dict['follow_emo'].append(emo_follow)
    sample_result_dict['follow'].append(reply_follow)

sample_result_df = pd.DataFrame(sample_result_dict)

sample_result_df.to_csv('../data/test/result/sample/retrieval_result_2.csv', index = False)
