import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

with open('emot_labels.txt', 'r') as f:
    emots = f.read().split(', ')
with open('colors.txt', 'r') as f:
    colors = f.read().split('\n')

df_post = pd.read_csv('../data/merged_q/all_posts_max_len_40_labeled.csv')
df_reply = pd.read_csv('../data/merged_q/all_replies_max_len_40_labeled.csv')

with open('../data/merged_q/all_posts_max_len_40_clusters/all_posts_clusters.pickle', 'rb') as f:
    post_clusters = pickle.load(f)[0.85]

with open('../data/merged_q/all_replies_max_len_40_clusters/all_replies_clusters_combine_1_centroid.pickle', 'rb') as f:
    reply_clusters = pickle.load(f)[0.8]

post_emots_all = df_post['emotion'].tolist()
reply_emots_all = df_reply['emotion'].tolist()

post_cluster_centers = [x[0] for x in post_clusters]
reply_cluster_centers = [x[0] for x in reply_clusters]

post_cluster_ids = set([i for x in post_clusters for i in x])
reply_cluster_ids = set([i for x in reply_clusters for i in x])
post_single_ids = [i for i in range(df_post.shape[0]) if i not in post_cluster_ids]
reply_single_ids = [i for i in range(df_reply.shape[0]) if i not in reply_cluster_ids]

post_emots = [post_emots_all[i].capitalize() for i in post_cluster_centers + post_single_ids]
reply_emots = [reply_emots_all[i].capitalize() for i in reply_cluster_centers + reply_single_ids]

post_emot_cnt = {x: 0 for x in emots}
reply_emot_cnt = {x: 0 for x in emots}

for emot in post_emots:
    post_emot_cnt[emot] += 1
for emot in reply_emots:
    reply_emot_cnt[emot] += 1

for emot in emots:
    post_emot_cnt[emot] /= len(post_emots)
for emot in emots:
    reply_emot_cnt[emot] /= len(reply_emots)

# with open('post_emot_dist.csv', 'w') as f:
#     f.write('Emotion,Probability\n')
#     for k, v in post_emot_cnt.items():
#         f.write('{},{}\n'.format(k, v))
# with open('reply_emot_dist.csv', 'w') as f:
#     f.write('Emotion,Probability\n')
#     for k, v in reply_emot_cnt.items():
#         f.write('{},{}\n'.format(k, v))

# plt.rcParams['figure.figsize'] = (10,1.5)
# plt.rcParams['font.family'] = 'Times New Roman'
# barlist = plt.bar(list(post_emot_cnt.keys()), list(post_emot_cnt.values()))
# for i in range(41):
#     barlist[i].set_color(colors[i])
# axes = plt.gca()
# axes.set_ylim([0, 0.175])
# plt.margins(x = 0.01)
# plt.xticks(rotation = 'vertical')
# plt.savefig('post_emot_dist.pdf', bbox_inches = 'tight')

plt.rcParams['figure.figsize'] = (10,1.5)
plt.rcParams['font.family'] = 'Times New Roman'
barlist = plt.bar(list(reply_emot_cnt.keys()), list(reply_emot_cnt.values()))
for i in range(41):
    barlist[i].set_color(colors[i])
axes = plt.gca()
axes.set_ylim([0, 0.175])
plt.margins(x = 0.01)
plt.xticks(rotation = 'vertical')
plt.savefig('reply_emot_dist.pdf', bbox_inches = 'tight')
