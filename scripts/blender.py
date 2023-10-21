import pickle
import subprocess
import numpy as np
import pandas as pd

test_posts_df = pd.read_csv('../data/test/posts.csv')
# sample_idx = np.load('../commonsense-in-dialogs/data/test/posts_indices_sample_idx.npy')
# sample_test_posts_df = test_posts_df.iloc[sample_idx]

# posts = sample_test_posts_df['text'].tolist()
posts = test_posts_df['text'].tolist()
replies = []

for i in range(len(posts)):
    print(i + 1)
    print(posts[i])

    process = subprocess.Popen(['python3', '/home/yubo/venvs/parlai/lib/python3.8/site-packages/parlai/scripts/interactive.py',
                                '--include-personas', 'False',
                                '-t', 'blended_skill_talk',
                                '-mf', 'zoo:blender/blender_90M/model'],
                               stdin = subprocess.PIPE,
                               stdout = subprocess.PIPE,
                               stderr = subprocess.PIPE)

    input_data = (posts[i] + '\n').encode('utf-8')
    stdout, stderr = process.communicate(input = input_data)
    output_data = stdout.decode('utf-8').split('\n')[-8]
    reply = output_data.split('[TransformerGenerator]: ')[-1]
    print(reply)
    replies.append(reply)

    process.kill()

with open('../data/test/result/all/blender_result.pickle', 'wb') as f:
    pickle.dump(replies, f)
