#!/usr/bin/env python
# coding: utf-8

import os
import time
import json
import random
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from datetime import timezone


# # Crawl Submissions from r/CasualConversation

end_date = datetime(2022, 1, 1)
end_timestamp = end_date.replace(tzinfo = timezone.utc).timestamp()
print(end_date, end_timestamp)

start_date = datetime(2021, 5, 1)
start_timestamp = start_date.replace(tzinfo = timezone.utc).timestamp()
print(start_date, start_timestamp)

submission_cols = ['id', 'created_utc', 'title', 'selftext', 'num_comments', 'subreddit', 'subreddit_id']
submission_cols_opt = ['score', 'author_fullname', 'link_flair_text']

submission_dict = {col: [] for col in submission_cols + submission_cols_opt}


def is_json(json_data):
    try:
        json_object = json.loads(json_data)
    except:
        return False
    return True


current_timestamp = end_timestamp
url = 'https://api.pushshift.io/reddit/search/submission/'
while current_timestamp > start_timestamp:
    if (current_timestamp - end_timestamp) % 86400 == 0:
        date_1 = datetime.fromtimestamp(current_timestamp - 86400, timezone.utc)
        date_2 = datetime.fromtimestamp(current_timestamp, timezone.utc)
        print('Crawling from {} to {}...'.format(date_1, date_2))
    params = {
        'subreddit': 'CasualConversation',
        'sort': 'desc',
        'sort_type': 'created_utc',
        'after': str(int(current_timestamp - 3600 - 1)),
        'before': str(int(current_timestamp)),
        'size': '100'
    }
    res = requests.get(url, params = params)
    if is_json(res.text):
        submissions = res.json()['data']
        for submission in submissions:

            all_cols_exist = True
            for col in submission_cols:
                if col not in submission:
                    all_cols_exist = False
                    break
            if not all_cols_exist:
                continue

            for col in submission_cols:
                submission_dict[col].append(submission[col])

            for col in submission_cols_opt:
                if col in submission:
                    submission_dict[col].append(submission[col])
                else:
                    submission_dict[col].append(None)

    current_timestamp = current_timestamp - 3600


print(len(submission_dict['id']), len(set(submission_dict['id'])))

submission_df = pd.DataFrame(submission_dict)

submission_df.to_csv('../data/reddit/raw/casual_conv_submissions_20210501_20211231.csv', index = False)


# # Crawl Comments According to the Submissions

submission_df = pd.read_csv('../data/reddit/raw/casual_conv_submissions_20210501_20211231.csv')

def is_json(json_data):
    try:
        json_object = json.loads(json_data)
    except:
        return False
    return True


comment_ids = []

url = 'https://api.pushshift.io/reddit/submission/comment_ids/{}'.format(submission_df.iloc[33426]['id'])
res = requests.get(url)
res.text

for i in tqdm(range(1956, submission_df.shape[0])):
    submission_id = submission_df.iloc[i]['id']
    url = 'https://api.pushshift.io/reddit/submission/comment_ids/{}'.format(submission_id)
    res = requests.get(url)
    if is_json(res.text):
        comment_ids += res.json()['data']

pd.DataFrame({'id': comment_ids}).to_csv('../data/reddit/raw/casual_conv_comment_ids_20210501_20211231.csv', index = False)


# --------------------


comment_cols = ['id', 'link_id', 'parent_id', 'created_utc', 'body', 'subreddit', 'subreddit_id']
comment_cols_opt = ['score', 'author_fullname']

comment_dict = {col: [] for col in comment_cols + comment_cols_opt}

def crawl_comments(comment_query):
    url = 'https://api.pushshift.io/reddit/comment/search?ids={}'.format(comment_query)
    res = requests.get(url)
    if not is_json(res.text):
        return
    comments = res.json()['data']

    for comment in comments:
        all_cols_exist = True
        for col in comment_cols:
            if col not in comment:
                all_cols_exist = False
                break
        if not all_cols_exist:
            continue

        for col in comment_cols:
            comment_dict[col].append(comment[col])

        for col in comment_cols_opt:
            if col in comment:
                comment_dict[col].append(comment[col])
            else:
                comment_dict[col].append(None)


N = len(comment_ids)
batch_size = 500
num_batches = N // batch_size

for batch in tqdm(range(num_batches)):
    s = batch * batch_size
    t = s + batch_size
    comment_query = ','.join(comment_ids[s:t])
    crawl_comments(comment_query)

if N % batch_size != 0:
    comment_query = ','.join(comment_ids[(num_batches*batch_size):N])
    crawl_comments(comment_query)

comment_df = pd.DataFrame(comment_dict)

comment_df.to_csv('../data/reddit/raw/casual_conv_comments_20210501_20211231.csv', index = False)
