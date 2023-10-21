import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from os.path import join, exists, split
from os import makedirs
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

batch_size = 8
model_name = 'roberta-base'
restore_path = 'checkpoints/roberta_base_5e-6'
optimal_epochs = {'roberta-base': 5}
restore_epoch = optimal_epochs[model_name]

emos = ['surprised', 'excited', 'annoyed', 'proud', 'angry', 'sad', 'grateful', 'lonely',
        'impressed', 'afraid', 'disgusted', 'confident', 'terrified', 'hopeful', 'anxious', 'disappointed',
        'joyful', 'prepared', 'guilty', 'furious', 'nostalgic', 'jealous', 'anticipating', 'embarrassed',
        'content', 'devastated', 'sentimental', 'caring', 'trusting', 'ashamed', 'apprehensive', 'faithful']

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, size):
        self.encodings = encodings
        self.size = size

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return self.size

def create_test_dataset(data_path, tokenizer):
    print('Creating the test dataset...')
    df_test = pd.read_csv(join(data_path, 'posts.csv'))
    def encode(df):
        texts = df['text'].tolist()
        encodings = tokenizer(texts, truncation = True, padding = True)
        dataset = Dataset(encodings, len(texts))
        return dataset
    test_dataset = encode(df_test)
    return test_dataset

if __name__ == '__main__':
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    test_dataset = create_test_dataset('../data/test', tokenizer)

    assert torch.cuda.is_available()
    device = torch.device('cuda')
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels = 32)
    state_dict = torch.load('{}/epoch_{:02d}.bin'.format(restore_path, restore_epoch))
    state_dict = {k.replace('module.', ''):v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size = batch_size)

    y_pred = []
    for batch in tqdm(test_loader, total = len(test_dataset) // batch_size):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask = attention_mask)
        pred = outputs['logits'].argmax(1)
        y_pred += pred.cpu().numpy().tolist()

    df_test = pd.read_csv('../data/test/posts.csv')
    assert df_test.shape[0] == len(y_pred)
    y_pred_emos = [emos[i] for i in y_pred]
    df_test['emotion'] = y_pred_emos

    # Replace with ED true labels
    df_ed = pd.read_csv('../data/ed/raw/test.csv')
    for i in tqdm(range(df_test.shape[0])):
        post_id = df_test.iloc[i]['id']
        if post_id.startswith('hit:'):
            conv_id = post_id[:-7]
            df = df_ed[df_ed['conv_id'] == conv_id]
            df_test.iloc[i]['emotion'] = df.iloc[0]['context']

    df_test.to_csv('../data/test/posts_labeled_32.csv', index = False)
