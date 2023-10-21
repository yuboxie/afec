import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from os import mkdir
from os.path import join, exists
from pytorch_transformers import RobertaTokenizer
from sklearn.metrics import precision_recall_fscore_support

from optimize import CustomSchedule
from model_utils import *
from model_emobert import EmoBERT, loss_function


# Some hyper-parameters
num_layers = 12
d_model = 768
num_heads = 12
dff = d_model * 4
hidden_act = 'gelu'  # Use 'gelu' or 'relu'
dropout_rate = 0.1
layer_norm_eps = 1e-5
max_position_embed = 514
num_emotions = 41  # Number of emotion categories

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
vocab_size = tokenizer.vocab_size

max_length = 100  # Maximum number of tokens
batch_size = 256
peak_lr = 2e-5
adam_beta_1 = 0.9
adam_beta_2 = 0.98
adam_epsilon = 1e-6

id2emot = {}
with open('ebp_labels.txt', 'r') as f:
    for line in f:
        emot, index = line.strip().split(',')
        id2emot[int(index)] = emot

w = 2 ** np.arange(50)

def create_event_dataset(df_event):
    print('Vocabulary size is {}.'.format(tokenizer.vocab_size))

    SOS_ID = tokenizer.encode('<s>')[0]
    EOS_ID = tokenizer.encode('</s>')[0]

    def create_dataset():
        dialogs = df_event['text'].tolist()
        inputs = np.ones((len(dialogs), max_length), dtype = np.int32)
        weights = np.ones((len(dialogs), max_length), dtype = np.float32)
        for i, dialog in tqdm(enumerate(dialogs), total = len(dialogs)):
            uttrs = [dialog]
            uttr_ids = []
            weight = []
            total_weight = np.sum(w[:len(uttrs)])
            for j in range(len(uttrs)-1, -1, -1):
                encoded = tokenizer.encode(uttrs[j])
                weight += [w[j] / total_weight] * (len(encoded) + 2)
                uttr_ids += [EOS_ID] + encoded + [EOS_ID]
            uttr_ids = uttr_ids[:max_length]
            weight = weight[:max_length]
            uttr_ids[0] = SOS_ID
            uttr_ids[-1] = EOS_ID
            inputs[i,:len(uttr_ids)] = uttr_ids
            weights[i,:len(uttr_ids)] = weight

        print('Created dataset with {} examples.'.format(inputs.shape[0]))

        return inputs, weights

    event_inputs, event_weights = create_dataset()
    event_dataset = (tf.data.Dataset.from_tensor_slices(event_inputs),
        tf.data.Dataset.from_tensor_slices(event_weights))
    event_dataset = tf.data.Dataset.zip(event_dataset).batch(batch_size)

    return event_dataset

def create_reply_dataset(df_reply, df_event):
    print('Vocabulary size is {}.'.format(tokenizer.vocab_size))

    SOS_ID = tokenizer.encode('<s>')[0]
    EOS_ID = tokenizer.encode('</s>')[0]

    id2uttr = {}
    for i in range(df_event.shape[0]):
        id2uttr[df_event.iloc[i]['id']] = df_event.iloc[i]['text']

    def create_dataset():
        dialogs = df_reply['text'].tolist()
        parent_ids = df_reply['parent_id'].tolist()
        inputs = np.ones((len(dialogs), max_length), dtype = np.int32)
        weights = np.ones((len(dialogs), max_length), dtype = np.float32)
        for i, dialog in tqdm(enumerate(dialogs), total = len(dialogs)):
            uttrs = [id2uttr[parent_ids[i]], dialog]
            uttr_ids = []
            weight = []
            total_weight = np.sum(w[:len(uttrs)])
            for j in range(len(uttrs)-1, -1, -1):
                encoded = tokenizer.encode(uttrs[j])
                weight += [w[j] / total_weight] * (len(encoded) + 2)
                uttr_ids += [EOS_ID] + encoded + [EOS_ID]
            uttr_ids = uttr_ids[:max_length]
            weight = weight[:max_length]
            uttr_ids[0] = SOS_ID
            uttr_ids[-1] = EOS_ID
            inputs[i,:len(uttr_ids)] = uttr_ids
            weights[i,:len(uttr_ids)] = weight

        print('Created dataset with {} examples.'.format(inputs.shape[0]))

        return inputs, weights

    reply_inputs, reply_weights = create_dataset()
    reply_dataset = (tf.data.Dataset.from_tensor_slices(reply_inputs),
        tf.data.Dataset.from_tensor_slices(reply_weights))
    reply_dataset = tf.data.Dataset.zip(reply_dataset).batch(batch_size)

    return reply_dataset

def main():
    df_event = pd.read_csv('casual_conv/all_posts_max_len_40.csv')
    df_reply = pd.read_csv('casual_conv/all_replies_max_len_40.csv')
    event_dataset = create_event_dataset(df_event)
    reply_dataset = create_reply_dataset(df_reply, df_event)

    # Define the model.
    emobert = EmoBERT(num_layers, d_model, num_heads, dff, hidden_act, dropout_rate,
        layer_norm_eps, max_position_embed, vocab_size, num_emotions)

    # Define optimizer and metrics.
    learning_rate = peak_lr
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = adam_beta_1, beta_2 = adam_beta_2,
        epsilon = adam_epsilon)

    # Define the checkpoint manager.
    ckpt = tf.train.Checkpoint(model = emobert, optimizer = optimizer)

    checkpoint_path = 'checkpoints/emobert_high_sim_weighted'
    restore_epoch = 5
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep = None)
    ckpt.restore(ckpt_manager.checkpoints[restore_epoch - 1]).expect_partial()
    print('Checkpoint {} restored!!'.format(ckpt_manager.checkpoints[restore_epoch - 1]))

    def predict(dataset, N):
        y_pred = []
        for inputs in tqdm(dataset, total = N // batch_size):
            inp, weights = inputs
            enc_padding_mask = create_masks(inp)
            pred_emot = emobert(inp, weights, False, enc_padding_mask)
            pred_emot = np.argmax(pred_emot.numpy(), axis = 1)
            y_pred += pred_emot.tolist()
        return y_pred

    y_pred_event = predict(event_dataset, df_event.shape[0])
    y_pred_reply = predict(reply_dataset, df_reply.shape[0])

    emot_event = [id2emot[x] for x in y_pred_event]
    emot_reply = [id2emot[x] for x in y_pred_reply]

    df_event['emotion'] = emot_event
    df_reply['emotion'] = emot_reply

    df_event.to_csv('casual_conv/all_posts_max_len_40_labeled.csv', index = False)
    df_reply.to_csv('casual_conv/all_replies_max_len_40_labeled.csv', index = False)


if __name__ == '__main__':
    main()
