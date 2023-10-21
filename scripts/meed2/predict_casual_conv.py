import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from os import mkdir
from os.path import exists
from math import ceil
from pytorch_transformers import RobertaTokenizer

from model_utils import *
from model import MEED
from datasets import *
from beam_search import beam_search


# Some hyper-parameters
num_layers = 4
d_model = 300
num_heads = 6
dff = d_model * 4
hidden_act = 'gelu'  # Use 'gelu' or 'relu'
dropout_rate = 0.1
layer_norm_eps = 1e-5
max_position_embed = 102
type_vocab_size = 2  # Segments

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
vocab_size = tokenizer.vocab_size
SOS_ID = tokenizer.encode('<s>')[0]
EOS_ID = tokenizer.encode('</s>')[0]

beam_width = 32
alpha = 1.0  # Decoding length normalization coefficient
n_gram = 4  # n-gram repeat blocking in beam search

num_emotions = 41
max_length = 100  # Maximum number of tokens
buffer_size = 300000
batch_size = 1  # For prediction, we always use batch size 1.
learning_rate = 5e-5
adam_beta_1 = 0.9
adam_beta_2 = 0.98
adam_epsilon = 1e-6

model_optimal_epoch = {'meed_os': 50, 'meed_os_osed': 6, 'meed_os_ed': 10}

with open('casual_conv_test/ebp_labels.txt', 'r') as f:
    emo2id = {}
    id2emo = {}
    for line in f:
        items = line.strip().split(',')
        emo2id[items[0]] = int(items[1])
        id2emo[int(items[1])] = items[0]


def create_dataset():
    test_posts_df = pd.read_csv('casual_conv_test/posts_labeled_41.csv')
    # sample_idx = np.load('casual_conv_test/posts_indices_sample_idx.npy')
    # sample_test_posts_df = test_posts_df.iloc[sample_idx]
    # N = sample_test_posts_df.shape[0]
    N = test_posts_df.shape[0]

    # RoBERTa uses 1 as the padding value
    inputs = np.ones((N, max_length), dtype = np.int32)
    input_segments = np.ones((N, max_length), dtype = np.int32)
    input_emots = np.zeros((N, max_length), dtype = np.int32)
    target_segments = np.ones((N, max_length), dtype = np.int32)

    for i in tqdm(range(N)):
        post = test_posts_df.iloc[i]['text']
        emotion = test_posts_df.iloc[i]['emotion']
        uttr_ids = tokenizer.encode(post)
        inp_ids = [SOS_ID] + uttr_ids + [EOS_ID]
        inp_seg_ids = [0] * (len(uttr_ids) + 2)
        emo_id = emo2id[emotion]
        inp_emots = [emo_id] * (len(uttr_ids) + 2)
        inputs[i,:len(inp_ids)] = inp_ids
        input_segments[i,:len(inp_seg_ids)] = inp_seg_ids
        input_emots[i,:len(inp_ids)] = inp_emots

    test_dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(inputs),
        tf.data.Dataset.from_tensor_slices(input_segments),
        tf.data.Dataset.from_tensor_slices(input_emots),
        tf.data.Dataset.from_tensor_slices(target_segments))).batch(batch_size)

    return test_dataset, N

def evaluate(meed, inp, inp_seg, inp_emot, pred_tar_emot, tar_seg):
    enc_padding_mask = create_padding_mask(inp)
    enc_output = meed.encode(inp, inp_seg, inp_emot, False, enc_padding_mask)

    def iter_func(dec_inp, bw):
        enc_output_tiled = tf.tile(enc_output, [bw, 1, 1])
        dec_padding_mask = tf.tile(enc_padding_mask, [bw, 1, 1, 1])

        look_ahead_mask = create_look_ahead_mask(tf.shape(dec_inp)[1])
        dec_target_padding_mask = create_padding_mask(dec_inp)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        dec_inp_seg = tf.ones_like(dec_inp) * tar_seg[0,0]
        pred_tar_emot_tiled = tf.constant([pred_tar_emot] * dec_inp.shape[0])

        pred, attention_weights = meed.decode(enc_output_tiled, pred_tar_emot_tiled, dec_inp,
            dec_inp_seg, False, combined_mask, dec_padding_mask)
        return pred.numpy()

    result_seqs, log_probs = beam_search(iter_func, beam_width, max_length - 1, SOS_ID, EOS_ID, alpha, n_gram)

    return result_seqs, log_probs

def main(model_name):
    optimal_epoch = model_optimal_epoch[model_name]
    checkpoint_path = 'checkpoints/{}'.format(model_name)
    pred_emot_path = 'result/all/meed_emo_pred.csv'

    test_dataset, N = create_dataset()

    # Define the model.
    meed = MEED(num_layers, d_model, num_heads, dff, hidden_act, dropout_rate,
        layer_norm_eps, max_position_embed, type_vocab_size, vocab_size, num_emotions)

    # Build the model.
    build_meed_model(meed, max_length, vocab_size)
    print('Model has been built.')

    # Define optimizer and metrics.
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = adam_beta_1, beta_2 = adam_beta_2,
        epsilon = adam_epsilon)

    # Define the checkpoint manager.
    ckpt = tf.train.Checkpoint(model = meed, optimizer = optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep = None)

    # Restore from the optimal_epoch.
    ckpt.restore(ckpt_manager.checkpoints[optimal_epoch - 1]).expect_partial()
    print('Checkpoint {} restored!!'.format(ckpt_manager.checkpoints[optimal_epoch - 1]))


    pred_emot_df = pd.read_csv(pred_emot_path)
    print('pred_emot_df.shape = {}'.format(pred_emot_df.shape))

    pred_ys = []

    for (i, inputs) in tqdm(enumerate(test_dataset), total = ceil(N / batch_size)):
        inp, inp_seg, inp_emot, tar_seg = inputs
        pred_emot = pred_emot_df.iloc[i]['y_pred']

        tar_preds, log_probs = evaluate(meed, inp, inp_seg, inp_emot, pred_emot, tar_seg)
        tar_pred_dec = tokenizer.decode(tar_preds[0])  # top candidate of beam search
        pred_y = tar_pred_dec[0].strip() if len(tar_pred_dec) > 0 else ''
        pred_ys.append(pred_y)

    print('Saving the prediction results...')
    with open('result/all/meed_result.pickle', 'wb') as f:
        pickle.dump(pred_ys, f)

if __name__ == '__main__':
    main('meed_os_ed')
