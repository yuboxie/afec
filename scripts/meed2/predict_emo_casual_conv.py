import time
import numpy as np
import pandas as pd
import tensorflow as tf
from os import mkdir
from os.path import exists
from tqdm import tqdm
from math import ceil
from sklearn.metrics import precision_recall_fscore_support
from pytorch_transformers import RobertaTokenizer

from model_utils import *
from model_emo_pred import EmotionPredictor, loss_function
from datasets import *


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

num_emotions = 41
max_length = 100  # Maximum number of tokens
buffer_size = 300000
batch_size = 1
learning_rate = 5e-5
adam_beta_1 = 0.9
adam_beta_2 = 0.98
adam_epsilon = 1e-6

model_optimal_epoch = {'emo_pred_os': 3, 'emo_pred_os_osed': 2, 'emo_pred_os_ed': 4}

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
        # post = sample_test_posts_df.iloc[i]['text']
        # emotion = sample_test_posts_df.iloc[i]['emotion']
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

def main(model_name):
    optimal_epoch = model_optimal_epoch[model_name]
    checkpoint_path = 'checkpoints/{}'.format(model_name)

    test_dataset, N = create_dataset()

    # Define the model.
    emotion_predictor = EmotionPredictor(num_layers, d_model, num_heads, dff, hidden_act,
        dropout_rate, layer_norm_eps, max_position_embed, type_vocab_size, vocab_size, num_emotions)

    # Build the model.
    build_emo_pred_model(emotion_predictor, max_length, vocab_size)
    print('Model has been built.')

    # Define optimizer and metrics.
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = adam_beta_1, beta_2 = adam_beta_2,
        epsilon = adam_epsilon)

    # Define the checkpoint manager.
    ckpt = tf.train.Checkpoint(model = emotion_predictor, optimizer = optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep = None)

    # Restore from the optimal epoch.
    ckpt.restore(ckpt_manager.checkpoints[optimal_epoch - 1]).expect_partial()
    print('Checkpoint {} restored.'.format(ckpt_manager.checkpoints[optimal_epoch - 1]))

    y_pred = []
    for inputs in tqdm(test_dataset, total = ceil(N / batch_size)):
        inp, inp_seg, inp_emot, _ = inputs
        enc_padding_mask = create_padding_mask(inp)
        pred_emot = emotion_predictor(inp, inp_seg, inp_emot, False, enc_padding_mask)
        pred_emot = np.argmax(pred_emot.numpy(), axis = 1)
        y_pred += pred_emot.tolist()

    print('Saving the prediction results...')
    prediction = {'y_pred': y_pred}
    pd.DataFrame(prediction).to_csv('result/all/meed_emo_pred.csv', index = False)


if __name__ == '__main__':
    main('emo_pred_os_ed')
