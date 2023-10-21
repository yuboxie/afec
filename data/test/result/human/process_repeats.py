import random
import pandas as pd

cand_models_ordered = ['highest_degree', 'random', 'mime', 'cem', 'blender', 'meed']

df = pd.read_csv('data_1.csv')

cand_models = []
for i in range(df.shape[0]):
    model_replies = {}
    for model in cand_models_ordered:
        reply = df.iloc[i][model]
        the_key = ''
        for key in model_replies:
            if reply == model_replies[key]:
                the_key = key
                break
        if the_key != '':
            del model_replies[the_key]
            model_replies[the_key + '-' + model] = reply
        else:
            model_replies[model] = reply
    models = random.sample(list(model_replies.keys()), len(model_replies))
    cand_models.append(','.join(models))

df['cand_models'] = cand_models
df.to_csv('model_replies.csv')
