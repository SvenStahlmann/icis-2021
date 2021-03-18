import click
import pandas as pd
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from data_helpers import *

DEFAULT_ARGS = {'output_dir': '../../models/roberta-base-bs8-e6',
                 'use_cached_eval_features': False,
                 'reprocess_input_data': True,
                 'train_batch_size': 8,
                 'num_train_epochs': 6,
                 'fp16': False,
                 'overwrite_output_dir': True}


def train(train_df: pd.DataFrame, args = DEFAULT_ARGS):
    model = ClassificationModel('roberta', 'roberta-base', num_labels=2, args=args)
    model.train_model(train_df)

    return model

def predict(list, model):
    prediction, raw_outputs = model.predict(list)
    return prediction
