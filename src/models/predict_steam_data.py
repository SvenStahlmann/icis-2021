import click
import pandas as pd
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import f1_score, accuracy_score

def load_fold_data(path, current_fold):
    fold_train = pd.read_csv(path + 'fold-' + str(current_fold) + '-train.csv', delimiter=',')
    fold_test = pd.read_csv(path + 'fold-' + str(current_fold) + '-test.csv', delimiter=',')

    return fold_train, fold_test

def load_steam_data(path="../../data/raw/steam-reviews.csv"):
    steam_df = pd.read_csv(path)
    return steam_df


@click.command()
@click.option('--path', default='../../data/processed/', help='path to the training data.')
def main(path, valid_in_cat_path, valid_out_of_cat_path):
    steam_df = load_steam_data()
    i = 1
    print("starting training, using fold " + str(i))

    train, test = load_fold_data(path, i)
    # Train the model using roberta model
    args_dict = {'output_dir': '../../models/roberta-base-bs8-e6-fold' + str(i),
                 'use_cached_eval_features': False,
                 'reprocess_input_data': True,
                 'train_batch_size': 8,
                 'num_train_epochs': 6,
                 'fp16': False,
                 'overwrite_output_dir': True}
    model = ClassificationModel('roberta', 'roberta-base', num_labels=2, args=args_dict)
    model.train_model(train)
    print("done training model fold " + str(i))
    result, model_outputs, wrong_predictions = model.eval_model(test, acc=accuracy_score, f1=f1_score)
    acc = result['acc']
    f1 = result['f1']
    print(f"acc: {acc} , f1: {f1}")

    # Make predictions with the model
    save_path = '../../reports/steam-prediction.csv'
    print("predicting...")
    predictions, raw_outputs = model.predict(steam_df["sentence"].tolist())
    print(f"predicting finished - saved to {save_path}" )
    steam_df['prediction'] = predictions
    steam_df.to_csv(save_path, index=False)

if __name__ == '__main__':
    main()
