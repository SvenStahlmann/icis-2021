import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from simpletransformers.classification import ClassificationModel
import click


def load_training_data(trainingdata, testdata):
    train = pd.read_csv(trainingdata, delimiter=',')
    test = pd.read_csv(testdata, delimiter=',')
    return train, test

@click.command()
@click.option('--trainingdata', default='../../data/processed/in-cat-train.csv', help='path to the training data.')
@click.option('--testdata', default='../../data/processed/in-cat-test.csv', help='path to the test data.')
def main(trainingdata, testdata):
    train, test = load_training_data(trainingdata, testdata)
    print("data loaded")
    print(f"train: {len(train)}, test: {len(test)}")
    # Train the model using roberta model
    args_dict = {"output_dir": "../../models/roberta-base-bs8-e6",
                 'use_cached_eval_features': False,
                 'reprocess_input_data': True,
                 "train_batch_size": 8,
                 "num_train_epochs": 6,
                 "fp16": False,
                 'overwrite_output_dir': True}
    model = ClassificationModel('roberta', 'roberta-base', num_labels=2, args=args_dict)
    model.train_model(train)
    result, model_outputs, wrong_predictions = model.eval_model(test, acc=accuracy_score)
    print("test set evaluation")
    print(f"acc: {result['acc']}")


if __name__ == '__main__':
    main()