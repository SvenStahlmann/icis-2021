import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from simpletransformers.classification import ClassificationModel
import click


def load_data(trainingdata, testdata, validdata):
    train = pd.read_csv(trainingdata, delimiter=',')
    test = pd.read_csv(testdata, delimiter=',')
    valid = pd.read_csv(validdata, delimiter=',')
    return train, test, valid

@click.command()
@click.option('--trainingdata', default='../../data/processed/in-cat-train.csv', help='path to the training data.')
@click.option('--testdata', default='../../data/processed/in-cat-test.csv', help='path to the test data.')
@click.option('--validata', default='../../data/processed/out-of-cat-valid.csv', help='path to the validation data.')
def main(trainingdata, testdata, validata):
    train, test, valid = load_data(trainingdata, testdata, validata)
    print("data loaded")
    print(f"train: {len(train)}, test: {len(test)}, valid: {len(valid)}")

    #load trained roberta model
    args = {'use_cached_eval_features': False,
            'reprocess_input_data': True,
            "train_batch_size": 8,
            "num_train_epochs": 6,
            "fp16": False,
            'overwrite_output_dir': True}
    model = ClassificationModel('roberta', '../../models/roberta-base-bs8-e6/', num_labels=2, args=args)
    #eval the model
    test_result, _, _ = model.eval_model(test, acc=accuracy_score,f1=f1_score)
    valid_result, _, _ = model.eval_model(valid, acc=accuracy_score,f1=f1_score)

    print("test set evaluation")
    print(f"acc: {test_result['acc']}, f1: {test_result['f1']}")
    print("valid set evaluation")
    print(f"acc: {valid_result['acc']}, f1: {valid_result['f1']}")


if __name__ == '__main__':
    main()