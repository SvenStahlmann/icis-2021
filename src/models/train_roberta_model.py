import click
import pandas as pd
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import f1_score, accuracy_score


def load_validation_data(valid_in_cat_path, valid_out_of_cat_path):
    valid_in_cat = pd.read_csv(valid_in_cat_path, delimiter=',')
    valid_out_of_cat = pd.read_csv(valid_out_of_cat_path, delimiter=',')

    return valid_in_cat, valid_out_of_cat


def load_fold_data(path, current_fold):
    fold_train = pd.read_csv(path + 'fold-' + str(current_fold) + '-train.csv', delimiter=',')
    fold_test = pd.read_csv(path + 'fold-' + str(current_fold) + '-test.csv', delimiter=',')

    return fold_train, fold_test


@click.command()
@click.option('--path', default='../../data/processed/', help='path to the training data.')
@click.option('--valid_in_cat_path', default='../../data/processed/in-cat-test.csv', help='path to the training data.')
@click.option('--valid_out_of_cat_path', default='../../data/processed/out-of-cat-valid.csv',
              help='path to the training data.')
def main(path, valid_in_cat_path, valid_out_of_cat_path):
    report_df = []
    valid_in_cat, valid_out_of_cat = load_validation_data(valid_in_cat_path, valid_out_of_cat_path)
    print("validation data loaded")
    print(f"in cat: {len(valid_in_cat)}, out of cat: {len(valid_out_of_cat)}")
    print("starting fold training")
    for i in range(10):
        name = 'fold-' + str(i)
        print("working on " + name)
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
        in_cat_result, _, _ = model.eval_model(valid_in_cat, acc=accuracy_score, f1=f1_score)
        out_of_cat_result, _, _ = model.eval_model(valid_out_of_cat, acc=accuracy_score, f1=f1_score)
        acc_score_in_cat = in_cat_result['acc']
        f1_score_in_cat = in_cat_result['f1']
        acc_score_out_of_cat = out_of_cat_result['acc']
        f1_score_out_of_cat = out_of_cat_result['f1']

        report_df.append([name, acc_score_in_cat, f1_score_in_cat, acc_score_out_of_cat, f1_score_out_of_cat])

    report_df = pd.DataFrame(report_df, columns=['name', 'acc_in_cat', 'f1_in_cat', 'acc_out_of_cat', 'f1_out_of_cat'])

    print("evaluation in cat")
    print(f"acc: {report_df['acc_in_cat'].mean()}, f1: {report_df['f1_in_cat'].mean()}")

    print("evaluation out of cat")
    print(f"acc: {report_df['acc_out_of_cat'].mean()}, f1: {report_df['f1_out_of_cat'].mean()}")

    report_df.to_csv('../../reports/roberta-results.csv', index=False)


if __name__ == '__main__':
    main()
