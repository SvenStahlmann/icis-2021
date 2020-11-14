import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import click


def generateCategoryFigure(categories, values_roberta, values_svm_bow, values_svm_tfidf, values_cnn, savepath):
    models = ['RoBERTa', 'CNN', 'SVM - BOW', 'SVM - TF-IDF', ]

    values = np.array([values_roberta, values_cnn, values_svm_bow, values_svm_tfidf])

    c = categories

    fig, ax = plt.subplots(figsize=(18, 4))
    im = ax.imshow(values, cmap="YlGn")

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, cmap="YlGn")
    cbar.ax.set_ylabel(" ", rotation=-90, va="bottom")
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(c)))
    ax.set_yticks(np.arange(len(models)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(c, fontsize=10)
    ax.set_yticklabels(models, fontsize=14)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(models)):
        for j in range(len(values[0])):
            color = "black"
            if i == 0 and j > 20:
                color="white"
            text = ax.text(j, i, "{:.2f}".format(values[i, j]),
                           ha="center", va="center", color=color)

    fig.tight_layout()
    plt.savefig(savepath + 'heatmap.png', dpi=300, bbox_inches = "tight")
    plt.show()


def loadData(path, filter='\-acc$', in_category="Baby"):
    df_roberta = pd.read_csv(path + 'roberta-results.csv', delimiter=',')
    df_cnn = pd.read_csv(path + 'timoshenko-results.csv', delimiter=',')
    df_svm_bow = pd.read_csv(path + 'SVM-bow.csv', delimiter=',')
    df_svm_tfidf = pd.read_csv(path + 'SVM-tfidf.csv', delimiter=',')

    d_roberta = df_roberta.filter(regex=filter).mean().to_dict()
    d_cnn = df_cnn.filter(regex=filter).mean().to_dict()
    d_svm_bow = df_svm_bow.filter(regex=filter).mean().to_dict()
    d_svm_tfidf = df_svm_tfidf.filter(regex=filter).mean().to_dict()

    categories = []
    values_roberta = []
    values_svm_bow = []
    values_svm_tfidf = []
    values_cnn = []
    for key, value in sorted(d_roberta.items(), key=lambda x: x[1]):
        category = re.sub(filter, '', key)
        if category == in_category:
            continue
        categories.append(category)
        values_roberta.append(value * 100)
        values_svm_bow.append(d_svm_bow[key])
        values_svm_tfidf.append(d_svm_tfidf[key])
        values_cnn.append(d_cnn[key] * 100)

    return categories, values_roberta, values_svm_bow, values_svm_tfidf, values_cnn


@click.command()
@click.option('--path', default='../../reports/', help='path to the report data.')
@click.option('--savepath', default='../../reports/figures/', help='path to save the figures.')
def main(path, savepath):
    categories, values_roberta, values_svm_bow, values_svm_tfidf, values_cnn = loadData(path)
    generateCategoryFigure(categories, values_roberta, values_svm_bow, values_svm_tfidf, values_cnn, savepath)


if __name__ == '__main__':
    main()
