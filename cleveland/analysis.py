import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

pd.options.display.width = 0  # adjust according to terminal width

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # assuming this file in <proj_root>/src/<package>
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUT_DIR = os.path.join(ROOT_DIR, "out")
DEC = 4

FILE = "val_results.csv"


def hp_results():
    """Analyse results from hyperparameter grid searches"""
    file_path = os.path.join(OUT_DIR, FILE)
    print(f"Using file: {file_path}")
    results = pd.read_csv(file_path).drop_duplicates()  # Results from cross validation
    exp_cols = ['arch.', 'init', 'alpha', 'n_iter', 'reg']  # Things we varied in the experiments
    results['n_iter'] = results['n_iter'].apply(np.log10)

    print("All conditions, mean metrics ordered by mean auc:")  # metrics for all experimental conditions, mean of folds
    avg = results.drop('fold', axis=1).groupby(exp_cols).mean()  # fold is just an index, so don't want the mean
    with pd.option_context('display.max_rows', None):
        print(avg.sort_values('roc_auc', ascending=False))

    avg = avg.droplevel(level=['init', 'n_iter'])
    to_drop = ['tn', 'fp', 'fn', 'tp']
    print(avg.sort_values('roc_auc', ascending=False).drop(to_drop, axis=1)
          .reset_index().round(DEC).head(5).to_markdown(showindex=False))

    print()
    print("Just mean AUC, unstacked along architecture index dimension, so easier to compare conditions")
    avg_auc = avg.sort_values('roc_auc', ascending=False)['roc_auc'].unstack(level=0)
    # Order columns by architecture complexity:
    avg_auc = avg_auc[['1', '2_1', '4_1', '8_1', '16_1', '2_2_1', '2_4_1', '2_8_1', '2_16_1', '4_2_1',
                                     '4_4_1', '4_8_1', '4_16_1', '8_2_1', '8_4_1', '8_8_1', '8_16_1', '16_2_1',
                                     '16_4_1', '16_8_1', '16_16_1']]
    print(avg_auc)

    # Heatmap showing AUC for each of these conditions
    fig, ax = plt.subplots()
    mappable = ax.matshow(avg_auc)
    mappable.set_clim(0.5, 1.0)  # TODO change range for fine grain
    ax.set_xticks(range(avg_auc.shape[1]))
    ax.set_xticklabels(avg_auc.columns)
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_xlabel("Units per layer")
    ax.xaxis.set_label_position('top')

    ax.set_yticks(range(avg_auc.shape[0]))
    ax.set_yticklabels(avg_auc.index)
    ax.set_ylabel("(learning rate, regularisation term)")

    fig.colorbar(mappable, ax=ax)

    print(f"Using file: {file_path}")
    plt.show()


if __name__ == '__main__':
    hp_results()
