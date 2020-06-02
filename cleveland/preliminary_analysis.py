import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
import os
from cleveland import preproc

pd.options.display.width = 0  # adjust according to terminal width
DEC = 3  # decimal places for rounding

# ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # assuming this file in <proj_root>/src
# DATA_DIR = os.path.join(ROOT_DIR, "")
# OUT_DIR = os.path.join(ROOT_DIR, "")
OUT_DIR = os.getcwd()

FILE = "processed.cleveland.data.csv"
NAMES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca",
         "thal", "num"]
CATEGORICAL = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
QUANTITATIVE = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
MISSING_MARKER = "?"  # used in input data to denote missing data
FULL_NAMES = {
    'age': "Age in years",
    'sex': "Sex",
    'cp': "Chest pain type",
    'trestbps': "Resting blood pressure",
    'chol': "Serum cholestoral",
    'fbs': "Fasting blood sugar ",
    'restecg': "Resting electrocardiography",
    'thalach': "Max. heart rate ",
    'exang': "Exercise induced angina",
    'oldpeak': "Exercise-induced ST depression",
    'slope': "Slope of the peak exercise ST segment",
    'ca': "Number of major vessels",
    'thal': "Thallium scan result",
}
GROUND_TRUTH_LABEL = 'disease'
DESCRIPTION = "Identifying features with potential predictive value for heart disease"


# TODO deprecated
def plot_corrmat(df, method='kendall'):
    """ Produce correlation matrix plot """
    corr_mat_plot = df.corr(method=method)
    for i in range(len(df.columns)):
        for j in range(len(df.columns)):
            if i <= j or df.columns[i].split("-")[0] == df.columns[j].split("-")[0]:
                corr_mat_plot.iloc[i, j] = 0
    plt.matshow(corr_mat_plot, cmap='bwr', vmin=-1, vmax=1, origin="lower")
    plt.colorbar()
    # plt.title(f"Correlation: {method}\n\n")
    plt.xticks(range(df.shape[1]), df.columns, rotation=90)
    plt.yticks(range(df.shape[1]), df.columns)
    # plt.savefig(os.path.join(OUT_DIR, f"corr_mat_{METHOD}.png"), dpi=100)
    plt.show()


# TODO deprecated
def quant_hist(df, n_bins=20):
    """Quantitative features plot, both classes combined"""
    fig, axes = plt.subplots(3, 2, figsize=(5, 10))
    for feat, i, j in zip(QUANTITATIVE, [0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]):
        axes[i, j].hist(df[feat], alpha=1.0, density=True, bins=n_bins, color="b")
        axes[i, j].set_title(f"{feat}")
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.show()


def quant_hist_compare(df, label, n_bins=20):
    """Plot frequency distributions for quantitative features"""
    fig, axes = plt.subplots(3, 2, figsize=(7, 9))
    disease = df[df[label] == 1]
    no_disease = df[df[label] == 0]
    for feat, i, j, letter in zip(QUANTITATIVE, [0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1], list("ABCDEF")):
        axes[i, j].hist(no_disease[feat], alpha=0.5, density=True, bins=n_bins, color="b")
        axes[i, j].hist(disease[feat], alpha=0.5, density=True, bins=n_bins, color="r")
        axes[i, j].set_title(f"{letter}. {feat}")
    for i in [0, 1, 2]:
        axes[i, 0].set_ylabel("Density")
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    output_path = os.path.join(OUT_DIR, "quantitatives_plot.png")
    plt.savefig(output_path)
    print(f"Quantitative feature distribution plots saved to {output_path}", f"", "", sep="\n")
    # plt.show()  # TODO show legend?


def cat_bar_compare(df, label):
    """Plot frequency distributions for categorical features"""
    no_disease = df[df[label] == 0]
    disease = df[df[label] == 1]
    fig, axes = plt.subplots(4, 2, figsize=(6, 9))
    width = 0.4
    for feat, i, j, letter in zip(CATEGORICAL, [0, 0, 1, 1, 2, 2, 3], [0, 1, 0, 1, 0, 1, 0], list("ABCDEFG")):
        # We assume the set of categorical values is the same for disease/no-disease. Correct for current dataset
        df1_val_counts = no_disease[feat].value_counts(normalize=True).sort_index()
        df2_val_counts = disease[feat].value_counts(normalize=True).sort_index()
        # print(f"{df1_val_counts}", f"{df2_val_counts}", "", sep="\n")
        assert set(df1_val_counts.index) == set(df2_val_counts.index), \
            f"Values found in disease/nondisease not equal. feature={feat}"
        x = np.arange(len(df1_val_counts.index))  # NB assuming the same as df2_val_counts.index
        axes[i, j].bar(x-width/2., df1_val_counts.values, width=width, alpha=1, color="b")
        axes[i, j].bar(x+width/2., df2_val_counts.values, width=width, alpha=1, color="r")
        axes[i, j].set_xticks(x)
        # NB assuming the same as df2_val_counts.index:
        axes[i, j].set_xticklabels(df1_val_counts.index.to_numpy().astype('int64'))
        axes[i, j].set_ylim([0, 0.9])
        axes[i, j].set_title(f"{letter}. {feat}")
    axes[3, 1].set_axis_off()
    for i in [0, 1, 2, 3]:
        axes[i, 0].set_ylabel("Relative frequency")
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    output_path = os.path.join(OUT_DIR, "categoricals_plot.png")
    plt.savefig(output_path)
    print(f"Categorical feature distribution plots saved to {output_path}", f"", "", sep="\n")
    # plt.show()


def mwu(df, label, features):
    """Mann-whitney U test"""
    no_disease = df[df[label] == 0]
    disease = df[df[label] == 1]
    results = pd.DataFrame(columns=['feature', 'test', 'statistic', 'p'])
    for feature in features:
        u, p = stats.mannwhitneyu(no_disease[feature], disease[feature], alternative='two-sided')
        results = results.append({'feature': feature, 'test': 'MW-U', 'statistic': u, 'p': p}, ignore_index=True)
    results = results.set_index("feature")
    return results


def chi2(df, label, features):
    """Chi square contingency test"""
    results = pd.DataFrame(columns=['feature', 'test', 'statistic', 'p'])
    for feat in features:
        # print(pd.crosstab(df[feat], df['disease'], margins=True))
        stat, p, dof, expected = stats.chi2_contingency(pd.crosstab(df[feat], df[label]))
        results = results.append({'feature': feat, 'test': 'chi2', 'statistic': stat, 'p': p}, ignore_index=True)
    results = results.set_index("feature")
    return results


def multiple_test_correction(test_results, method, p_value_label='p'):
    """Correct for multiple hypothesis testing using the chosen method"""
    alpha = 0.05  # NB this is only used in deciding which hypotheses are marked True/False in the reject vector
    reject, corrected_p, alphac_sidak, alphac_bonf = multipletests(test_results[p_value_label], method=method,
                                                                   alpha=alpha)
    with_corrected = test_results.copy()
    with_corrected[f"p_{method}"] = corrected_p
    return with_corrected


def main():
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("data", help="Heart disease data file, comma separated values")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Error: cannot find data file {args.data}")
        exit()

    df = preproc.get_data(args.data)
    ground_truth_label = 'disease'

    print(pd.Series(FULL_NAMES).to_frame())
    print()

    # Comparing quantitative fields between disease and no disease
    mwu_results = mwu(df, ground_truth_label, QUANTITATIVE)
    print(f"Mann Whiteny U results, quantitative fields:", mwu_results, "", sep="\n")

    # Analysing categorical variables with chi2
    chi2_results = chi2(df, ground_truth_label, CATEGORICAL)
    print(f"chi2 results, categorical fields:", chi2_results, "", sep="\n")

    # Correct for multiple hypothesis testing
    combined = pd.concat([mwu_results, chi2_results], axis=0)
    combined_corrected = multiple_test_correction(combined, 'bonferroni')
    print(f"Combined corrected test results:", combined_corrected, "", sep="\n")

    # df_dummies = assign_dummies(df, CATEGORICALS)
    # plot_corrmat(df_dummies.drop('disease', axis=1), 'kendall')

    quant_hist_compare(df, ground_truth_label)
    cat_bar_compare(df, ground_truth_label)


if __name__ == "__main__":
    main()
