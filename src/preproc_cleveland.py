import pandas as pd
import numpy as np

FILE = "processed.cleveland.data.csv"
NAMES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca",
         "thal", "num"]
CATEGORICAL = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
QUANTITATIVE = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
MISSING_MARKER = "?"  # used in input data to denote missing data
DUMMY_SEPARATOR = "-"

# FULL_NAMES = {
#     'age': "Age in years",
#     'sex': "Sex",
#     'cp': "Chest pain type",
#     'trestbps': "Resting blood pressure",
#     'chol': "Serum cholestoral",
#     'fbs': "Fasting blood sugar ",
#     'restecg': "Resting electrocardiography",
#     'thalach': "Max. heart rate ",
#     'exang': "Exercise induced angina",
#     'oldpeak': "Exercise-induced ST depression",
#     'slope': "Slope of the peak exercise ST segment",
#     'ca': "Number of major vessels",
#     'thal': "Thallium scan result",
# }
GROUND_TRUTH_LABEL = 'disease'
DESCRIPTION = "Identifying features with potential predictive value for heart disease"


def from_file(path):
    df = get_data(path)
    df = assign_dummies(df, CATEGORICAL)
    return df


def get_data(file_path):
    """Data pre-processing"""
    assert set(CATEGORICAL + QUANTITATIVE) == set(NAMES) - set(["num"]), "Inconsistent categorical/quantiative names"
    df = pd.read_csv(file_path, names=NAMES).drop_duplicates()
    df = df.replace(to_replace={MISSING_MARKER: np.nan})  # Deal with missing data
    df = df.astype('float64')

    # Impute missing values with model value for each feature
    df = df.fillna(df.mode().iloc[0])

    # Group num values > 0 as disease or no disease
    # print(f"num value counts", df['num'].value_counts().to_frame().to_csv(sep="\t"), "", sep="\n")
    df[GROUND_TRUTH_LABEL] = df['num'].apply(lambda x: 0 if x == 0 else 1)
    df = df.drop("num", axis=1)
    df = df[[GROUND_TRUTH_LABEL] + [col for col in df.columns if col != GROUND_TRUTH_LABEL]]  # move to first column
    return df


def assign_dummies(df, names):
    for name in names:
        df = pd.concat([df.drop(name, axis=1),  # .astype('int64'),
                        pd.get_dummies(df[name], prefix=name, prefix_sep=DUMMY_SEPARATOR, drop_first=True)], axis=1)
    return df


def main():
    pass


if __name__ == '__main__':
    main()