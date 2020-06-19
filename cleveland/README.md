# Heart Disease Detection

Christopher Monit

Summer 2020

## Introduction
Coronary heart disease (CHD) affects 2.3 million people and is responsible for 64,000 deaths annually in the UK (<a id="bhf" href="https://www.bhf.org.uk/what-we-do/our-research/heart-statistics">British Heart Foundation</a>). It is caused by aggregation of fatty deposits in the coronary arteries and normally diagnosed by electrocardiogram or chest x-ray, but using machine learning it may be possible to accurately identify CHD from indirect measurements such as blood tests and patient reported symptoms, saving time and costs.

Here we investigate the predictive potential of shallow neural networks in identifying CHD from the commonly studied <a id="bhf" href="https://archive.ics.uci.edu/ml/datasets/Heart+Disease">Cleveland dataset</a>, comprising 303 records of 13 measurements together with a ground truth labels indicating 'CHD' or 'no CHD'.

## Results
### Investigating features
[Comment somewhere that the classes are approximately balanced, could show the table I had in earlier report]

Firstly, we investigate whether there is discriminatory signal in these variables. 


Visualising the features by presence/absence of CHD reveals clearly distinct distributions for some but not all measurements:
<p align="center">
<img src="../docs/cleveland/categoricals_plot.png" alt="Continuous variable distributions" width="600"/>
</p>
<p align="center">
<img src="../docs/cleveland/quantitatives_plot.png" alt="Continuous variable distributions" width="600"/>
</p>

But are these differences statistically significant? We applied the two-sided Mann-Whitney U test for the six quantitative features, as this is a nonparametric test that requires no assumption about the underlying distributions, while for the seven categorical variables we apply Pearson's chi squared test. We reject the null hypothesis of identical distributions if p < 0.05. With 13 features to investigate, performing multiple hypothesis tests increases the danger of type 1 error and therefore adjust the resulting *p* values using the Bonferroni correction (arguably the most conservative of correction procedures) and retain the 0.05 threshold (values to 5 decimal places):

| feature   | description                           | test   |   statistic |       p |   p_bon | p_bon < 0.05   |
|:----------|:--------------------------------------|:-------|------------:|--------:|--------:|:---------------|
| age       | Age in years                          | MW-U   |   8274.5    | 4e-05   | 0.00051 | Yes            |
| trestbps  | Resting blood pressure                | MW-U   |   9710      | 0.02597 | 0.33764 | No             |
| chol      | Serum cholestoral                     | MW-U   |   9798.5    | 0.03536 | 0.45967 | No             |
| thalach   | Max. heart rate                       | MW-U   |  16989.5    | 0       | 0       | Yes            |
| oldpeak   | Exercise-induced ST depression        | MW-U   |   6037      | 0       | 0       | Yes            |
| ca        | Number of major vessels               | MW-U   |   5711.5    | 0       | 0       | Yes            |
| sex       | Sex                                   | chi2   |     22.0426 | 0       | 3e-05   | Yes            |
| cp        | Chest pain type                       | chi2   |     81.8158 | 0       | 0       | Yes            |
| fbs       | Fasting blood sugar                   | chi2   |      0.0771 | 0.78127 | 1       | No             |
| restecg   | Resting electrocardiography           | chi2   |     10.0515 | 0.00657 | 0.08536 | No             |
| exang     | Exercise induced angina               | chi2   |     54.6864 | 0       | 0       | Yes            |
| slope     | Slope of the peak exercise ST segment | chi2   |     45.7846 | 0       | 0       | Yes            |
| thal      | Thallium scan result                  | chi2   |     82.6846 | 0       | 0       | Yes            |

We concluded there is significant discriminatory signal among some features which may be useful for CHD classification.

Computing Kendall's rank correlation coefficient between features (with categorical features transformed to quantative using one hot encoding), showed there is not an overwhelmingly strong correlation between any of the features:

<p align="center">
<img src="../docs/cleveland/corr_mat_kendall.png" alt="Correlation matrix" width="600"/>
</p>

### Classification models

aimed to make a classifier, binary classification task
NB using all features
Define notation for architectures, make clear tanh used
#### Models
Simple, fully connected architectures with units in hidden layers using hyperbolic tangent activation functions and the final output layer comprising a single sigmoid function unit. The complexity of the networks ranged from no hidden units (i.e. logistic regression) to [N] hidden layers, comprising [M] units. Weight and bias parameters were determined by [TBC] iterations of standard, batch gradient descent using the cross entropy cost function, with a range of learning rate (alpha) values. Frobenius norm (matrix L2 norm) regularisation for weight parameters was used to limit overfitting to training data, with regularisation parameter (lambda) values ranging from [TBC]. 

#### Model training and validation

20% of the total dataset was uniformly randomly subsampled, preserving the proportions of CHD and non-CHD cases, and reserved as a test set. The remaining 80% was used for hyperparameter tuning using 4-fold cross validation, by uniformly randomly subsampling 4 non-overlapping validation sets and for each of these using the remaining 3 folds for training each model, before measuring the models' performances on the given validation set [mention whether stratified or not]. Area under the ROC curve (AUC) was chosen as a performance metric to compare models, as this summarises the compromise between a model's sensitivity and specificity. For each experimental condition the mean AUC was calculated across the 4 validation folds.

The figures below summarise the influence of hyperparameters on mean ROC AUC from 4-fold cross validation, on both the training and validation sets, with an arbitrarily selected grid search over hyperparameter values:

<p align="center">
<img src="../docs/cleveland/coarse_train_mean_auc.png" alt="Correlation matrix" width="600"/>
</p>
Hyperparameter grid search, mean AUC on cross validation training datasets. 

<p align="center">
<img src="../docs/cleveland/coarse_val_mean_auc.png" alt="Correlation matrix" width="600"/>
</p>
Hyperparameter grid search, mean AUC on cross validation validation datasets. 

While there is considerable overfitting to the training sets, nonetheless there is respectable performance on the validation sets for most models. Performance metrics of the five models with highest ROC AUC are shown below:  [comment some models have failed to train]

|   arch. |   alpha |   reg |   roc_auc |   sens. |   spec. |   acc. |
|--------:|--------:|------:|----------:|--------:|--------:|-------:|
|     2_1 |     1   |   0   |    0.869  |  0.8806 |  0.8574 | 0.8677 |
|   2_2_1 |     0.5 |   0   |    0.8613 |  0.8806 |  0.842  | 0.8593 |
|     2_1 |     0.5 |   0   |    0.8596 |  0.8621 |  0.8572 | 0.8593 |
|   2_2_1 |     0.1 |   0   |    0.8513 |  0.8528 |  0.8498 | 0.851  |
|   2_4_1 |     0.1 |   0.5 |    0.8486 |  0.8251 |  0.8721 | 0.851  |

We then pursued a finer grid search over a narrower range of values in this high-performing region of the hyperparameter space, and measured model performance as before:

<p align="center">
<img src="../docs/cleveland/fine_val_mean_auc.png" alt="Correlation matrix" width="600"/>
</p>

The five highest performing models were as follows: 

|   arch. |   alpha |   reg |   roc_auc |   sens. |   spec. |   acc. |
|--------:|--------:|------:|----------:|--------:|--------:|-------:|
|     2_1 |     1.1 |     0 |    0.8728 |  0.8806 |  0.865  | 0.8719 |
|   2_2_1 |     0.8 |     0 |    0.8697 |  0.8899 |  0.8496 | 0.8676 |
|     2_1 |     1   |     0 |    0.869  |  0.8806 |  0.8574 | 0.8677 |
|     2_1 |     1.2 |     0 |    0.869  |  0.8806 |  0.8574 | 0.8677 |
|     2_1 |     0.9 |     0 |    0.8644 |  0.8714 |  0.8574 | 0.8635 |

This suggested the optimum model had architecture '2_1', learning rate 1.1 and without any regularisation term'.

#### Test set performance

We then trained model [X] on the whole training/validation set (i.e. pooling all 4 CV folds).



[show performance of best model on test set, single row table of performance stats]

[ROC curve for the best model, I guess on the test set]


## Discussion
limitations? I haven't worked out how much performance is beneficial, so not clear if this is good enough.
could have done more feature engineering? would help to give an example of what to do though
these are impressive, better than kaggle competition winners?

Several implementation strategies could have improved training and allowed more efficient model development. Here we have applied a standard, batch gradient descent algorithm to learn weight and bias parameters but there may have been benefit from applying stochastic (mini-batch) gradient descent, where updates to parameters are determined using random smaller subsets of the data at a time. Additional augmented search algorithms such as Adam (adaptive momentum estimation), which help maintain efficient training despite the stochastic exposure to training examples, may have had further benefit. 

learning rate decay
activation functions - testing this now anyway
perturbing values if not changing much
would be better to have more flexible definition of optimisation end time

if results say reg didn't work well, could say we should have tried dropout or something

## Methods

### Implementation and model experiments
All source code can be found in this repository. Models and optimisation algorithms have been implemented from scratch in pure Python, using standard numerical libraries, namely numpy, pandas and sci-kit learn.

Initial weight parameters were sampled from a standard normal distribution [define distribution], ranging from 10^-3 to 1.1; bias parameters were all initialised as 0.

### Data Preprocessing

#### Missing data
Features 'ca' and 'thal' (blood vessels and thallium scan) have four and two missing entries respectively, of 303 records in total. Since 'thal' is categorical and ‘ca’ is ordinal, we impute missing values using the modal value for each feature.

#### Ground truth labels
The dataset's documentation is somewhat ambiguous regarding the ground-truth labelling with/without heart disease. It states that value 0 represents < 50% vessel diameter narrowing, while value 1 represents > 50% diameter narrowing; however, additional values are found in this column with the following frequencies:

| Value     | 0   | 1  | 2  | 3  | 4  |
|-----------|-----|----|----|----|----|
| Frequency | 164 | 55 | 36 | 35 | 13 |

In the absence of expert opinion, we have assumed categories 2-4 also represent disease states, as others have previously (e.g. https://gallery.azure.ai/Experiment/Heart-Disease-Prediction-5), yielding 164 ‘no disease’ cases and 139 ‘disease’ cases. While there is a relatively small imbalance in the samples for each class, this inequality should not affect the accuracies of these tests since the absolute number for each class is large.

#### Categorical variables
Seven of the 13 variables are categorical or binary, and therefore not directly suitable for training by neural networks which expect input variables to be continuous. Therefore these seven features were transformed as dummy variables (so called 'one hot encoding') whereby a categorical feature comprising _N_ unique states is replaced with _N_ dummy features, each populated with 1 or 0 depending whether the state is present or absent.

#### Standardising values
Models reliant on large numbers of linear operations, including neural networks, are vulnerable to overly conditioning predictions based on the magnitude or range of some features; but this can be avoided by transforming input features and enforcing a consistent distribution. Following one hot encoding of categorical variables, we scaled training cases such that each feature had mean equal to 0 and variance equal to 1. Each fold's validation set and the test set were transformed by the same procedure as corresponding training data. 


Could have a note about code tests used. This could be a report about implementation as much as results

## References
<a id="1">[1]</a> 


