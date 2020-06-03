# Heart Disease Detection

Christopher Monit

Summer 2020

## Introduction
Coronary heart disease (CHD) affects 2.3 million people and is responsible for 64,000 deaths annually in the UK (<a id="bhf" href="https://www.bhf.org.uk/what-we-do/our-research/heart-statistics">British Heart Foundation</a>). It is caused by aggregation of fatty deposits in the coronary arteries and normally diagnosed by electrocardiogram or chest x-ray, but using machine learning it may be possible to accurately identify CHD from indirect measurements such as blood tests and patient reported symptoms, saving time and costs.

Here we investigate the predictive potential of shallow neural networks in identifying CHD from the commonly studied <a id="bhf" href="https://archive.ics.uci.edu/ml/datasets/Heart+Disease">Cleveland dataset</a>, comprising 303 records of 13 measurements together with a ground truth labels indicating 'CHD' or 'no CHD'.

## Results
### Investigating features
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
Simple architectures with units in hidden layers using hyperbolic tangent activation functions and the final output layer comprising a single sigmoid function unit. The complexity of the networks ranged from no hidden units (i.e. logistic regression) to two hidden layers, each comprising two units. Weight and bias parameters were determined by 10^4 iterations of standard gradient descent using the cross entropy cost function, with a range of learning rate (alpha) values. Frobenius norm (matrix L2 norm) regularisation for weight parameters was used to limit overfitting to the training set, with regularisation parameter (lambda) values ranging from 0 to 10. The data were split 3:1 into training and test sets, by randomly sampled and preserving the proportions of CHD and non-CHD cases. Candidate models were assessed by the ROC area under the curve (AUC) on the test set.

Summarise the distribution of performances? Violin plot or something?

The following table summarises the performance of the top AUC model:

| layers | weight scaling | alpha | iterations   | lambda | AUC    | sens. | spec. | acc.  | TN | FP | FN | TP |
|--------|----------------|-------|--------------|--------|--------|-------|-------|-------|----|----|----|----|
| 2_2_1  | 1.100          | 0.2   | 10000        | 2      | 0.908  | 0.921 | 0.895 | 0.908 | 34 | 4  | 3  | 35 |

ROC 

## Conclusions
limitations? I haven't worked out how much performance is beneficial, so not clear if this is good enough.
could note that it would be preferable to have a separate purely test set
could have done more feature engineering?
these are impressive, better than kaggle competition winners?

## Methods

### Implementation and model experiments
All source code can be found in this repository. Models and optimisation algorithms have been implemented from scratch in pure Python, using standard numerical libraries (e.g. numpy).

Initial weight parameters were sampled from a standard normal distribution and then multiplied by one of several scalars, ranging from 10^-3 to 1.1; bias parameters were all initialised as 0.

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
One hot encoding

#### Standardising values

## References
<a id="1">[1]</a> 


