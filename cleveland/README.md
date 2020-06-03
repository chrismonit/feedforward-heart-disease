# Heart Disease Detection

Christopher Monit

Summer 2020

## Introduction
Coronary heart disease (CHD) affects 2.3 million people and is responsible for 64,000 deaths annually in the UK (<a id="bhf" href="https://www.bhf.org.uk/what-we-do/our-research/heart-statistics">British Heart Foundation</a>). It is caused by aggregation of fatty deposits in the coronary arteries and normally diagnosed by electrocardiogram or chest x-ray, but using machine learning it may be possible to accurately identify CHD from indirect measurements such as blood tests and patient reported symptoms, saving time and costs.

Here we investigate the predictive potential of shallow neural networks in identifying CHD from the commonly studied <a id="bhf" href="https://archive.ics.uci.edu/ml/datasets/Heart+Disease">Cleveland dataset</a>, comprising 303 records of 13 measurements together with a ground truth labels indicating 'CHD' or 'no CHD'.

## Results
### Investigating features
Firstly, we investigate whether there is discriminatory signal in these variables. The following table summarises the  six features quantitative/ordinal and seven categorical features and their definitions:

| Feature   | Description                           |
|:----------|:--------------------------------------|
| age       | Age in years                          |
| sex       | Sex                                   |
| cp        | Chest pain type                       |
| trestbps  | Resting blood pressure                |
| chol      | Serum cholestoral                     |
| fbs       | Fasting blood sugar                   |
| restecg   | Resting electrocardiography           |
| thalach   | Max. heart rate                       |
| exang     | Exercise induced angina               |
| oldpeak   | Exercise-induced ST depression        |
| slope     | Slope of the peak exercise ST segment |
| ca        | Number of major vessels               |
| thal      | Thallium scan result                  |

Visualising the features by presence/absence of CHD reveals clearly distinct distributions for some but not all measurements:
<p align="center">
<img src="../docs/cleveland/categoricals_plot.png" alt="Continuous variable distributions" width="300"/>
</p>
<p align="center">
<img src="../docs/cleveland/quantitatives_plot.png" alt="Continuous variable distributions" width="300"/>
</p>

But are these differences statistically significant? We applied the two-sided Mann-Whitney U test for quantitative features, as this is a nonparametric test that requires no assumption about the underlying distributions, while for categorical variables we apply Pearson's chi squared test. We reject the null hypothesis of identical distributions if p < 0.05. With 13 features to investigate, performing multiple hypothesis tests increases the danger of type 1 error and therefore adjust the resulting *p* values using the Bonferroni correction (arguably the most conservative of correction procedures) and retain the 0.05 threshold:

|     Feature     |     Test    |     Statistic    |     p            |     p_Bon        |     p_Bon < 0.05   |
|-----------------|-------------|------------------|------------------|------------------|--------------------|
|     age         |     MW-U    |     8274.50      |     4 x 10-5     |     5 x 10-4     |     Yes            |
|     trestbps    |     MW-U    |     9710.00      |     0.03         |     0.34         |     No             |
|     chol        |     MW-U    |     9798.50      |     0.04         |     0.46         |     No             |
|     thalach     |     MW-U    |     16989.50     |     2 x 10-13    |     2 x 10-12    |     Yes            |
|     oldpeak     |     MW-U    |     6037.00      |     7 x 10-13    |     9 x 10-12    |     Yes            |
|     ca          |     MW-U    |     5711.50      |     2 x 10-17    |     3 x 10-16    |     Yes            |
|     sex         |     chi2    |     22.04        |     3 x 10-6     |     3 x 10-5     |     Yes            |
|     cp          |     chi2    |     81.81        |     10-17        |     2 x 10-16    |     Yes            |
|     fbs         |     chi2    |     0.08         |     0.78         |     1            |     No             |
|     restecg     |     chi2    |     10.05        |     0.01         |     0.09         |     No             |
|     exang       |     chi2    |     54.69        |     10-13        |     2 x 10-12    |     Yes            |
|     slope       |     chi2    |     45.78        |     10-10        |     10-9         |     Yes            |
|     thal        |     chi2    |     82.68        |     10-18        |     10-17        |     Yes            |

We concluded there is significant discriminatory signal among some features. 

Correlation matrix?

### Classification models
NB using all features

## Conclusions

## Methods

### Models
Details of implementation, optimisation, the hypter parameters that were tweaked etc

### Preprocessing

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


