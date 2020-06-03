# Heart Disease Detection

## Summary

significance of heart disease? could reference elsewhere
Could move boring methods to the end

## Introduction
background
aims

## Results
### Investigating features
Plots of distibutions and statistical tests. conclude there is discriminatory signal, but will keep all features in case there are higher order relationships that may be valuable to more complex models.
correlation matrix?

The data comprise six quantitative (or ordinal) and seven categorical features. We apply the two-sided Mann-Whitney U test for quantitative features, as this is a nonparametric test that requires no assumption about the underlying distributions. For categorical variables we apply Pearson's chi squared test. We reject the null hypothesis of identical distributions if p < 0.05. With 13 features to investigate, performing multiple hypothesis tests increases the danger of type 1 error. We therefore adjust the resulting p values using the Bonferroni correction (arguably the most conservative of correction procedures) and retain the 0.05 threshold.

<p align="center">
<img src="../docs/cleveland/categoricals_plot.png" alt="Continuous variable distributions" width="300"/>
</p>

<p align="center">
<img src="../docs/cleveland/quantitatives_plot.png" alt="Continuous variable distributions" width="300"/>
</p>

### Classification models

## Conclusions

## Methods

### Models


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



