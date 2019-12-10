---
layout: post
author: "Unarine Tshiwawa"
---
### 1. Identifying the problem:

In this task, I seek to demonstrate how unsupervised learning machine leaning can be used in an unlabelled dataset.  Just like in classification, each instance gets assigned to a group. However, this is an unsupervised task.  

`In this work, the goal is to group similar instances together into clusters.`



```python
import pandas as pd
from IPython.display import display
import seaborn as sns
sns.set('notebook')
import matplotlib.pylab as plt
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
```

# Dataset

The dataset contained different clusters of stars from some field.  I will use clustering algorithm to attempts to find distinct groups of data without reference to any labels.


```python
df = pd.read_csv('data.csv')

df1 = df.copy()
```

Let's check the shape of the data


```python
df1.shape
```




    (18061, 160)



In this case, we see that we have 18061 samples and each sample in the dataset has 29 features.

Let's impute missing values:


```python
print('\n---------------------------Impute missing values----------------------')
display(df1.isnull().sum())
```


    ---------------------------Impute missing values----------------------



    semester                                    0
    FieldHalf                                   0
    FIELD_1                                     0
    Half                                        0
    CHIP                                        0
    RA_1                                      546
    DEC_1                                     546
    X_1                                       546
    Y_1                                       546
    MAG                                       546
    MAGe                                      546
    Name_1                                      0
    Mean_Mag_1                                  0
    RMS_1                                       0
    Expected_RMS_1                              0
    Alarm_2                                     0
    Chi2_3                                      0
    Jstet_4                                     0
    Kurtosis_4                                  0
    Lstet_4                                     0
    LS_Period_1_5                               0
    Log10_LS_Prob_1_5                           0
    LS_SNR_1_5                                  0
    Killharm_Mean_Mag_6                         0
    Killharm_Per1_Fundamental_Sincoeff_6        0
    Killharm_Per1_Fundamental_Coscoeff_6        0
    Killharm_Per1_Amplitude_6                   0
    Period_1_7                                  0
    AOV_1_7                                     0
    AOV_SNR_1_7                                 0
                                            ...  
    rmagAB                                  12476
    e_rmag                                  12476
    chir                                    12207
    warningr                                12021
    rmagap                                  12242
    rmagapAB                                12242
    e_rmagap                                12242
    snrr                                    12220
    rmaglim                                 12219
    PSFFWHMr                                12021
    MJDr                                    12021
    detIDr                                  12021
    cleani                                  12021
    imag                                    12707
    imagAB                                  12707
    e_imag                                  12707
    chii                                    12222
    warningi                                12021
    imagap                                  12252
    imagapAB                                12252
    e_imagap                                12252
    snri                                    12222
    imaglim                                 12220
    PSFFWHMi                                12021
    MJDi                                    12021
    detIDi                                  12021
    Field_2                                 12021
    Ext                                     12021
    nbDist                                  12021
    Separation                              12021
    Length: 160, dtype: int64


Here we will remove redundant columns from the dataset.


```python
df1 = df1.iloc[:,5:34]


display(df1.head(8))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RA_1</th>
      <th>DEC_1</th>
      <th>X_1</th>
      <th>Y_1</th>
      <th>MAG</th>
      <th>MAGe</th>
      <th>Name_1</th>
      <th>Mean_Mag_1</th>
      <th>RMS_1</th>
      <th>Expected_RMS_1</th>
      <th>...</th>
      <th>Killharm_Per1_Fundamental_Sincoeff_6</th>
      <th>Killharm_Per1_Fundamental_Coscoeff_6</th>
      <th>Killharm_Per1_Amplitude_6</th>
      <th>Period_1_7</th>
      <th>AOV_1_7</th>
      <th>AOV_SNR_1_7</th>
      <th>AOV_NEG_LN_FAP_1_7</th>
      <th>RA2000</th>
      <th>DEC2000</th>
      <th>lspermin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>267.637420</td>
      <td>-32.505489</td>
      <td>1309.394</td>
      <td>117.711</td>
      <td>11.6591</td>
      <td>0.0026</td>
      <td>100009</td>
      <td>15.01506</td>
      <td>0.01028</td>
      <td>0.00179</td>
      <td>...</td>
      <td>0.01155</td>
      <td>0.00484</td>
      <td>0.02505</td>
      <td>0.015565</td>
      <td>3.41553</td>
      <td>4.67124</td>
      <td>2.28794</td>
      <td>17:50:33.0</td>
      <td>-32:30:19.8</td>
      <td>113.177390</td>
    </tr>
    <tr>
      <th>1</th>
      <td>267.568298</td>
      <td>-32.472861</td>
      <td>324.079</td>
      <td>665.137</td>
      <td>10.8853</td>
      <td>0.0018</td>
      <td>100218</td>
      <td>14.18320</td>
      <td>0.00339</td>
      <td>0.00118</td>
      <td>...</td>
      <td>0.00211</td>
      <td>0.00338</td>
      <td>0.00797</td>
      <td>0.014812</td>
      <td>5.02959</td>
      <td>6.17483</td>
      <td>4.81995</td>
      <td>17:50:16.4</td>
      <td>-32:28:22.3</td>
      <td>92.062210</td>
    </tr>
    <tr>
      <th>2</th>
      <td>267.555268</td>
      <td>-32.285918</td>
      <td>126.340</td>
      <td>3818.489</td>
      <td>16.5748</td>
      <td>0.0332</td>
      <td>200510</td>
      <td>20.58942</td>
      <td>0.27845</td>
      <td>0.05443</td>
      <td>...</td>
      <td>-0.26668</td>
      <td>-0.23009</td>
      <td>0.70445</td>
      <td>0.017541</td>
      <td>5.07752</td>
      <td>8.06232</td>
      <td>4.88935</td>
      <td>17:50:13.3</td>
      <td>-32:17:09.3</td>
      <td>104.545224</td>
    </tr>
    <tr>
      <th>3</th>
      <td>267.758599</td>
      <td>-32.485136</td>
      <td>886.993</td>
      <td>462.508</td>
      <td>17.1670</td>
      <td>0.0503</td>
      <td>100445</td>
      <td>21.99514</td>
      <td>0.18661</td>
      <td>0.15743</td>
      <td>...</td>
      <td>-0.05318</td>
      <td>0.15253</td>
      <td>0.32307</td>
      <td>0.007814</td>
      <td>3.24337</td>
      <td>3.98715</td>
      <td>1.99459</td>
      <td>17:51:02.1</td>
      <td>-32:29:06.5</td>
      <td>18.891792</td>
    </tr>
    <tr>
      <th>4</th>
      <td>267.797234</td>
      <td>-32.472325</td>
      <td>1436.175</td>
      <td>679.620</td>
      <td>12.2333</td>
      <td>0.0034</td>
      <td>100546</td>
      <td>15.69679</td>
      <td>0.00462</td>
      <td>0.00248</td>
      <td>...</td>
      <td>-0.00545</td>
      <td>-0.00253</td>
      <td>0.01201</td>
      <td>0.006415</td>
      <td>4.39385</td>
      <td>4.75353</td>
      <td>3.86889</td>
      <td>17:51:11.3</td>
      <td>-32:28:20.4</td>
      <td>71.308296</td>
    </tr>
    <tr>
      <th>5</th>
      <td>267.767202</td>
      <td>-32.368453</td>
      <td>1003.500</td>
      <td>2430.221</td>
      <td>15.2115</td>
      <td>0.0146</td>
      <td>200327</td>
      <td>18.97295</td>
      <td>0.02625</td>
      <td>0.01531</td>
      <td>...</td>
      <td>-0.00799</td>
      <td>-0.02875</td>
      <td>0.05967</td>
      <td>0.038873</td>
      <td>4.48847</td>
      <td>4.89371</td>
      <td>3.91457</td>
      <td>17:51:04.1</td>
      <td>-32:22:06.4</td>
      <td>97.136510</td>
    </tr>
    <tr>
      <th>6</th>
      <td>267.768327</td>
      <td>-32.367157</td>
      <td>1019.457</td>
      <td>2452.106</td>
      <td>16.0023</td>
      <td>0.0231</td>
      <td>200346</td>
      <td>20.05330</td>
      <td>0.05390</td>
      <td>0.03288</td>
      <td>...</td>
      <td>-0.01641</td>
      <td>-0.05696</td>
      <td>0.11855</td>
      <td>0.010505</td>
      <td>5.27810</td>
      <td>6.39846</td>
      <td>5.05030</td>
      <td>17:51:04.4</td>
      <td>-32:22:01.8</td>
      <td>92.754403</td>
    </tr>
    <tr>
      <th>7</th>
      <td>267.720869</td>
      <td>-32.337827</td>
      <td>341.550</td>
      <td>2945.095</td>
      <td>16.4765</td>
      <td>0.0312</td>
      <td>200687</td>
      <td>20.54297</td>
      <td>0.05399</td>
      <td>0.04898</td>
      <td>...</td>
      <td>0.04113</td>
      <td>-0.03545</td>
      <td>0.10859</td>
      <td>0.017723</td>
      <td>3.73648</td>
      <td>4.71103</td>
      <td>2.74842</td>
      <td>17:50:53.0</td>
      <td>-32:20:16.2</td>
      <td>31.794682</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 29 columns</p>
</div>


Let's now check the shape of the data.


```python
df1.shape
```




    (18061, 29)



#### Removing all rows containing nans:


```python
display(df1.isnull().sum())
```


    RA_1                                    546
    DEC_1                                   546
    X_1                                     546
    Y_1                                     546
    MAG                                     546
    MAGe                                    546
    Name_1                                    0
    Mean_Mag_1                                0
    RMS_1                                     0
    Expected_RMS_1                            0
    Alarm_2                                   0
    Chi2_3                                    0
    Jstet_4                                   0
    Kurtosis_4                                0
    Lstet_4                                   0
    LS_Period_1_5                             0
    Log10_LS_Prob_1_5                         0
    LS_SNR_1_5                                0
    Killharm_Mean_Mag_6                       0
    Killharm_Per1_Fundamental_Sincoeff_6      0
    Killharm_Per1_Fundamental_Coscoeff_6      0
    Killharm_Per1_Amplitude_6                 0
    Period_1_7                                0
    AOV_1_7                                   0
    AOV_SNR_1_7                               0
    AOV_NEG_LN_FAP_1_7                        0
    RA2000                                  546
    DEC2000                                 546
    lspermin                                  0
    dtype: int64


Let's drop all rows/samples containing nans:


```python
df1 = df1.dropna()

df1.shape
```




    (17515, 29)




```python
display(df1.head(10))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RA_1</th>
      <th>DEC_1</th>
      <th>X_1</th>
      <th>Y_1</th>
      <th>MAG</th>
      <th>MAGe</th>
      <th>Name_1</th>
      <th>Mean_Mag_1</th>
      <th>RMS_1</th>
      <th>Expected_RMS_1</th>
      <th>...</th>
      <th>Killharm_Per1_Fundamental_Sincoeff_6</th>
      <th>Killharm_Per1_Fundamental_Coscoeff_6</th>
      <th>Killharm_Per1_Amplitude_6</th>
      <th>Period_1_7</th>
      <th>AOV_1_7</th>
      <th>AOV_SNR_1_7</th>
      <th>AOV_NEG_LN_FAP_1_7</th>
      <th>RA2000</th>
      <th>DEC2000</th>
      <th>lspermin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>267.637420</td>
      <td>-32.505489</td>
      <td>1309.394</td>
      <td>117.711</td>
      <td>11.6591</td>
      <td>0.0026</td>
      <td>100009</td>
      <td>15.01506</td>
      <td>0.01028</td>
      <td>0.00179</td>
      <td>...</td>
      <td>0.01155</td>
      <td>0.00484</td>
      <td>0.02505</td>
      <td>0.015565</td>
      <td>3.41553</td>
      <td>4.67124</td>
      <td>2.28794</td>
      <td>17:50:33.0</td>
      <td>-32:30:19.8</td>
      <td>113.177390</td>
    </tr>
    <tr>
      <th>1</th>
      <td>267.568298</td>
      <td>-32.472861</td>
      <td>324.079</td>
      <td>665.137</td>
      <td>10.8853</td>
      <td>0.0018</td>
      <td>100218</td>
      <td>14.18320</td>
      <td>0.00339</td>
      <td>0.00118</td>
      <td>...</td>
      <td>0.00211</td>
      <td>0.00338</td>
      <td>0.00797</td>
      <td>0.014812</td>
      <td>5.02959</td>
      <td>6.17483</td>
      <td>4.81995</td>
      <td>17:50:16.4</td>
      <td>-32:28:22.3</td>
      <td>92.062210</td>
    </tr>
    <tr>
      <th>2</th>
      <td>267.555268</td>
      <td>-32.285918</td>
      <td>126.340</td>
      <td>3818.489</td>
      <td>16.5748</td>
      <td>0.0332</td>
      <td>200510</td>
      <td>20.58942</td>
      <td>0.27845</td>
      <td>0.05443</td>
      <td>...</td>
      <td>-0.26668</td>
      <td>-0.23009</td>
      <td>0.70445</td>
      <td>0.017541</td>
      <td>5.07752</td>
      <td>8.06232</td>
      <td>4.88935</td>
      <td>17:50:13.3</td>
      <td>-32:17:09.3</td>
      <td>104.545224</td>
    </tr>
    <tr>
      <th>3</th>
      <td>267.758599</td>
      <td>-32.485136</td>
      <td>886.993</td>
      <td>462.508</td>
      <td>17.1670</td>
      <td>0.0503</td>
      <td>100445</td>
      <td>21.99514</td>
      <td>0.18661</td>
      <td>0.15743</td>
      <td>...</td>
      <td>-0.05318</td>
      <td>0.15253</td>
      <td>0.32307</td>
      <td>0.007814</td>
      <td>3.24337</td>
      <td>3.98715</td>
      <td>1.99459</td>
      <td>17:51:02.1</td>
      <td>-32:29:06.5</td>
      <td>18.891792</td>
    </tr>
    <tr>
      <th>4</th>
      <td>267.797234</td>
      <td>-32.472325</td>
      <td>1436.175</td>
      <td>679.620</td>
      <td>12.2333</td>
      <td>0.0034</td>
      <td>100546</td>
      <td>15.69679</td>
      <td>0.00462</td>
      <td>0.00248</td>
      <td>...</td>
      <td>-0.00545</td>
      <td>-0.00253</td>
      <td>0.01201</td>
      <td>0.006415</td>
      <td>4.39385</td>
      <td>4.75353</td>
      <td>3.86889</td>
      <td>17:51:11.3</td>
      <td>-32:28:20.4</td>
      <td>71.308296</td>
    </tr>
    <tr>
      <th>5</th>
      <td>267.767202</td>
      <td>-32.368453</td>
      <td>1003.500</td>
      <td>2430.221</td>
      <td>15.2115</td>
      <td>0.0146</td>
      <td>200327</td>
      <td>18.97295</td>
      <td>0.02625</td>
      <td>0.01531</td>
      <td>...</td>
      <td>-0.00799</td>
      <td>-0.02875</td>
      <td>0.05967</td>
      <td>0.038873</td>
      <td>4.48847</td>
      <td>4.89371</td>
      <td>3.91457</td>
      <td>17:51:04.1</td>
      <td>-32:22:06.4</td>
      <td>97.136510</td>
    </tr>
    <tr>
      <th>6</th>
      <td>267.768327</td>
      <td>-32.367157</td>
      <td>1019.457</td>
      <td>2452.106</td>
      <td>16.0023</td>
      <td>0.0231</td>
      <td>200346</td>
      <td>20.05330</td>
      <td>0.05390</td>
      <td>0.03288</td>
      <td>...</td>
      <td>-0.01641</td>
      <td>-0.05696</td>
      <td>0.11855</td>
      <td>0.010505</td>
      <td>5.27810</td>
      <td>6.39846</td>
      <td>5.05030</td>
      <td>17:51:04.4</td>
      <td>-32:22:01.8</td>
      <td>92.754403</td>
    </tr>
    <tr>
      <th>7</th>
      <td>267.720869</td>
      <td>-32.337827</td>
      <td>341.550</td>
      <td>2945.095</td>
      <td>16.4765</td>
      <td>0.0312</td>
      <td>200687</td>
      <td>20.54297</td>
      <td>0.05399</td>
      <td>0.04898</td>
      <td>...</td>
      <td>0.04113</td>
      <td>-0.03545</td>
      <td>0.10859</td>
      <td>0.017723</td>
      <td>3.73648</td>
      <td>4.71103</td>
      <td>2.74842</td>
      <td>17:50:53.0</td>
      <td>-32:20:16.2</td>
      <td>31.794682</td>
    </tr>
    <tr>
      <th>8</th>
      <td>267.782309</td>
      <td>-32.314028</td>
      <td>1216.120</td>
      <td>3348.355</td>
      <td>17.2033</td>
      <td>0.0518</td>
      <td>200950</td>
      <td>21.85581</td>
      <td>0.59356</td>
      <td>0.12884</td>
      <td>...</td>
      <td>-0.57910</td>
      <td>-0.02658</td>
      <td>1.15941</td>
      <td>0.005993</td>
      <td>3.53978</td>
      <td>4.56843</td>
      <td>2.42907</td>
      <td>17:51:07.8</td>
      <td>-32:18:50.5</td>
      <td>110.145859</td>
    </tr>
    <tr>
      <th>9</th>
      <td>267.835602</td>
      <td>-32.319098</td>
      <td>1976.117</td>
      <td>3264.169</td>
      <td>17.4567</td>
      <td>0.0626</td>
      <td>200985</td>
      <td>22.73157</td>
      <td>0.33644</td>
      <td>0.29747</td>
      <td>...</td>
      <td>-0.01420</td>
      <td>-0.29635</td>
      <td>0.59337</td>
      <td>0.006781</td>
      <td>4.54292</td>
      <td>6.00984</td>
      <td>3.99572</td>
      <td>17:51:20.5</td>
      <td>-32:19:08.8</td>
      <td>8.191454</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 29 columns</p>
</div>


We still have three redundant columns, let's remove them:


```python
x = df1.drop(['RA2000','DEC2000', 'Name_1'], axis = 1)

print('\n-----------------------------------Shape of the dataset---------------------------------')
display(x.shape)
```


    -----------------------------------Shape of the dataset---------------------------------



    (17515, 26)


This dataset is now 21 dimensional: there are 21 features describing each sample.  Here we will use principal component analysis (PCA), which is a fast linear dimensionality reduction technique. It reduces the dimensions of a dataset by projecting the data onto a lower-dimensional subspace.  Reducing the number of dimensions down to two (or three) makes it possible to plot a condensed view of high dimensional training set on a graph and often gain some important insight by visually detecting patterns, such as clusters.

For 2-dimensional subspace: The first principal component (PCA1) covers the maximum variance in the data and the second
principal component (PCA2) is orthogonal to the first principal component--all principal components are orthogonal to each other.

A vital part of using PCA in practice is the ability to estimate how many components are needed to describe the data.  The cumulative explained variance technique, which measures how well PCA preserves the content of the data will be employed.




```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np

data_rescaled = MinMaxScaler().fit_transform(x)

pca = PCA().fit(data_rescaled)

plt.figure(figsize=(12,8))
plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of components')
plt.ylabel('Variance (%)') #for each component
plt.title('Cumulative explained variance')

```




    Text(0.5, 1.0, 'Cumulative explained variance')




![](https://raw.githubusercontent.com/tshuna001/images/master/the_stars_21_1.png?token=AE5UC74GFIBFY5HWAWG5XRC57ELZY)


Here we see that our two-dimensional projection will lose a lot of information (as measured by the explained variance) and that we’d need about 6 components to retain about 90$\%$ of the variance.

The curve above quantifies how much of the total, 21-dimensional variance is contained within the first N components.

`We will convert our dataset to two & three dimension:`


```python

model = PCA(n_components = 2).fit(data_rescaled)  #project from 29 to 2 dimensions
x_2D = model.transform(data_rescaled) #Transform the data to two dimensions

modell = PCA(n_components = 3).fit(data_rescaled) #project from 29 to 3 dimensions
x_3D = modell.transform(data_rescaled) #Transform the data to three dimensions


```


```python

```


```python
x['PCA1'] = x_2D[:, 0]
x['PCA2'] = x_2D[:, 1]

x['x1'] = x_3D[:, 0]
x['x2'] = x_3D[:, 1]
x['z'] = x_3D[:, 2]

```


```python
display(x.head(10))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RA_1</th>
      <th>DEC_1</th>
      <th>X_1</th>
      <th>Y_1</th>
      <th>MAG</th>
      <th>MAGe</th>
      <th>Mean_Mag_1</th>
      <th>RMS_1</th>
      <th>Expected_RMS_1</th>
      <th>Alarm_2</th>
      <th>...</th>
      <th>Period_1_7</th>
      <th>AOV_1_7</th>
      <th>AOV_SNR_1_7</th>
      <th>AOV_NEG_LN_FAP_1_7</th>
      <th>lspermin</th>
      <th>PCA1</th>
      <th>PCA2</th>
      <th>x1</th>
      <th>x2</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>267.637420</td>
      <td>-32.505489</td>
      <td>1309.394</td>
      <td>117.711</td>
      <td>11.6591</td>
      <td>0.0026</td>
      <td>15.01506</td>
      <td>0.01028</td>
      <td>0.00179</td>
      <td>6.39196</td>
      <td>...</td>
      <td>0.015565</td>
      <td>3.41553</td>
      <td>4.67124</td>
      <td>2.28794</td>
      <td>113.177390</td>
      <td>0.605842</td>
      <td>-0.486187</td>
      <td>0.605842</td>
      <td>-0.486187</td>
      <td>0.286910</td>
    </tr>
    <tr>
      <th>1</th>
      <td>267.568298</td>
      <td>-32.472861</td>
      <td>324.079</td>
      <td>665.137</td>
      <td>10.8853</td>
      <td>0.0018</td>
      <td>14.18320</td>
      <td>0.00339</td>
      <td>0.00118</td>
      <td>5.10836</td>
      <td>...</td>
      <td>0.014812</td>
      <td>5.02959</td>
      <td>6.17483</td>
      <td>4.81995</td>
      <td>92.062210</td>
      <td>0.276313</td>
      <td>-0.506676</td>
      <td>0.276313</td>
      <td>-0.506676</td>
      <td>-0.206979</td>
    </tr>
    <tr>
      <th>2</th>
      <td>267.555268</td>
      <td>-32.285918</td>
      <td>126.340</td>
      <td>3818.489</td>
      <td>16.5748</td>
      <td>0.0332</td>
      <td>20.58942</td>
      <td>0.27845</td>
      <td>0.05443</td>
      <td>7.80936</td>
      <td>...</td>
      <td>0.017541</td>
      <td>5.07752</td>
      <td>8.06232</td>
      <td>4.88935</td>
      <td>104.545224</td>
      <td>0.586485</td>
      <td>0.137347</td>
      <td>0.586485</td>
      <td>0.137347</td>
      <td>-0.610663</td>
    </tr>
    <tr>
      <th>3</th>
      <td>267.758599</td>
      <td>-32.485136</td>
      <td>886.993</td>
      <td>462.508</td>
      <td>17.1670</td>
      <td>0.0503</td>
      <td>21.99514</td>
      <td>0.18661</td>
      <td>0.15743</td>
      <td>0.66387</td>
      <td>...</td>
      <td>0.007814</td>
      <td>3.24337</td>
      <td>3.98715</td>
      <td>1.99459</td>
      <td>18.891792</td>
      <td>-0.652359</td>
      <td>-0.498145</td>
      <td>-0.652359</td>
      <td>-0.498145</td>
      <td>-0.029480</td>
    </tr>
    <tr>
      <th>4</th>
      <td>267.797234</td>
      <td>-32.472325</td>
      <td>1436.175</td>
      <td>679.620</td>
      <td>12.2333</td>
      <td>0.0034</td>
      <td>15.69679</td>
      <td>0.00462</td>
      <td>0.00248</td>
      <td>3.85503</td>
      <td>...</td>
      <td>0.006415</td>
      <td>4.39385</td>
      <td>4.75353</td>
      <td>3.86889</td>
      <td>71.308296</td>
      <td>0.060235</td>
      <td>-0.349633</td>
      <td>0.060235</td>
      <td>-0.349633</td>
      <td>0.261898</td>
    </tr>
    <tr>
      <th>5</th>
      <td>267.767202</td>
      <td>-32.368453</td>
      <td>1003.500</td>
      <td>2430.221</td>
      <td>15.2115</td>
      <td>0.0146</td>
      <td>18.97295</td>
      <td>0.02625</td>
      <td>0.01531</td>
      <td>5.04427</td>
      <td>...</td>
      <td>0.038873</td>
      <td>4.48847</td>
      <td>4.89371</td>
      <td>3.91457</td>
      <td>97.136510</td>
      <td>0.355485</td>
      <td>-0.035592</td>
      <td>0.355485</td>
      <td>-0.035592</td>
      <td>-0.109320</td>
    </tr>
    <tr>
      <th>6</th>
      <td>267.768327</td>
      <td>-32.367157</td>
      <td>1019.457</td>
      <td>2452.106</td>
      <td>16.0023</td>
      <td>0.0231</td>
      <td>20.05330</td>
      <td>0.05390</td>
      <td>0.03288</td>
      <td>2.51186</td>
      <td>...</td>
      <td>0.010505</td>
      <td>5.27810</td>
      <td>6.39846</td>
      <td>5.05030</td>
      <td>92.754403</td>
      <td>0.266163</td>
      <td>-0.027926</td>
      <td>0.266163</td>
      <td>-0.027926</td>
      <td>-0.112835</td>
    </tr>
    <tr>
      <th>7</th>
      <td>267.720869</td>
      <td>-32.337827</td>
      <td>341.550</td>
      <td>2945.095</td>
      <td>16.4765</td>
      <td>0.0312</td>
      <td>20.54297</td>
      <td>0.05399</td>
      <td>0.04898</td>
      <td>1.10537</td>
      <td>...</td>
      <td>0.017723</td>
      <td>3.73648</td>
      <td>4.71103</td>
      <td>2.74842</td>
      <td>31.794682</td>
      <td>-0.469403</td>
      <td>-0.028505</td>
      <td>-0.469403</td>
      <td>-0.028505</td>
      <td>-0.486495</td>
    </tr>
    <tr>
      <th>8</th>
      <td>267.782309</td>
      <td>-32.314028</td>
      <td>1216.120</td>
      <td>3348.355</td>
      <td>17.2033</td>
      <td>0.0518</td>
      <td>21.85581</td>
      <td>0.59356</td>
      <td>0.12884</td>
      <td>10.07342</td>
      <td>...</td>
      <td>0.005993</td>
      <td>3.53978</td>
      <td>4.56843</td>
      <td>2.42907</td>
      <td>110.145859</td>
      <td>0.659339</td>
      <td>0.190395</td>
      <td>0.659339</td>
      <td>0.190395</td>
      <td>-0.102460</td>
    </tr>
    <tr>
      <th>9</th>
      <td>267.835602</td>
      <td>-32.319098</td>
      <td>1976.117</td>
      <td>3264.169</td>
      <td>17.4567</td>
      <td>0.0626</td>
      <td>22.73157</td>
      <td>0.33644</td>
      <td>0.29747</td>
      <td>-0.58983</td>
      <td>...</td>
      <td>0.006781</td>
      <td>4.54292</td>
      <td>6.00984</td>
      <td>3.99572</td>
      <td>8.191454</td>
      <td>-0.756770</td>
      <td>0.265002</td>
      <td>-0.756770</td>
      <td>0.265002</td>
      <td>0.161073</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 31 columns</p>
</div>


# Visualise data in 2D


```python
plt.figure(figsize=(15,10))
plt.scatter(x['PCA1'], x['PCA2'], marker = '.')


plt.xlabel('PCA1')
plt.ylabel('PCA2')
```




    Text(0, 0.5, 'PCA2')




![](https://raw.githubusercontent.com/tshuna001/images/master/the_stars_29_1.png?token=AE5UC75AYWVRCMKAQF2KQKS57EMDQ)


# Visualise data in 3D:


```python
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure(figsize = (15, 10))
ax = fig.add_subplot(111, projection = '3d')

ax.scatter(x['x1'], x['x2'], x['z'], cmap = plt.cm.hot)
ax.set_xlabel("PCA1", fontsize=12)
ax.set_ylabel("PCA2", fontsize=12)
ax.set_zlabel("PCA3", fontsize=12)


```




    Text(0.5, 0, 'PCA3')




![](https://raw.githubusercontent.com/tshuna001/images/master/the_stars_31_1.png?token=AE5UC72CWBEDAVCLS7DXTKK57EMHE)


### Let's consider some clustering algorithms:

We will instantiates different algorithms

# 1. K-Means


```python
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

from sklearn.cluster import KMeans


```

## Optimal number of clusters and cluster evaluation:

We shall use `the elbow method` to determine the optimal number of clusters in k-means clustering.



```python
from scipy.spatial.distance import cdist, pdist
import numpy as np
# Avg. within-cluster sum of squares

#
K = np.arange(1, 15)

model1 = [KMeans(n_clusters = k).fit(data_rescaled) for k in K]

#centroids
cent_ = [k.cluster_centers_ for k in model1]

#Compute distance between each pair of the two collections of inputs
D_k = [cdist(data_rescaled, centrds, 'euclidean') for centrds in cent_]

#compute minimum distance to each centroid
dist = [np.min(D, axis = 1) for D in D_k]

#average distance within cluster-sum
avgWithinSS = [sum(d)/data_rescaled.shape[0] for d in dist]

```


```python

```

`Let's view elbow curve`:


```python
# elbow curve - Avg. within-cluster sum of squares

plt.figure(figsize = (10,6))
plt.plot(K, avgWithinSS, 'b*-')
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
```




    Text(0, 0.5, 'Average within-cluster sum of squares')




![](https://raw.githubusercontent.com/tshuna001/images/master/the_stars_40_1.png?token=AE5UC77H6B3W477UTIWBRGK57EMKW)


It is important to scale the input features before you run K-Means, or else the clusters may be very stretched, and K-Means will perform poorly. However, scaling the features does not guarantee that all the clusters will be nice and spherical, but it generally improves things.

Scaling input features, in this work, will be performed in the `Pipeline` using `StandardScaler()` class.


```python
kmeans = Pipeline(steps=[('preprocessor', preprocessing.StandardScaler()),
                     ('model', KMeans(n_clusters = 3))])

model1 = kmeans.fit(x)

y_kmeans = model1.predict(x)

x['clusterKM'] = y_kmeans

x.clusterKM.unique()
```




    array([0, 2, 1])




```python

```


```python
plt.figure(figsize=(15,10))

plt.scatter(x['PCA1'], x['PCA2'], marker = '.', c = y_kmeans, s = 40, cmap = 'viridis')


plt.xlabel('PCA1')
plt.ylabel('PCA2')
```




    Text(0, 0.5, 'PCA2')




![](https://raw.githubusercontent.com/tshuna001/images/master/the_stars_44_1.png?token=AE5UC7ZMEPIB7QZTAJA64UC57EMN2)



```python

```


```python
fig = plt.figure(figsize = (15, 10))
ax = fig.add_subplot(111, projection = '3d')

ax.scatter(x['x1'], x['x2'], x['z'], c = y_kmeans, cmap = 'viridis')
ax.set_xlabel("PCA1", fontsize=12)
ax.set_ylabel("PCA2", fontsize=12)
ax.set_zlabel("PCA3", fontsize=12)

```




    Text(0.5, 0, 'PCA3')




![](https://raw.githubusercontent.com/tshuna001/images/master/the_stars_46_1.png?token=AE5UC7YCNB2TABIUSJJO7DC57EMP6)


The figures above shows that, the dataset contain 3 groups of stars.


```python
display(x['clusterKM'].value_counts())

#x.clusterGMM.value_counts().plot('bar')


x['clusterKM'].value_counts().plot(kind = 'pie',
                                 explode=(0,0,0.1), autopct='%1.2f%%',
                                 shadow=True,
                                 legend = True, figsize= (6,4), labels=None)

plt.tight_layout()
```


    0    9756
    1    5451
    2    2308
    Name: clusterKM, dtype: int64



![](https://raw.githubusercontent.com/tshuna001/images/master/the_stars_48_1.png?token=AE5UC75SAKPIKMYSUIMZ5S257EMSG)


## 2 Gaussian Mixture Model:

A Gaussian mixture model (GMM) is a probabilistic model that assumes that the instances were generated from a mixture of several Gaussian distributions whose parameters are unknown. All the instances generated from a single Gaussian distribution form a cluster that typically looks like an ellipsoid.

This class relies on the Expectation-Maximization (EM) algorithm, which has many similarities with the K-Means algorithm: it also initializes the cluster parameters randomly, then it repeats two steps until convergence.

Just like K-Means, EM can end up converging to poor solutions, so it needs to be run several times, keeping only the
best solution. This is why we set `n_init` to 10.


```python
from sklearn.mixture import GaussianMixture
```

### Selecting the number of components in a classical Gaussian Mixture Mode.

We shall use the Bayesian information criterion (BIC) to select the number of components in a Gaussian Mixture in an efficient way.  


```python

gm_bic= []
gm_score=[]

for i in range(1,10):

    gm = GaussianMixture(n_components = i, n_init = 10, tol = 1e-3, max_iter = 1000).fit(data_rescaled)

    print("BIC for number of cluster(s) {}: {}".format(i,gm.bic(data_rescaled)))
    print("Log-likelihood score for number of cluster(s) {}: {}".format(i,gm.score(data_rescaled)))
    print("-"*100)

    gm_bic.append(-gm.bic(data_rescaled))
    gm_score.append(gm.score(data_rescaled))



plt.figure(figsize = (12,8))
plt.title("The Gaussian Mixture model BIC \nfor determining number of clusters\n",fontsize=16)
plt.scatter(x = [i for i in range(1,10)],y = np.log(gm_bic), s = 150, edgecolor='k')
plt.xlabel("Number of clusters",fontsize = 14)
plt.ylabel("Log of Gaussian mixture BIC score",fontsize = 15)
plt.yticks(fontsize = 15)
plt.show()

```

    BIC for number of cluster(s) 1: -1676190.6216027052
    Log-likelihood score for number of cluster(s) 1: 47.95530168653835
    ----------------------------------------------------------------------------------------------------
    BIC for number of cluster(s) 2: -2171495.710339565
    Log-likelihood score for number of cluster(s) 2: 62.20019052543439
    ----------------------------------------------------------------------------------------------------
    BIC for number of cluster(s) 3: -2300486.7314481805
    Log-likelihood score for number of cluster(s) 3: 65.9879264203326
    ----------------------------------------------------------------------------------------------------
    BIC for number of cluster(s) 4: -2353688.393926406
    Log-likelihood score for number of cluster(s) 4: 67.6121065450227
    ----------------------------------------------------------------------------------------------------
    BIC for number of cluster(s) 5: -2399421.322070661
    Log-likelihood score for number of cluster(s) 5: 69.02307701130657
    ----------------------------------------------------------------------------------------------------
    BIC for number of cluster(s) 6: -2413156.6007096865
    Log-likelihood score for number of cluster(s) 6: 69.52061186510886
    ----------------------------------------------------------------------------------------------------
    BIC for number of cluster(s) 7: -2425277.5878507434
    Log-likelihood score for number of cluster(s) 7: 69.97206360449583
    ----------------------------------------------------------------------------------------------------
    BIC for number of cluster(s) 8: -2445482.0796555667
    Log-likelihood score for number of cluster(s) 8: 70.65427482614848
    ----------------------------------------------------------------------------------------------------
    BIC for number of cluster(s) 9: -2447326.030156673
    Log-likelihood score for number of cluster(s) 9: 70.81234841423796
    ----------------------------------------------------------------------------------------------------



![](https://raw.githubusercontent.com/tshuna001/images/master/the_stars_52_1.png?token=AE5UC725PTQ57ODXQYWJVT257EMUI)


`We shall take 3, as the optimal number of cluster: n_components = 3`


```python
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


GMM = Pipeline(steps = [('preprocessor', preprocessing.StandardScaler()),
                     ('model', GaussianMixture(n_components = 3, n_init = 10, max_iter = 1000, covariance_type = 'full'))])

model = GMM.fit(x)

#print(model.score(x))

y_gmm = model.predict(x)

x['clusterGMM'] = y_gmm
```


```python

```

Check whether or not the algorithm converged and how many iterations it took:


```python
l = [GaussianMixture(n_components = 10, n_init = i, max_iter = 1000,
                     covariance_type = 'full').fit(x).converged_ for i in range(1, 10)]
l
```




    [True, True, True, True, True, True, True, True, True]



The algorithm took one iteration to converge.

`Let's view our 2D & 3D figures:`


```python

plt.figure(figsize=(15,10))

plt.scatter(x['PCA1'], x['PCA2'], marker = '.', c = y_gmm, s = 40, cmap = 'viridis')


plt.xlabel('PCA1')
plt.ylabel('PCA2')

```




    Text(0, 0.5, 'PCA2')




![](https://raw.githubusercontent.com/tshuna001/images/master/the_stars_60_1.png?token=AE5UC73TTHWQYC52JNCBWA257EMWQ)



```python
fig = plt.figure(figsize = (15, 10))
ax = fig.add_subplot(111, projection = '3d')

ax.scatter(x['x1'], x['x2'], x['z'], c = y_gmm, cmap = 'viridis')
ax.set_xlabel("PCA1", fontsize=12)
ax.set_ylabel("PCA2", fontsize=12)
ax.set_zlabel("PCA3", fontsize=12)
```




    Text(0.5, 0, 'PCA3')




![](https://raw.githubusercontent.com/tshuna001/images/master/the_stars_61_1.png?token=AE5UC72VVF7VEL57YKHNTZ257EMZG)


The figure above shows that, the dataset contain 3 groups of stars.


```python
display(x.clusterGMM.value_counts())

#x.clusterGMM.value_counts().plot('bar')


x['clusterGMM'].value_counts().plot(kind = 'pie',
                                 explode=(0,0,0.1), autopct='%1.2f%%',
                                 shadow=True,
                                 legend = True, figsize= (6,4), labels=None)

plt.tight_layout()

```


    1    9756
    2    5451
    0    2308
    Name: clusterGMM, dtype: int64



![](https://raw.githubusercontent.com/tshuna001/images/master/the_stars_63_1.png?token=AE5UC74EO6IFNRKYLTVPEGS57EM24)


### Conclusion from machine model:

Both models performs better!


```python

```
