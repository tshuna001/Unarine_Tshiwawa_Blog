---
layout: "post"
author: "Unarine Tshiwawa"
---
## Identifying the problem:

In this task, I seek to demonstrate how supervised machine leaning can be used in a labelled dataset.

We have a large dataset containing clusters of stars that are labelled.  The data set will be divided into three sets, training, validation and test set.  In this exercise, we will use a model trained on the training set containing different members of star clusters to classify different kinds of star clusters in the test set.

#### Main point: we are trying to build a model that will predict new labels in a new data.

Clearly, we see that this is a machine leaning - classification - type of a problem.  An output to this problem is a categorical quantity.

Objective is to demonstrate:

    - binary classification
    - cross-validation
    - hyperparameter searching
    - evaluations
    - predictions




```python
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
%matplotlib inline
import os
from IPython.display import display
import seaborn as sns; sns.set('notebook')
import warnings
warnings.filterwarnings('ignore')
import pandas_profiling
```

Data source: [Gaia](https://www.cosmos.esa.int/web/gaia/data)

Read data:


```python
df = pd.read_csv('Gaiadr2.csv')

df1 = df.copy()
```

Dataset information:

The info() method is useful to get a quick description of the data


```python
display(df1.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 236620 entries, 0 to 236619
    Data columns (total 85 columns):
    RAdeg                      236620 non-null float64
    DEdeg                      236620 non-null float64
    e_RAdeg                    236620 non-null float64
    e_DEdeg                    236620 non-null float64
    HIP                        236620 non-null int64
    TYC2                       229789 non-null object
    Source                     236620 non-null int64
    plx                        236620 non-null float64
    e_plx                      236620 non-null float64
    pmRA                       236620 non-null float64
    e_pmRA                     236620 non-null float64
    pmDE                       236620 non-null float64
    e_pmDE                     236620 non-null float64
    NobsG                      236620 non-null int64
    Gmag                       236620 non-null float64
    2MASSID                    236620 non-null int64
    RA2deg                     236620 non-null float64
    DE2deg                     236620 non-null float64
    UCAC4                      236620 non-null int64
    RAUdeg                     236620 non-null float64
    DEUdeg                     236620 non-null float64
    pmRAU                      236620 non-null float64
    e_pmRAU                    236620 non-null float64
    pmDEU                      236620 non-null float64
    e_pmDEU                    236620 non-null float64
    Jmag                       236620 non-null float64
    Hmag                       236620 non-null float64
    Kmag                       236620 non-null float64
    Bmag                       236620 non-null float64
    Vmag                       236620 non-null float64
    gmag                       236620 non-null float64
    rmag                       236620 non-null float64
    imag                       236620 non-null float64
    RADEcor                    236620 non-null float64
    RAplxcor                   236620 non-null float64
    RApmRAcor                  236620 non-null float64
    RApmDEcor                  236620 non-null float64
    DEplxcor                   236620 non-null float64
    DEpmRAcor                  236620 non-null float64
    DEpmDEcor                  236620 non-null float64
    plxpmRAcor                 236620 non-null float64
    plxpmDEcor                 236620 non-null float64
    pmRApmDEcor                236620 non-null float64
    Cluster                    236620 non-null object
    PMused                     236620 non-null object
    Pfinal                     236620 non-null float64
    distL1350                  236620 non-null float64
    disterrL1350               236620 non-null object
    MG                         236620 non-null float64
    ra_epoch2000               236620 non-null float64
    dec_epoch2000              236620 non-null float64
    errHalfMaj                 236620 non-null float64
    errHalfMin                 236620 non-null float64
    errPosAng                  236620 non-null float64
    source_id                  236620 non-null int64
    ra                         236620 non-null float64
    ra_error                   236620 non-null float64
    dec                        236620 non-null float64
    dec_error                  236620 non-null float64
    parallax                   236541 non-null float64
    parallax_error             236541 non-null float64
    pmra_x                     236541 non-null float64
    pmra_error                 236541 non-null float64
    pmdec                      236541 non-null float64
    pmdec_error                236541 non-null float64
    duplicated_source          236620 non-null bool
    phot_g_mean_flux           236620 non-null float64
    phot_g_mean_flux_error     236620 non-null float64
    phot_g_mean_mag            236620 non-null float64
    phot_bp_mean_flux          236473 non-null float64
    phot_bp_mean_flux_error    236473 non-null float64
    phot_bp_mean_mag           236473 non-null float64
    phot_rp_mean_flux          236473 non-null float64
    phot_rp_mean_flux_error    236473 non-null float64
    phot_rp_mean_mag           236473 non-null float64
    bp_rp                      236473 non-null float64
    radial_velocity            135897 non-null float64
    radial_velocity_error      135897 non-null float64
    rv_nb_transits             236620 non-null int64
    teff_val                   236473 non-null float64
    a_g_val                    149023 non-null float64
    e_bp_min_rp_val            149023 non-null float64
    radius_val                 204314 non-null float64
    lum_val                    204314 non-null float64
    angDist                    236620 non-null float64
    dtypes: bool(1), float64(73), int64(7), object(4)
    memory usage: 151.9+ MB



    None


- NB: There are 236620 instances in the dataset, which means that it is very small by Machine Learning standards, but it’s perfect to get started.

- All attributes are numerical, except the `TYC2`, `Cluster`, `PMused ` and `disterrL1350` field are catergotical and `duplicated_source` field is boolean.

### Take a Quick Look at the Data Structure

Let’s take a look at the top five rows using the DataFrame’s head() method


```python
display(df1.head())
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
      <th>RAdeg</th>
      <th>DEdeg</th>
      <th>e_RAdeg</th>
      <th>e_DEdeg</th>
      <th>HIP</th>
      <th>TYC2</th>
      <th>Source</th>
      <th>plx</th>
      <th>e_plx</th>
      <th>pmRA</th>
      <th>...</th>
      <th>bp_rp</th>
      <th>radial_velocity</th>
      <th>radial_velocity_error</th>
      <th>rv_nb_transits</th>
      <th>teff_val</th>
      <th>a_g_val</th>
      <th>e_bp_min_rp_val</th>
      <th>radius_val</th>
      <th>lum_val</th>
      <th>angDist</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>300.18040</td>
      <td>-11.480575</td>
      <td>0.335450</td>
      <td>0.153877</td>
      <td>0</td>
      <td>5746-2488-1</td>
      <td>4189191459910407168</td>
      <td>1.636349</td>
      <td>0.389813</td>
      <td>7.722513</td>
      <td>...</td>
      <td>2.061183</td>
      <td>-13.96</td>
      <td>0.17</td>
      <td>16</td>
      <td>3907.37</td>
      <td>0.9492</td>
      <td>0.4130</td>
      <td>31.25</td>
      <td>205.109</td>
      <td>0.087258</td>
    </tr>
    <tr>
      <th>1</th>
      <td>300.19095</td>
      <td>-11.627663</td>
      <td>0.245796</td>
      <td>0.117092</td>
      <td>0</td>
      <td>5746-2191-1</td>
      <td>4189184794121166080</td>
      <td>1.401285</td>
      <td>0.340852</td>
      <td>2.877548</td>
      <td>...</td>
      <td>0.695882</td>
      <td>-10.71</td>
      <td>8.75</td>
      <td>12</td>
      <td>6358.00</td>
      <td>0.6735</td>
      <td>0.3627</td>
      <td>1.47</td>
      <td>3.163</td>
      <td>0.031192</td>
    </tr>
    <tr>
      <th>2</th>
      <td>300.08572</td>
      <td>-11.700761</td>
      <td>0.419372</td>
      <td>0.176058</td>
      <td>0</td>
      <td>5746-2457-1</td>
      <td>4189178265770877824</td>
      <td>0.939089</td>
      <td>0.588407</td>
      <td>2.269324</td>
      <td>...</td>
      <td>1.319349</td>
      <td>-41.42</td>
      <td>0.19</td>
      <td>9</td>
      <td>4761.46</td>
      <td>0.3520</td>
      <td>0.1613</td>
      <td>7.86</td>
      <td>28.574</td>
      <td>0.078189</td>
    </tr>
    <tr>
      <th>3</th>
      <td>299.70734</td>
      <td>-11.233611</td>
      <td>0.206056</td>
      <td>0.085100</td>
      <td>0</td>
      <td>5742-1871-1</td>
      <td>4189447268161556992</td>
      <td>1.138048</td>
      <td>0.336019</td>
      <td>-6.546313</td>
      <td>...</td>
      <td>1.322530</td>
      <td>101.46</td>
      <td>0.82</td>
      <td>9</td>
      <td>4658.56</td>
      <td>0.3000</td>
      <td>0.1405</td>
      <td>8.58</td>
      <td>31.260</td>
      <td>0.204904</td>
    </tr>
    <tr>
      <th>4</th>
      <td>299.78894</td>
      <td>-11.727871</td>
      <td>0.243463</td>
      <td>0.183640</td>
      <td>0</td>
      <td>5746-632-1</td>
      <td>4189344326385432192</td>
      <td>6.234146</td>
      <td>0.253613</td>
      <td>46.162357</td>
      <td>...</td>
      <td>0.808651</td>
      <td>-34.89</td>
      <td>0.37</td>
      <td>9</td>
      <td>5817.70</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.91</td>
      <td>3.763</td>
      <td>0.684125</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 85 columns</p>
</div>


Total number of clusters in the sample:


```python
df1['Cluster'].nunique()
```




    128



So we can find out what categories exist and how many clusters belong to each category by using the value_counts() method:


```python
df1['Cluster'].value_counts()
```




    Platais_8        24984
    Platais_10       12971
    ASCC_123         10994
    Platais_9        10966
    Ruprecht_147      9317
    Alessi_9          9074
    Alessi_13         8991
    Chereul_1         8965
    Alessi_5          6445
    Platais_3         6366
    Stock_2           5701
    Stock_1           4290
    NGC_2451B         4280
    BH_99             4243
    Collinder_135     4040
    Alessi_3          3947
    Alessi_6          3543
    Stock_12          3478
    Ruprecht_98       3365
    ASCC_113          2894
    ASCC_99           2855
    Roslund_6         2702
    Turner_5          2638
    NGC_2527          2607
    Trumpler_10       2544
    Collinder_359     2371
    Alessi_21         2284
    ASCC_41           2219
    NGC_3228          2004
    Stock_10          1833
                     ...  
    NGC_4609           388
    NGC_3680           387
    NGC_1647           385
    NGC_6811           377
    NGC_5617           360
    NGC_1912           352
    NGC_2670           343
    ASCC_10            340
    NGC_2548           331
    Ruprecht_1         330
    Roslund_3          327
    NGC_2567           306
    NGC_6866           297
    NGC_2477           295
    NGC_6793           245
    NGC_5138           234
    NGC_1960           226
    vdBergh_92         221
    NGC_2360           214
    NGC_2539           191
    NGC_2215           187
    NGC_2244           169
    NGC_2099           159
    NGC_1778           148
    NGC_6694           117
    NGC_2682           112
    Melotte_101         93
    NGC_6705            91
    NGC_6604            87
    Trumpler_33         78
    Name: Cluster, Length: 128, dtype: int64



Basic statistics: Summary of each numerical attribute.


```python
display(df1.describe())
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
      <th>RAdeg</th>
      <th>DEdeg</th>
      <th>e_RAdeg</th>
      <th>e_DEdeg</th>
      <th>HIP</th>
      <th>Source</th>
      <th>plx</th>
      <th>e_plx</th>
      <th>pmRA</th>
      <th>e_pmRA</th>
      <th>...</th>
      <th>bp_rp</th>
      <th>radial_velocity</th>
      <th>radial_velocity_error</th>
      <th>rv_nb_transits</th>
      <th>teff_val</th>
      <th>a_g_val</th>
      <th>e_bp_min_rp_val</th>
      <th>radius_val</th>
      <th>lum_val</th>
      <th>angDist</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>2.366200e+05</td>
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>...</td>
      <td>236473.000000</td>
      <td>135897.000000</td>
      <td>135897.000000</td>
      <td>236620.000000</td>
      <td>236473.000000</td>
      <td>149023.000000</td>
      <td>149023.000000</td>
      <td>204314.000000</td>
      <td>204314.000000</td>
      <td>236620.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>180.825870</td>
      <td>-14.879140</td>
      <td>0.355299</td>
      <td>0.346581</td>
      <td>1578.708516</td>
      <td>4.057617e+18</td>
      <td>1.830292</td>
      <td>0.365872</td>
      <td>-3.104849</td>
      <td>1.445930</td>
      <td>...</td>
      <td>0.899647</td>
      <td>2.202646</td>
      <td>1.499619</td>
      <td>4.931333</td>
      <td>6062.430660</td>
      <td>0.606305</td>
      <td>0.301194</td>
      <td>8.970707</td>
      <td>88.907142</td>
      <td>0.177691</td>
    </tr>
    <tr>
      <th>std</th>
      <td>90.163169</td>
      <td>45.426534</td>
      <td>0.394171</td>
      <td>0.382129</td>
      <td>10523.348859</td>
      <td>1.873770e+18</td>
      <td>1.632959</td>
      <td>0.145089</td>
      <td>11.181830</td>
      <td>1.010278</td>
      <td>...</td>
      <td>0.527076</td>
      <td>35.041674</td>
      <td>2.387136</td>
      <td>5.658857</td>
      <td>1476.617275</td>
      <td>0.400072</td>
      <td>0.201070</td>
      <td>15.479586</td>
      <td>530.294051</td>
      <td>0.163883</td>
    </tr>
    <tr>
      <th>min</th>
      <td>21.903664</td>
      <td>-71.945206</td>
      <td>0.055154</td>
      <td>0.063976</td>
      <td>0.000000</td>
      <td>1.248838e+17</td>
      <td>-4.349789</td>
      <td>0.205713</td>
      <td>-309.383060</td>
      <td>0.012372</td>
      <td>...</td>
      <td>-0.394635</td>
      <td>-487.290000</td>
      <td>0.110000</td>
      <td>0.000000</td>
      <td>3286.750000</td>
      <td>0.005300</td>
      <td>0.003000</td>
      <td>0.530000</td>
      <td>0.071000</td>
      <td>0.000157</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>114.362129</td>
      <td>-55.415602</td>
      <td>0.184421</td>
      <td>0.195051</td>
      <td>0.000000</td>
      <td>2.026967e+18</td>
      <td>0.853903</td>
      <td>0.270152</td>
      <td>-7.754609</td>
      <td>0.819638</td>
      <td>...</td>
      <td>0.533606</td>
      <td>-18.400000</td>
      <td>0.340000</td>
      <td>0.000000</td>
      <td>4809.780000</td>
      <td>0.294500</td>
      <td>0.144700</td>
      <td>1.740000</td>
      <td>4.324000</td>
      <td>0.073183</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>156.673930</td>
      <td>-34.119017</td>
      <td>0.250617</td>
      <td>0.252606</td>
      <td>0.000000</td>
      <td>5.236241e+18</td>
      <td>1.411869</td>
      <td>0.318205</td>
      <td>-3.189553</td>
      <td>1.208190</td>
      <td>...</td>
      <td>0.779076</td>
      <td>2.150000</td>
      <td>0.620000</td>
      <td>4.000000</td>
      <td>5895.670000</td>
      <td>0.549000</td>
      <td>0.272800</td>
      <td>3.000000</td>
      <td>13.500000</td>
      <td>0.124560</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>266.211767</td>
      <td>36.304339</td>
      <td>0.404249</td>
      <td>0.374406</td>
      <td>0.000000</td>
      <td>5.524287e+18</td>
      <td>2.316317</td>
      <td>0.404540</td>
      <td>1.379009</td>
      <td>1.772708</td>
      <td>...</td>
      <td>1.273891</td>
      <td>22.590000</td>
      <td>1.460000</td>
      <td>8.000000</td>
      <td>7039.620000</td>
      <td>0.847000</td>
      <td>0.418000</td>
      <td>10.400000</td>
      <td>50.643750</td>
      <td>0.220847</td>
    </tr>
    <tr>
      <th>max</th>
      <td>358.557920</td>
      <td>77.288320</td>
      <td>15.939135</td>
      <td>15.179961</td>
      <td>120132.000000</td>
      <td>6.881455e+18</td>
      <td>51.615630</td>
      <td>0.999922</td>
      <td>348.905120</td>
      <td>13.996302</td>
      <td>...</td>
      <td>7.099366</td>
      <td>478.050000</td>
      <td>20.000000</td>
      <td>75.000000</td>
      <td>9777.000000</td>
      <td>3.140500</td>
      <td>1.610300</td>
      <td>589.700000</td>
      <td>53737.550000</td>
      <td>0.999855</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 80 columns</p>
</div>


The count , mean , min , and max rows are self-explanatory.  Note that the null values are ignored. The std row shows the standard deviation, which measures how dispersed the values are. The 25%, 50%, and 75% rows show the corresponding percentiles: a percentile indicates the value below which a given percentage of observations in a group of observations falls

### Impute for missing values


```python
display(df1.isnull().sum())
```


    RAdeg                           0
    DEdeg                           0
    e_RAdeg                         0
    e_DEdeg                         0
    HIP                             0
    TYC2                         6831
    Source                          0
    plx                             0
    e_plx                           0
    pmRA                            0
    e_pmRA                          0
    pmDE                            0
    e_pmDE                          0
    NobsG                           0
    Gmag                            0
    2MASSID                         0
    RA2deg                          0
    DE2deg                          0
    UCAC4                           0
    RAUdeg                          0
    DEUdeg                          0
    pmRAU                           0
    e_pmRAU                         0
    pmDEU                           0
    e_pmDEU                         0
    Jmag                            0
    Hmag                            0
    Kmag                            0
    Bmag                            0
    Vmag                            0
                                ...  
    ra                              0
    ra_error                        0
    dec                             0
    dec_error                       0
    parallax                       79
    parallax_error                 79
    pmra_x                         79
    pmra_error                     79
    pmdec                          79
    pmdec_error                    79
    duplicated_source               0
    phot_g_mean_flux                0
    phot_g_mean_flux_error          0
    phot_g_mean_mag                 0
    phot_bp_mean_flux             147
    phot_bp_mean_flux_error       147
    phot_bp_mean_mag              147
    phot_rp_mean_flux             147
    phot_rp_mean_flux_error       147
    phot_rp_mean_mag              147
    bp_rp                         147
    radial_velocity            100723
    radial_velocity_error      100723
    rv_nb_transits                  0
    teff_val                      147
    a_g_val                     87597
    e_bp_min_rp_val             87597
    radius_val                  32306
    lum_val                     32306
    angDist                         0
    Length: 85, dtype: int64


Removing redundant columns:


```python

```


```python
df2 = df1.drop(['e_RAdeg','e_DEdeg','plx','e_plx','pmRA','e_pmRA','pmDE','HIP','TYC2','Source','2MASSID',
                'PMused','Pfinal','disterrL1350','duplicated_source','source_id','RA2deg', 'DE2deg',
                'ra_epoch2000','dec_epoch2000', 'errHalfMaj','errHalfMin','errPosAng','ra','ra_error',
                'dec','dec_error', ], axis = 1)

df2.head()
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
      <th>RAdeg</th>
      <th>DEdeg</th>
      <th>e_pmDE</th>
      <th>NobsG</th>
      <th>Gmag</th>
      <th>UCAC4</th>
      <th>RAUdeg</th>
      <th>DEUdeg</th>
      <th>pmRAU</th>
      <th>e_pmRAU</th>
      <th>...</th>
      <th>bp_rp</th>
      <th>radial_velocity</th>
      <th>radial_velocity_error</th>
      <th>rv_nb_transits</th>
      <th>teff_val</th>
      <th>a_g_val</th>
      <th>e_bp_min_rp_val</th>
      <th>radius_val</th>
      <th>lum_val</th>
      <th>angDist</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>300.18040</td>
      <td>-11.480575</td>
      <td>0.587125</td>
      <td>219</td>
      <td>7.624412</td>
      <td>61682627</td>
      <td>300.18036</td>
      <td>-11.480573</td>
      <td>9.4</td>
      <td>1.1</td>
      <td>...</td>
      <td>2.061183</td>
      <td>-13.96</td>
      <td>0.17</td>
      <td>16</td>
      <td>3907.37</td>
      <td>0.9492</td>
      <td>0.4130</td>
      <td>31.25</td>
      <td>205.109</td>
      <td>0.087258</td>
    </tr>
    <tr>
      <th>1</th>
      <td>300.19095</td>
      <td>-11.627663</td>
      <td>0.866389</td>
      <td>231</td>
      <td>12.635215</td>
      <td>61581376</td>
      <td>300.19095</td>
      <td>-11.627684</td>
      <td>0.2</td>
      <td>1.2</td>
      <td>...</td>
      <td>0.695882</td>
      <td>-10.71</td>
      <td>8.75</td>
      <td>12</td>
      <td>6358.00</td>
      <td>0.6735</td>
      <td>0.3627</td>
      <td>1.47</td>
      <td>3.163</td>
      <td>0.031192</td>
    </tr>
    <tr>
      <th>2</th>
      <td>300.08572</td>
      <td>-11.700761</td>
      <td>1.402877</td>
      <td>98</td>
      <td>10.881881</td>
      <td>61533071</td>
      <td>300.08572</td>
      <td>-11.700747</td>
      <td>4.3</td>
      <td>1.1</td>
      <td>...</td>
      <td>1.319349</td>
      <td>-41.42</td>
      <td>0.19</td>
      <td>9</td>
      <td>4761.46</td>
      <td>0.3520</td>
      <td>0.1613</td>
      <td>7.86</td>
      <td>28.574</td>
      <td>0.078189</td>
    </tr>
    <tr>
      <th>3</th>
      <td>299.70734</td>
      <td>-11.233611</td>
      <td>0.729939</td>
      <td>189</td>
      <td>11.155972</td>
      <td>61844543</td>
      <td>299.70737</td>
      <td>-11.233577</td>
      <td>-4.7</td>
      <td>1.7</td>
      <td>...</td>
      <td>1.322530</td>
      <td>101.46</td>
      <td>0.82</td>
      <td>9</td>
      <td>4658.56</td>
      <td>0.3000</td>
      <td>0.1405</td>
      <td>8.58</td>
      <td>31.260</td>
      <td>0.204904</td>
    </tr>
    <tr>
      <th>4</th>
      <td>299.78894</td>
      <td>-11.727871</td>
      <td>0.867933</td>
      <td>71</td>
      <td>9.283994</td>
      <td>61515591</td>
      <td>299.78876</td>
      <td>-11.727827</td>
      <td>47.2</td>
      <td>1.4</td>
      <td>...</td>
      <td>0.808651</td>
      <td>-34.89</td>
      <td>0.37</td>
      <td>9</td>
      <td>5817.70</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.91</td>
      <td>3.763</td>
      <td>0.684125</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 58 columns</p>
</div>




```python
df2.isnull().sum()
```




    RAdeg                           0
    DEdeg                           0
    e_pmDE                          0
    NobsG                           0
    Gmag                            0
    UCAC4                           0
    RAUdeg                          0
    DEUdeg                          0
    pmRAU                           0
    e_pmRAU                         0
    pmDEU                           0
    e_pmDEU                         0
    Jmag                            0
    Hmag                            0
    Kmag                            0
    Bmag                            0
    Vmag                            0
    gmag                            0
    rmag                            0
    imag                            0
    RADEcor                         0
    RAplxcor                        0
    RApmRAcor                       0
    RApmDEcor                       0
    DEplxcor                        0
    DEpmRAcor                       0
    DEpmDEcor                       0
    plxpmRAcor                      0
    plxpmDEcor                      0
    pmRApmDEcor                     0
    Cluster                         0
    distL1350                       0
    MG                              0
    parallax                       79
    parallax_error                 79
    pmra_x                         79
    pmra_error                     79
    pmdec                          79
    pmdec_error                    79
    phot_g_mean_flux                0
    phot_g_mean_flux_error          0
    phot_g_mean_mag                 0
    phot_bp_mean_flux             147
    phot_bp_mean_flux_error       147
    phot_bp_mean_mag              147
    phot_rp_mean_flux             147
    phot_rp_mean_flux_error       147
    phot_rp_mean_mag              147
    bp_rp                         147
    radial_velocity            100723
    radial_velocity_error      100723
    rv_nb_transits                  0
    teff_val                      147
    a_g_val                     87597
    e_bp_min_rp_val             87597
    radius_val                  32306
    lum_val                     32306
    angDist                         0
    dtype: int64



Droping all rows consisting of null:


```python
df3 = df2.dropna()
```

Take a quick look at the new data structure


```python
df3.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 88990 entries, 0 to 236610
    Data columns (total 58 columns):
    RAdeg                      88990 non-null float64
    DEdeg                      88990 non-null float64
    e_pmDE                     88990 non-null float64
    NobsG                      88990 non-null int64
    Gmag                       88990 non-null float64
    UCAC4                      88990 non-null int64
    RAUdeg                     88990 non-null float64
    DEUdeg                     88990 non-null float64
    pmRAU                      88990 non-null float64
    e_pmRAU                    88990 non-null float64
    pmDEU                      88990 non-null float64
    e_pmDEU                    88990 non-null float64
    Jmag                       88990 non-null float64
    Hmag                       88990 non-null float64
    Kmag                       88990 non-null float64
    Bmag                       88990 non-null float64
    Vmag                       88990 non-null float64
    gmag                       88990 non-null float64
    rmag                       88990 non-null float64
    imag                       88990 non-null float64
    RADEcor                    88990 non-null float64
    RAplxcor                   88990 non-null float64
    RApmRAcor                  88990 non-null float64
    RApmDEcor                  88990 non-null float64
    DEplxcor                   88990 non-null float64
    DEpmRAcor                  88990 non-null float64
    DEpmDEcor                  88990 non-null float64
    plxpmRAcor                 88990 non-null float64
    plxpmDEcor                 88990 non-null float64
    pmRApmDEcor                88990 non-null float64
    Cluster                    88990 non-null object
    distL1350                  88990 non-null float64
    MG                         88990 non-null float64
    parallax                   88990 non-null float64
    parallax_error             88990 non-null float64
    pmra_x                     88990 non-null float64
    pmra_error                 88990 non-null float64
    pmdec                      88990 non-null float64
    pmdec_error                88990 non-null float64
    phot_g_mean_flux           88990 non-null float64
    phot_g_mean_flux_error     88990 non-null float64
    phot_g_mean_mag            88990 non-null float64
    phot_bp_mean_flux          88990 non-null float64
    phot_bp_mean_flux_error    88990 non-null float64
    phot_bp_mean_mag           88990 non-null float64
    phot_rp_mean_flux          88990 non-null float64
    phot_rp_mean_flux_error    88990 non-null float64
    phot_rp_mean_mag           88990 non-null float64
    bp_rp                      88990 non-null float64
    radial_velocity            88990 non-null float64
    radial_velocity_error      88990 non-null float64
    rv_nb_transits             88990 non-null int64
    teff_val                   88990 non-null float64
    a_g_val                    88990 non-null float64
    e_bp_min_rp_val            88990 non-null float64
    radius_val                 88990 non-null float64
    lum_val                    88990 non-null float64
    angDist                    88990 non-null float64
    dtypes: float64(54), int64(3), object(1)
    memory usage: 40.1+ MB


Our sample now have 88990 instances, each consist of 63 features.  This is fairly enough by Machine Learning standards, and it’s perfect to get started.  Therefore we shall proceed with the exploratory data analysis.


```python

```

# 1. Exploratory Data Analysis

A histogram for each numerical attribute:


```python
df3.hist(bins=50, figsize=(30,25))
plt.show()
```

![png](https://drive.google.com/uc?export=view&id=1u-iHOEnJ9pA4sw9kf73eweGWOjsAVavE)


```python

```


```python
df3.Cluster.value_counts()
```




    Platais_8        9591
    Chereul_1        5487
    Alessi_13        5353
    Ruprecht_147     4055
    Platais_10       3735
    Platais_9        3708
    Alessi_9         3642
    ASCC_123         3477
    Platais_3        3356
    Alessi_3         2141
    Collinder_135    1889
    NGC_2451B        1581
    Stock_1          1553
    Stock_12         1485
    Alessi_5         1422
    Turner_5         1383
    Alessi_6         1362
    Stock_2          1330
    ASCC_113         1276
    ASCC_41          1065
    Collinder_359    1022
    Roslund_6         948
    Ruprecht_98       890
    BH_99             865
    NGC_2527          849
    ASCC_99           786
    ASCC_51           767
    Trumpler_10       758
    NGC_3228          757
    ASCC_112          750
                     ...
    NGC_6866          124
    NGC_2353          121
    NGC_2168          118
    NGC_6416          117
    NGC_1545          113
    NGC_4609          110
    NGC_6716          106
    NGC_2567          104
    NGC_2477           96
    IC_4725            95
    NGC_2670           94
    NGC_5617           90
    NGC_4103           87
    NGC_2539           73
    NGC_6793           70
    NGC_2682           67
    NGC_5138           67
    NGC_1912           64
    NGC_2360           60
    NGC_2215           56
    vdBergh_92         53
    NGC_1960           51
    NGC_2244           33
    NGC_1778           28
    NGC_2099           27
    Melotte_101        25
    NGC_6694           20
    Trumpler_33        13
    NGC_6705           10
    NGC_6604           10
    Name: Cluster, Length: 128, dtype: int64



## Sample selection for Machine Learning

 #### We are going to only select clusters which contains more than 2000 members


```python
df4 = df3[df3['Cluster'].str.contains('Platais_8|Chereul_1|Alessi_13|Ruprecht_147|Platais_10|Platais_9|Alessi_9|ASCC_123|Platais_3|Alessi_3', case=True,na=False)]

#reset index
df4 = df4.reset_index()
```

Let's view the number of unique clusters by using the value_counts() method:


```python
df4.Cluster.value_counts()
```




    Platais_8       9591
    Chereul_1       5487
    Alessi_13       5353
    Ruprecht_147    4055
    Platais_10      3735
    Platais_9       3708
    Alessi_9        3642
    ASCC_123        3477
    Platais_3       3356
    Alessi_3        2141
    Name: Cluster, dtype: int64



Summary of each numerical attribute after `sample selection`


```python
df4.describe()
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
      <th>index</th>
      <th>RAdeg</th>
      <th>DEdeg</th>
      <th>e_pmDE</th>
      <th>NobsG</th>
      <th>Gmag</th>
      <th>UCAC4</th>
      <th>RAUdeg</th>
      <th>DEUdeg</th>
      <th>pmRAU</th>
      <th>...</th>
      <th>bp_rp</th>
      <th>radial_velocity</th>
      <th>radial_velocity_error</th>
      <th>rv_nb_transits</th>
      <th>teff_val</th>
      <th>a_g_val</th>
      <th>e_bp_min_rp_val</th>
      <th>radius_val</th>
      <th>lum_val</th>
      <th>angDist</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>44545.000000</td>
      <td>44545.000000</td>
      <td>44545.000000</td>
      <td>44545.000000</td>
      <td>44545.000000</td>
      <td>44545.000000</td>
      <td>4.454500e+04</td>
      <td>44545.000000</td>
      <td>44545.000000</td>
      <td>44545.000000</td>
      <td>...</td>
      <td>44545.000000</td>
      <td>44545.000000</td>
      <td>44545.000000</td>
      <td>44545.000000</td>
      <td>44545.000000</td>
      <td>44545.000000</td>
      <td>44545.000000</td>
      <td>44545.000000</td>
      <td>44545.000000</td>
      <td>44545.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>112987.066427</td>
      <td>176.990576</td>
      <td>-16.775341</td>
      <td>1.216994</td>
      <td>134.977573</td>
      <td>10.801915</td>
      <td>4.886167e+07</td>
      <td>176.990602</td>
      <td>-16.775338</td>
      <td>-2.506010</td>
      <td>...</td>
      <td>1.158160</td>
      <td>0.928931</td>
      <td>1.227197</td>
      <td>9.216365</td>
      <td>5204.759310</td>
      <td>0.480608</td>
      <td>0.237349</td>
      <td>10.177536</td>
      <td>74.307798</td>
      <td>0.236688</td>
    </tr>
    <tr>
      <th>std</th>
      <td>72224.621547</td>
      <td>87.303770</td>
      <td>48.881238</td>
      <td>1.088160</td>
      <td>60.472568</td>
      <td>1.046743</td>
      <td>3.868362e+07</td>
      <td>87.303777</td>
      <td>48.881249</td>
      <td>14.332578</td>
      <td>...</td>
      <td>0.429306</td>
      <td>33.982241</td>
      <td>1.988719</td>
      <td>5.124055</td>
      <td>894.487403</td>
      <td>0.366602</td>
      <td>0.183149</td>
      <td>13.495322</td>
      <td>278.626901</td>
      <td>0.200888</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1453.000000</td>
      <td>39.205887</td>
      <td>-66.757520</td>
      <td>0.020972</td>
      <td>15.000000</td>
      <td>4.825314</td>
      <td>5.219950e+06</td>
      <td>39.205864</td>
      <td>-66.757515</td>
      <td>-102.400000</td>
      <td>...</td>
      <td>0.288623</td>
      <td>-448.410000</td>
      <td>0.110000</td>
      <td>2.000000</td>
      <td>3300.000000</td>
      <td>0.005300</td>
      <td>0.003000</td>
      <td>0.550000</td>
      <td>0.071000</td>
      <td>0.000655</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>41447.000000</td>
      <td>124.356290</td>
      <td>-55.607655</td>
      <td>0.668536</td>
      <td>87.000000</td>
      <td>10.245734</td>
      <td>1.642835e+07</td>
      <td>124.356346</td>
      <td>-55.607640</td>
      <td>-8.700000</td>
      <td>...</td>
      <td>0.770141</td>
      <td>-18.540000</td>
      <td>0.320000</td>
      <td>5.000000</td>
      <td>4482.050000</td>
      <td>0.198700</td>
      <td>0.097000</td>
      <td>1.390000</td>
      <td>2.341000</td>
      <td>0.095354</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>149555.000000</td>
      <td>145.877580</td>
      <td>-42.084763</td>
      <td>0.859277</td>
      <td>129.000000</td>
      <td>10.987285</td>
      <td>2.987989e+07</td>
      <td>145.877700</td>
      <td>-42.084827</td>
      <td>-2.400000</td>
      <td>...</td>
      <td>1.224521</td>
      <td>1.730000</td>
      <td>0.570000</td>
      <td>8.000000</td>
      <td>4873.750000</td>
      <td>0.396300</td>
      <td>0.193000</td>
      <td>8.630000</td>
      <td>31.830000</td>
      <td>0.170014</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>175612.000000</td>
      <td>258.287630</td>
      <td>50.192200</td>
      <td>1.294153</td>
      <td>171.000000</td>
      <td>11.547281</td>
      <td>1.035755e+08</td>
      <td>258.287660</td>
      <td>50.192190</td>
      <td>3.500000</td>
      <td>...</td>
      <td>1.460418</td>
      <td>21.120000</td>
      <td>1.180000</td>
      <td>12.000000</td>
      <td>5920.670000</td>
      <td>0.685000</td>
      <td>0.337000</td>
      <td>12.270000</td>
      <td>60.132000</td>
      <td>0.312108</td>
    </tr>
    <tr>
      <th>max</th>
      <td>209503.000000</td>
      <td>348.434500</td>
      <td>77.260574</td>
      <td>16.676441</td>
      <td>625.000000</td>
      <td>13.379665</td>
      <td>1.130387e+08</td>
      <td>348.434540</td>
      <td>77.260560</td>
      <td>73.600000</td>
      <td>...</td>
      <td>3.378872</td>
      <td>478.050000</td>
      <td>20.000000</td>
      <td>64.000000</td>
      <td>7998.330000</td>
      <td>3.140500</td>
      <td>1.610300</td>
      <td>380.440000</td>
      <td>31967.203000</td>
      <td>0.999831</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 58 columns</p>
</div>



Plot for each categorical attribute after sample selection:


```python
plt.figure(figsize = (15,5))
plt.subplot(1, 2, 1)
df4['Cluster'].value_counts().plot(kind = 'bar', title = 'Number of Cluster', rot = 45)

plt.subplot(1, 2, 2)
df4['Cluster'].value_counts().plot(kind = 'pie',
                                 autopct='%1.2f%%',
                                 shadow=True)

```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fef4faef978>




![png](https://drive.google.com/uc?export=view&id=1QXLiHzZj_Uakdu8G0r72-y9AxFUfjm35)

A histogram for each numerical attribute after sample selection:


```python
df4.hist(bins=50, figsize=(30,25))
plt.show()
```


![png](stars_updated_files/stars_updated_43_0.png)


# 2. Preprocessing

We are going to generate a feature matrix and target vector.

Note: Our target variable is a string, so we need to perform one-hot encouding technique to the target variable.



#### Feature matrix:


```python
x = df4.drop('Cluster', axis = 1)

```

#### Target variables:

Our target variables are categorical.  This means that categorical data must be converted to a numerical form using `onehot encoding` technique.


```python
df4['Cluster'] = df4['Cluster'].astype('category')
df4 = pd.get_dummies(df4)

```


```python
display(df4.head(6))
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
      <th>index</th>
      <th>RAdeg</th>
      <th>DEdeg</th>
      <th>e_pmDE</th>
      <th>NobsG</th>
      <th>Gmag</th>
      <th>UCAC4</th>
      <th>RAUdeg</th>
      <th>DEUdeg</th>
      <th>pmRAU</th>
      <th>...</th>
      <th>Cluster_ASCC_123</th>
      <th>Cluster_Alessi_13</th>
      <th>Cluster_Alessi_3</th>
      <th>Cluster_Alessi_9</th>
      <th>Cluster_Chereul_1</th>
      <th>Cluster_Platais_10</th>
      <th>Cluster_Platais_3</th>
      <th>Cluster_Platais_8</th>
      <th>Cluster_Platais_9</th>
      <th>Cluster_Ruprecht_147</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1453</td>
      <td>44.110203</td>
      <td>-43.595192</td>
      <td>1.136372</td>
      <td>111</td>
      <td>11.208868</td>
      <td>28496473</td>
      <td>44.110172</td>
      <td>-43.595203</td>
      <td>2.1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1454</td>
      <td>45.372612</td>
      <td>-42.736840</td>
      <td>1.352581</td>
      <td>121</td>
      <td>12.121879</td>
      <td>29296032</td>
      <td>45.372547</td>
      <td>-42.736862</td>
      <td>3.4</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1460</td>
      <td>45.003326</td>
      <td>-43.359417</td>
      <td>0.819278</td>
      <td>193</td>
      <td>10.238273</td>
      <td>28718887</td>
      <td>45.003456</td>
      <td>-43.359314</td>
      <td>-21.3</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1463</td>
      <td>45.100906</td>
      <td>-42.635580</td>
      <td>0.824140</td>
      <td>131</td>
      <td>11.200096</td>
      <td>29385371</td>
      <td>45.100822</td>
      <td>-42.635630</td>
      <td>6.4</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1464</td>
      <td>45.165424</td>
      <td>-42.671764</td>
      <td>0.939244</td>
      <td>202</td>
      <td>11.915485</td>
      <td>29353383</td>
      <td>45.165108</td>
      <td>-42.671880</td>
      <td>47.2</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1466</td>
      <td>45.520718</td>
      <td>-43.329730</td>
      <td>0.944784</td>
      <td>177</td>
      <td>11.725976</td>
      <td>28746451</td>
      <td>45.520664</td>
      <td>-43.329680</td>
      <td>4.1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>6 rows × 68 columns</p>
</div>



```python
y = df4.iloc[:,-10:]
display(y.head(10))
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
      <th>Cluster_ASCC_123</th>
      <th>Cluster_Alessi_13</th>
      <th>Cluster_Alessi_3</th>
      <th>Cluster_Alessi_9</th>
      <th>Cluster_Chereul_1</th>
      <th>Cluster_Platais_10</th>
      <th>Cluster_Platais_3</th>
      <th>Cluster_Platais_8</th>
      <th>Cluster_Platais_9</th>
      <th>Cluster_Ruprecht_147</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
labels = ['Cluster_ASCC_123','Cluster_Alessi_13','Cluster_Alessi_3','Cluster_Alessi_9','Cluster_Chereul_1',
          'Cluster_Platais_10','Cluster_Platais_3','Cluster_Platais_8','Cluster_Platais_9','Cluster_Ruprecht_147']
```

### Data splicing - data partitioning:  

We are going to split the data into three sets -  80$\%$ train set, 10$\%$ validation set, and 10% test set.  This means, 80% of the data will be set to train and optimize our machine learning model, 10% for model valoidation and the remaining 10$\%$ will be used to test the model.




```python
from sklearn.model_selection import train_test_split


#data splicing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify = y)
X_test, X_val, Y_test, Y_val = train_test_split(x_test, y_test, test_size = 0.5, random_state = 42)

print('---------------------Size of each dataset----------------------')
print(x_train.shape)
print(x_test.shape)
print(X_val.shape)

```

    ---------------------Size of each dataset----------------------
    (35636, 58)
    (8909, 58)
    (4455, 58)


Let's confirm again if data partition criteria was archieved!


```python
for dataset in [y_train, Y_val, Y_test]:


    print(round((len(dataset)/len(y))*100, 2))

print('>>>>>>>>>>>> Done!!<<<<<<<<<<<<<<')

```

    80.0
    10.0
    10.0
    >>>>>>>>>>>> Done!!<<<<<<<<<<<<<<


#### Getting data into shape:

Data are not usually presented to the machine learning algorithm in exactly the same raw form as it is found. Usually data are scaled to a specific range in a process called normalization.

We are going to make selected features to the same scale for optimal performance, which is often achieved by transforming the features in the range [0, 1]: standardize features by removing the mean and scaling to unit variance.  This will be done in the pipeline to run multiple processes in the order that they are listed. The purpose of the pipeline is to assemble several steps that can be cross-validated together while setting different parameters.

But, `Decision Trees` and `Random Forest` need very little data preparation. In particular, they don’t require feature scaling or centering at all.  However `K-Nearest Neighbor` is sensitive to the feature scales.

# 3. Machine Learning Models

Now, the input and output variables are ready for training. This is the part where different class will be instintiated and a model which score high accuracy in both training and testing set will be used for further predictions.

We shall use the `trainin set` which consitute 80% of the training dataset to train different models


```python
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(x_train, y_train)

print('-----------------------Decision Tree Classifier---------------------')
print('Accuracy in training set: {:.2f}' .format(accuracy_score(y_train, DT.predict(x_train))))
print('Accuracy in validation set: {:.2f}' .format(accuracy_score(Y_val, DT.predict(X_val))))


#Random forest
from sklearn.ensemble import RandomForestClassifier

#instantiate random forest class
RF = RandomForestClassifier()
RF.fit(x_train, y_train)

print('\n-----------------------Random Forest Classifier---------------------')
print('Accuracy in training set: {:.2f}' .format(accuracy_score(y_train, RF.predict(x_train))))
print('Accuracy in validation set: {:.2f}' .format(accuracy_score(Y_val, RF.predict(X_val))))

from sklearn.neighbors import KNeighborsClassifier
KNN = Pipeline(steps=[('preprocessor', preprocessing.StandardScaler()),
                     ('model', KNeighborsClassifier(n_neighbors = 10))])
KNN.fit(x_train, y_train)

print('\n---------------------K-Nearest Neighbor Classifier------------------')
print('Accuracy in training set: {:.2f}' .format(accuracy_score(y_train, KNN.predict(x_train))))
print('Accuracy in validation set: {:.2f}' .format(accuracy_score(Y_val, KNN.predict(X_val))))


```

    -----------------------Decision Tree Classifier---------------------
    Accuracy in training set: 1.00
    Accuracy in validation set: 1.00

    -----------------------Random Forest Classifier---------------------
    Accuracy in training set: 1.00
    Accuracy in validation set: 1.00

    ---------------------K-Nearest Neighbor Classifier------------------
    Accuracy in training set: 0.94
    Accuracy in validation set: 0.93


Clearly, `Decision Tree` and `Random Forest` Classifier are overfitting!


```python

```

# 4. Fine-Tune Models

Let's check parameters of each model:


```python
print("\033[1m"+'Decision Tree Classifier:'+"\033[10m")
display(DT.get_params)

print("\033[1m"+'Random Forest Classifier:'+"\033[10m")
display(RF.get_params)

print("\033[1m"+'K-Nearest Neighbors Classifier:'+"\033[10m")
display(KNN.get_params)
```

    Decision Tree Classifier:



    <bound method BaseEstimator.get_params of DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best')>


    Random Forest Classifier:



    <bound method BaseEstimator.get_params of RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)>


    K-Nearest Neighbors Classifier:



    <bound method Pipeline.get_params of Pipeline(memory=None,
         steps=[('preprocessor', StandardScaler(copy=True, with_mean=True, with_std=True)), ('model', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=None, n_neighbors=10, p=2,
               weights='uniform'))])>


## Grid Search

Let's fiddle with the hyperparameters manually, until you find a great combination of hyperparameter values. This can be very tedious work and time consuming to explore many combinations.

We shall use **Randomized Search**.  `RandomizedSearchCV` class can be used in much the same way as the `GridSearchCV` class, but instead of trying out all possible combinations, it evaluates a given number of random combinations by selecting a random value for each hyperparameter at every iteration.


```python
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV
```

#### Decision Tree


```python
params1 = {"max_depth": [3, 32, None], "min_samples_leaf": [np.random.randint(1,9)],
               "criterion": ["gini","entropy"],'max_depth': [1,32], "max_features": [2, 4, 6] }


DT_search = RandomizedSearchCV(DT, param_distributions=params1, random_state=0,
                               n_iter = 200, cv = 5, verbose = 1, n_jobs = 1, return_train_score=True)
DT_search.fit(x_train, y_train)

print('\n-----------------------Decision Tree Classifier---------------------')

print('Accuracy in training set: {:.2f}' .format(accuracy_score(y_train, DT.predict(x_train))))
print('Accuracy in validation set: {:.2f}' .format(accuracy_score(Y_val, DT.predict(X_val))))
print('Accuracy in training set(hyperparameter tunning): {:.2f}' .format(accuracy_score(y_train, DT_search.predict(x_train))))
print('Accuracy in validation set(hyperparameter tunning): {:.2f}' .format(accuracy_score(Y_val, DT_search.predict(X_val))))
```

    Fitting 5 folds for each of 12 candidates, totalling 60 fits


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done  60 out of  60 | elapsed:   13.0s finished



    -----------------------Decision Tree Classifier---------------------
    Accuracy in training set: 1.00
    Accuracy in validation set: 1.00
    Accuracy in training set(hyperparameter tunning): 0.99
    Accuracy in validation set(hyperparameter tunning): 0.99


#### Random Forest


```python
#instantiate random forest class
RF = RandomForestClassifier()
RF.fit(x_train, y_train)

#parameter settings
params2 = {"max_depth": [3, 32, None],'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8],
           'bootstrap': [False], 'n_estimators': [3, 10, 15], 'max_features': [2, 3, 4],
           "criterion": ["gini","entropy"]}


RF_search = RandomizedSearchCV(RF, param_distributions=params2, random_state=0,
                               n_iter = 200, cv = 5, verbose = 1, n_jobs = 1, return_train_score=True)

RF_search.fit(x_train, y_train)


print('\n-----------------------Random Forest Classifier---------------------')

print('Accuracy in training set: {:.2f}' .format(accuracy_score(y_train, RF.predict(x_train))))
print('Accuracy in validation set: {:.2f}' .format(accuracy_score(Y_val, RF.predict(X_val))))
print('Accuracy in training set(hyperparameter tunning): {:.2f}' .format(accuracy_score(y_train, RF_search.predict(x_train))))
print('Accuracy in validation set(hyperparameter tunning): {:.2f}' .format(accuracy_score(Y_val, RF_search.predict(X_val))))

accuracy_score(Y_val, RF_search.predict(X_val))
```

    Fitting 5 folds for each of 54 candidates, totalling 270 fits


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done 270 out of 270 | elapsed: 10.6min finished



    -----------------------Random Forest Classifier---------------------
    Accuracy in training set: 1.00
    Accuracy in validation set: 1.00
    Accuracy in training set(hyperparameter tunning): 1.00
    Accuracy in validation set(hyperparameter tunning): 1.00





    1.0



#### K-Nearest Neighbors

For this model, we don't need parameter tunning, the performance of the model is accepted.

**Conclusion from hyperparameter tunning:**  We see that Random Forest is still overfitting.  More sophisticated approach will be required for parameter tunning.  Therefore, Decision Tree is performing the better than KNN model.  Consequently we shall deploy it for further predictions


```python

```

After parameter tunning and feauture importance results, Decision Tree model perfomance has improved remarkably.  Therefore the model will be used for predictions.

# 5. Model evaluation

We shall now evaluate machine learning models in the `validation set` which consitute 10% of the training data.  We shall now show a confusion matrix showing the frequency of misclassifications by our
classifier.  


```python
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
```


```python

```

### Decision Tree


```python
#predictions
pred1 = DT_search.predict(X_val)

#target_names = order

mat1 = confusion_matrix(Y_val.values.argmax(axis=1), pred1.argmax(axis=1))

#Normalise
cmat1 = mat1.astype('float')/mat1.sum(axis=1)[:, np.newaxis]

plt.figure(figsize = (10,8))
sns.set(font_scale=0.9)
sns.heatmap(cmat1, cbar = True, square=True, annot=True,yticklabels = labels,
            xticklabels = labels ,annot_kws={'size': 9}, cmap='RdPu')

plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.tight_layout()
```

![png](https://drive.google.com/uc?export=view&id=1DciQynHCbiiWyPWc5923sLMKwDP7guTG)


```python
summ_conf1 = classification_report(Y_val.values.argmax(axis=1), pred1.argmax(axis=1), target_names = labels)


print('\n----------------------Classification Report-----------------------')
print('\n', summ_conf1)
print('--'*33)
```


    ----------------------Classification Report-----------------------

                           precision    recall  f1-score   support

        Cluster_ASCC_123       0.97      1.00      0.98       337
       Cluster_Alessi_13       1.00      0.99      1.00       506
        Cluster_Alessi_3       0.99      1.00      0.99       229
        Cluster_Alessi_9       1.00      0.98      0.99       362
       Cluster_Chereul_1       1.00      0.99      1.00       556
      Cluster_Platais_10       0.98      0.97      0.97       379
       Cluster_Platais_3       1.00      1.00      1.00       328
       Cluster_Platais_8       0.99      0.99      0.99       949
       Cluster_Platais_9       1.00      0.99      0.99       382
    Cluster_Ruprecht_147       1.00      1.00      1.00       427

               micro avg       0.99      0.99      0.99      4455
               macro avg       0.99      0.99      0.99      4455
            weighted avg       0.99      0.99      0.99      4455

    ------------------------------------------------------------------


### Random Forest


```python
#predictions
pred2 = RF_search.predict(X_val)

#target_names = order

mat2 = confusion_matrix(Y_val.values.argmax(axis=1), pred2.argmax(axis=1))

#Normalise
cmat2 = mat2.astype('float')/mat2.sum(axis=1)[:, np.newaxis]

plt.figure(figsize = (10,8))
sns.set(font_scale=0.9)
sns.heatmap(cmat2, cbar = True, square=True, annot=True,yticklabels = labels,
            xticklabels = labels, annot_kws={'size': 9}, cmap='RdPu')

plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.tight_layout()
```


![png](https://drive.google.com/uc?export=view&id=1WMbexrurN_hoLI0mgBwZeWz8wOh30Lbg)

```python
summ_conf2 = classification_report(Y_val.values.argmax(axis=1), pred2.argmax(axis=1), target_names = labels)


print('\n----------------------Classification Report-----------------------')
print('\n', summ_conf2)
print('--'*33)
```


    ----------------------Classification Report-----------------------

                           precision    recall  f1-score   support

        Cluster_ASCC_123       1.00      1.00      1.00       337
       Cluster_Alessi_13       1.00      1.00      1.00       506
        Cluster_Alessi_3       1.00      1.00      1.00       229
        Cluster_Alessi_9       1.00      1.00      1.00       362
       Cluster_Chereul_1       1.00      1.00      1.00       556
      Cluster_Platais_10       1.00      1.00      1.00       379
       Cluster_Platais_3       1.00      1.00      1.00       328
       Cluster_Platais_8       1.00      1.00      1.00       949
       Cluster_Platais_9       1.00      1.00      1.00       382
    Cluster_Ruprecht_147       1.00      1.00      1.00       427

               micro avg       1.00      1.00      1.00      4455
               macro avg       1.00      1.00      1.00      4455
            weighted avg       1.00      1.00      1.00      4455

    ------------------------------------------------------------------


### K-Nearest neighbor


```python
#predictions
pred3 = KNN.predict(X_val)

#target_names = order

mat3 = confusion_matrix(Y_val.values.argmax(axis=1), pred3.argmax(axis=1))

#Normalise
cmat3 = mat3.astype('float')/mat3.sum(axis=1)[:, np.newaxis]

plt.figure(figsize = (10,8))
sns.set(font_scale=0.9)
sns.heatmap(cmat3, cbar = True, square=True, annot=True,yticklabels = labels,
            xticklabels = labels, annot_kws={'size': 9}, cmap='RdPu')

plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.tight_layout()
```
![png](https://drive.google.com/uc?export=view&id=1drf-yTXVDnLF93xQbjQqRk8Qlra7NC3j)

```python
summ_conf3 = classification_report(Y_val.values.argmax(axis=1), pred3.argmax(axis=1),target_names = labels)


print('\n----------------------Classification Report-----------------------')
print('\n', summ_conf3)
print('--'*33)
```


    ----------------------Classification Report-----------------------

                           precision    recall  f1-score   support

        Cluster_ASCC_123       0.64      0.99      0.78       337
       Cluster_Alessi_13       0.99      0.98      0.99       506
        Cluster_Alessi_3       1.00      0.90      0.95       229
        Cluster_Alessi_9       1.00      0.96      0.98       362
       Cluster_Chereul_1       0.99      0.97      0.98       556
      Cluster_Platais_10       1.00      0.92      0.96       379
       Cluster_Platais_3       1.00      0.99      0.99       328
       Cluster_Platais_8       0.90      0.92      0.91       949
       Cluster_Platais_9       0.93      0.65      0.76       382
    Cluster_Ruprecht_147       1.00      0.98      0.99       427

               micro avg       0.93      0.93      0.93      4455
               macro avg       0.94      0.93      0.93      4455
            weighted avg       0.94      0.93      0.93      4455

    ------------------------------------------------------------------



```python

```

# 6. Predictions

We shall now make predictions on the remainin `test set` which consitute 10% of the training data


```python
#predictions in the new dataset
pred_test = DT_search.predict(X_test)

#new data frame with column features label
New_df = pd.DataFrame(data = pred_test, columns = labels)

#reverse onehot encoding to actual values
New_df['Cluster_new'] = (New_df.iloc[:, :] == 1).idxmax(1)

New_df.Cluster_new.unique()
```




    array(['Cluster_Alessi_13', 'Cluster_Ruprecht_147', 'Cluster_Chereul_1',
           'Cluster_ASCC_123', 'Cluster_Platais_8', 'Cluster_Alessi_9',
           'Cluster_Platais_10', 'Cluster_Platais_9', 'Cluster_Platais_3',
           'Cluster_Alessi_3'], dtype=object)




```python
display(X_test.head())
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
      <th>index</th>
      <th>RAdeg</th>
      <th>DEdeg</th>
      <th>e_pmDE</th>
      <th>NobsG</th>
      <th>Gmag</th>
      <th>UCAC4</th>
      <th>RAUdeg</th>
      <th>DEUdeg</th>
      <th>pmRAU</th>
      <th>...</th>
      <th>bp_rp</th>
      <th>radial_velocity</th>
      <th>radial_velocity_error</th>
      <th>rv_nb_transits</th>
      <th>teff_val</th>
      <th>a_g_val</th>
      <th>e_bp_min_rp_val</th>
      <th>radius_val</th>
      <th>lum_val</th>
      <th>angDist</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4725</th>
      <td>9278</td>
      <td>55.03035</td>
      <td>-31.007528</td>
      <td>0.956076</td>
      <td>176</td>
      <td>11.905979</td>
      <td>41282984</td>
      <td>55.030346</td>
      <td>-31.007566</td>
      <td>-2.5</td>
      <td>...</td>
      <td>0.709966</td>
      <td>11.32</td>
      <td>0.52</td>
      <td>17</td>
      <td>6342.25</td>
      <td>0.7730</td>
      <td>0.3895</td>
      <td>1.46</td>
      <td>3.126</td>
      <td>0.151713</td>
    </tr>
    <tr>
      <th>41350</th>
      <td>202233</td>
      <td>287.32214</td>
      <td>-17.587954</td>
      <td>4.745018</td>
      <td>70</td>
      <td>11.876279</td>
      <td>56484256</td>
      <td>287.322140</td>
      <td>-17.587927</td>
      <td>4.3</td>
      <td>...</td>
      <td>0.436925</td>
      <td>-23.91</td>
      <td>9.52</td>
      <td>4</td>
      <td>7628.50</td>
      <td>0.8282</td>
      <td>0.3950</td>
      <td>1.76</td>
      <td>9.433</td>
      <td>0.098402</td>
    </tr>
    <tr>
      <th>17786</th>
      <td>75646</td>
      <td>219.32070</td>
      <td>55.912876</td>
      <td>0.673176</td>
      <td>176</td>
      <td>11.607101</td>
      <td>106922376</td>
      <td>219.320650</td>
      <td>55.912968</td>
      <td>3.4</td>
      <td>...</td>
      <td>0.688489</td>
      <td>-29.02</td>
      <td>3.28</td>
      <td>15</td>
      <td>6375.21</td>
      <td>1.1493</td>
      <td>0.5433</td>
      <td>1.86</td>
      <td>5.132</td>
      <td>0.357957</td>
    </tr>
    <tr>
      <th>14449</th>
      <td>51788</td>
      <td>347.22424</td>
      <td>56.580067</td>
      <td>0.700600</td>
      <td>104</td>
      <td>11.155023</td>
      <td>107276626</td>
      <td>347.224270</td>
      <td>56.580082</td>
      <td>-1.1</td>
      <td>...</td>
      <td>1.711387</td>
      <td>-42.22</td>
      <td>0.33</td>
      <td>9</td>
      <td>4126.97</td>
      <td>0.8795</td>
      <td>0.4170</td>
      <td>20.68</td>
      <td>111.776</td>
      <td>0.063378</td>
    </tr>
    <tr>
      <th>3409</th>
      <td>7099</td>
      <td>46.87359</td>
      <td>-32.264603</td>
      <td>0.861915</td>
      <td>111</td>
      <td>11.075043</td>
      <td>39839956</td>
      <td>46.873478</td>
      <td>-32.264668</td>
      <td>17.2</td>
      <td>...</td>
      <td>0.812772</td>
      <td>17.08</td>
      <td>8.65</td>
      <td>13</td>
      <td>5794.50</td>
      <td>0.4540</td>
      <td>0.2273</td>
      <td>1.48</td>
      <td>2.214</td>
      <td>0.401812</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 58 columns</p>
</div>


Reset index:


```python
X_test = X_test.reset_index()
```


```python
New_df2 = pd.concat([X_test, New_df['Cluster_new']], axis = 1)

print('-------------------Impact cause predicted in the test dataset--------------')
display(New_df2.head(8))

print('---------------------------------------End--------------------------------')
```

    -------------------Impact cause predicted in the test dataset--------------



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
      <th>level_0</th>
      <th>index</th>
      <th>RAdeg</th>
      <th>DEdeg</th>
      <th>e_pmDE</th>
      <th>NobsG</th>
      <th>Gmag</th>
      <th>UCAC4</th>
      <th>RAUdeg</th>
      <th>DEUdeg</th>
      <th>...</th>
      <th>radial_velocity</th>
      <th>radial_velocity_error</th>
      <th>rv_nb_transits</th>
      <th>teff_val</th>
      <th>a_g_val</th>
      <th>e_bp_min_rp_val</th>
      <th>radius_val</th>
      <th>lum_val</th>
      <th>angDist</th>
      <th>Cluster_new</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4725</td>
      <td>9278</td>
      <td>55.03035</td>
      <td>-31.007528</td>
      <td>0.956076</td>
      <td>176</td>
      <td>11.905979</td>
      <td>41282984</td>
      <td>55.030346</td>
      <td>-31.007566</td>
      <td>...</td>
      <td>11.32</td>
      <td>0.52</td>
      <td>17</td>
      <td>6342.25</td>
      <td>0.7730</td>
      <td>0.3895</td>
      <td>1.46</td>
      <td>3.126</td>
      <td>0.151713</td>
      <td>Cluster_Alessi_13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>41350</td>
      <td>202233</td>
      <td>287.32214</td>
      <td>-17.587954</td>
      <td>4.745018</td>
      <td>70</td>
      <td>11.876279</td>
      <td>56484256</td>
      <td>287.322140</td>
      <td>-17.587927</td>
      <td>...</td>
      <td>-23.91</td>
      <td>9.52</td>
      <td>4</td>
      <td>7628.50</td>
      <td>0.8282</td>
      <td>0.3950</td>
      <td>1.76</td>
      <td>9.433</td>
      <td>0.098402</td>
      <td>Cluster_Ruprecht_147</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17786</td>
      <td>75646</td>
      <td>219.32070</td>
      <td>55.912876</td>
      <td>0.673176</td>
      <td>176</td>
      <td>11.607101</td>
      <td>106922376</td>
      <td>219.320650</td>
      <td>55.912968</td>
      <td>...</td>
      <td>-29.02</td>
      <td>3.28</td>
      <td>15</td>
      <td>6375.21</td>
      <td>1.1493</td>
      <td>0.5433</td>
      <td>1.86</td>
      <td>5.132</td>
      <td>0.357957</td>
      <td>Cluster_Chereul_1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14449</td>
      <td>51788</td>
      <td>347.22424</td>
      <td>56.580067</td>
      <td>0.700600</td>
      <td>104</td>
      <td>11.155023</td>
      <td>107276626</td>
      <td>347.224270</td>
      <td>56.580082</td>
      <td>...</td>
      <td>-42.22</td>
      <td>0.33</td>
      <td>9</td>
      <td>4126.97</td>
      <td>0.8795</td>
      <td>0.4170</td>
      <td>20.68</td>
      <td>111.776</td>
      <td>0.063378</td>
      <td>Cluster_ASCC_123</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3409</td>
      <td>7099</td>
      <td>46.87359</td>
      <td>-32.264603</td>
      <td>0.861915</td>
      <td>111</td>
      <td>11.075043</td>
      <td>39839956</td>
      <td>46.873478</td>
      <td>-32.264668</td>
      <td>...</td>
      <td>17.08</td>
      <td>8.65</td>
      <td>13</td>
      <td>5794.50</td>
      <td>0.4540</td>
      <td>0.2273</td>
      <td>1.48</td>
      <td>2.214</td>
      <td>0.401812</td>
      <td>Cluster_Alessi_13</td>
    </tr>
    <tr>
      <th>5</th>
      <td>32125</td>
      <td>172505</td>
      <td>128.71957</td>
      <td>-61.842804</td>
      <td>0.840512</td>
      <td>134</td>
      <td>10.520447</td>
      <td>9617104</td>
      <td>128.719640</td>
      <td>-61.842842</td>
      <td>...</td>
      <td>81.70</td>
      <td>0.34</td>
      <td>8</td>
      <td>4084.32</td>
      <td>0.6995</td>
      <td>0.2933</td>
      <td>21.55</td>
      <td>116.427</td>
      <td>0.124436</td>
      <td>Cluster_Platais_8</td>
    </tr>
    <tr>
      <th>6</th>
      <td>31189</td>
      <td>170417</td>
      <td>135.73608</td>
      <td>-61.562088</td>
      <td>0.755777</td>
      <td>153</td>
      <td>11.078180</td>
      <td>9908785</td>
      <td>135.736220</td>
      <td>-61.562096</td>
      <td>...</td>
      <td>7.15</td>
      <td>1.17</td>
      <td>6</td>
      <td>6194.47</td>
      <td>0.1845</td>
      <td>0.0957</td>
      <td>1.28</td>
      <td>2.162</td>
      <td>0.205106</td>
      <td>Cluster_Platais_8</td>
    </tr>
    <tr>
      <th>7</th>
      <td>6598</td>
      <td>15762</td>
      <td>113.55030</td>
      <td>-44.734245</td>
      <td>1.062795</td>
      <td>71</td>
      <td>10.895708</td>
      <td>27404779</td>
      <td>113.550285</td>
      <td>-44.734257</td>
      <td>...</td>
      <td>22.10</td>
      <td>0.32</td>
      <td>5</td>
      <td>4640.43</td>
      <td>0.2993</td>
      <td>0.1487</td>
      <td>9.74</td>
      <td>39.638</td>
      <td>0.065104</td>
      <td>Cluster_Alessi_9</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 60 columns</p>
</div>


    ---------------------------------------End--------------------------------



```python

```


```python

```
