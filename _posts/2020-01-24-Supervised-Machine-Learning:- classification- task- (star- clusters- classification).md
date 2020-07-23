---
layout: post
author: "Unarine Tshiwawa"
---
## Identifying the problem:

In this task, I seek to demonstrate how supervised learning machine leaning can be used in a labelled dataset.

We have a large dataset containing clusters of stars that are labelled.  The data set will be divided into three sets, training, validation and test set.  In this exercise, we will use a model trained on the training set containing different members of star clusters to classify different kinds of star clusters in the test set.

#### Main point: we are trying to build a model that will predict labels for a new data.

Clearly, we see that this is a machine leaning - classification - type of a problem.  An output to this problem is a categorical quantity.

Objective is to demonstrate:

    - Multi-classification
    - Preprocessing
    - Hyperparameter searching
    - Evaluations
    - Predictions




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


```python
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
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


- NB: There are 236620 instances in the dataset, which is sufficient by Machine Learning standards and perfect to get started.

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
      <th>e_pmRA</th>
      <th>pmDE</th>
      <th>e_pmDE</th>
      <th>NobsG</th>
      <th>Gmag</th>
      <th>2MASSID</th>
      <th>RA2deg</th>
      <th>DE2deg</th>
      <th>UCAC4</th>
      <th>RAUdeg</th>
      <th>DEUdeg</th>
      <th>pmRAU</th>
      <th>e_pmRAU</th>
      <th>pmDEU</th>
      <th>e_pmDEU</th>
      <th>Jmag</th>
      <th>Hmag</th>
      <th>Kmag</th>
      <th>Bmag</th>
      <th>Vmag</th>
      <th>gmag</th>
      <th>rmag</th>
      <th>imag</th>
      <th>RADEcor</th>
      <th>RAplxcor</th>
      <th>RApmRAcor</th>
      <th>RApmDEcor</th>
      <th>DEplxcor</th>
      <th>DEpmRAcor</th>
      <th>DEpmDEcor</th>
      <th>plxpmRAcor</th>
      <th>plxpmDEcor</th>
      <th>pmRApmDEcor</th>
      <th>Cluster</th>
      <th>PMused</th>
      <th>Pfinal</th>
      <th>distL1350</th>
      <th>disterrL1350</th>
      <th>MG</th>
      <th>ra_epoch2000</th>
      <th>dec_epoch2000</th>
      <th>errHalfMaj</th>
      <th>errHalfMin</th>
      <th>errPosAng</th>
      <th>source_id</th>
      <th>ra</th>
      <th>ra_error</th>
      <th>dec</th>
      <th>dec_error</th>
      <th>parallax</th>
      <th>parallax_error</th>
      <th>pmra_x</th>
      <th>pmra_error</th>
      <th>pmdec</th>
      <th>pmdec_error</th>
      <th>duplicated_source</th>
      <th>phot_g_mean_flux</th>
      <th>phot_g_mean_flux_error</th>
      <th>phot_g_mean_mag</th>
      <th>phot_bp_mean_flux</th>
      <th>phot_bp_mean_flux_error</th>
      <th>phot_bp_mean_mag</th>
      <th>phot_rp_mean_flux</th>
      <th>phot_rp_mean_flux_error</th>
      <th>phot_rp_mean_mag</th>
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
      <td>0.814793</td>
      <td>1.014459</td>
      <td>0.587125</td>
      <td>219</td>
      <td>7.624412</td>
      <td>255101265</td>
      <td>300.18040</td>
      <td>-11.480514</td>
      <td>61682627</td>
      <td>300.18036</td>
      <td>-11.480573</td>
      <td>9.4</td>
      <td>1.1</td>
      <td>0.1</td>
      <td>1.9</td>
      <td>5.313</td>
      <td>4.645</td>
      <td>4.242</td>
      <td>10.162</td>
      <td>8.752</td>
      <td>9.820</td>
      <td>8.523</td>
      <td>7.092</td>
      <td>0.391851</td>
      <td>0.057094</td>
      <td>-0.257313</td>
      <td>-0.033846</td>
      <td>0.157448</td>
      <td>-0.171257</td>
      <td>-0.351710</td>
      <td>-0.332498</td>
      <td>-0.448010</td>
      <td>0.615296</td>
      <td>Alessi_10</td>
      <td>T</td>
      <td>0.0</td>
      <td>674.503920</td>
      <td>(521.1853981018066, 2329.7882080078125)</td>
      <td>-1.520510</td>
      <td>300.180365</td>
      <td>-11.480577</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>90.0</td>
      <td>4189191464210827904</td>
      <td>300.180399</td>
      <td>0.0405</td>
      <td>-11.480574</td>
      <td>0.0229</td>
      <td>2.2562</td>
      <td>0.0459</td>
      <td>7.714</td>
      <td>0.081</td>
      <td>0.637</td>
      <td>0.046</td>
      <td>False</td>
      <td>14905400.0</td>
      <td>5628.460</td>
      <td>7.755004</td>
      <td>4100140.0</td>
      <td>4410.180</td>
      <td>8.819390</td>
      <td>15903200.0</td>
      <td>15970.6000</td>
      <td>6.758207</td>
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
      <td>1.484827</td>
      <td>2.417860</td>
      <td>0.866389</td>
      <td>231</td>
      <td>12.635215</td>
      <td>254585654</td>
      <td>300.19095</td>
      <td>-11.627668</td>
      <td>61581376</td>
      <td>300.19095</td>
      <td>-11.627684</td>
      <td>0.2</td>
      <td>1.2</td>
      <td>2.0</td>
      <td>2.4</td>
      <td>11.821</td>
      <td>11.624</td>
      <td>11.559</td>
      <td>13.321</td>
      <td>12.840</td>
      <td>13.031</td>
      <td>12.722</td>
      <td>12.507</td>
      <td>0.615456</td>
      <td>0.377359</td>
      <td>-0.517338</td>
      <td>-0.433912</td>
      <td>0.427563</td>
      <td>-0.483119</td>
      <td>-0.533560</td>
      <td>-0.730528</td>
      <td>-0.748843</td>
      <td>0.928598</td>
      <td>Alessi_10</td>
      <td>T</td>
      <td>0.0</td>
      <td>786.294009</td>
      <td>(605.2994728088379, 2593.4600830078125)</td>
      <td>3.157291</td>
      <td>300.190943</td>
      <td>-11.627669</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>90.0</td>
      <td>4189184794121166080</td>
      <td>300.190949</td>
      <td>0.0359</td>
      <td>-11.627663</td>
      <td>0.0203</td>
      <td>1.3792</td>
      <td>0.0399</td>
      <td>1.359</td>
      <td>0.074</td>
      <td>1.534</td>
      <td>0.042</td>
      <td>False</td>
      <td>155344.0</td>
      <td>29.751</td>
      <td>12.710131</td>
      <td>88844.9</td>
      <td>109.992</td>
      <td>12.979807</td>
      <td>97993.6</td>
      <td>71.8672</td>
      <td>12.283925</td>
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
      <td>2.364083</td>
      <td>-3.461736</td>
      <td>1.402877</td>
      <td>98</td>
      <td>10.881881</td>
      <td>254329749</td>
      <td>300.08572</td>
      <td>-11.700753</td>
      <td>61533071</td>
      <td>300.08572</td>
      <td>-11.700747</td>
      <td>4.3</td>
      <td>1.1</td>
      <td>-4.0</td>
      <td>1.1</td>
      <td>9.312</td>
      <td>8.731</td>
      <td>8.642</td>
      <td>12.409</td>
      <td>11.318</td>
      <td>11.808</td>
      <td>10.920</td>
      <td>10.504</td>
      <td>0.647593</td>
      <td>0.635636</td>
      <td>-0.439750</td>
      <td>-0.419539</td>
      <td>0.486089</td>
      <td>-0.441343</td>
      <td>-0.488741</td>
      <td>-0.683287</td>
      <td>-0.681095</td>
      <td>0.939302</td>
      <td>Alessi_10</td>
      <td>T</td>
      <td>0.0</td>
      <td>1858.621897</td>
      <td>(1024.9900817871094, 7674.6368408203125)</td>
      <td>-0.464074</td>
      <td>300.085707</td>
      <td>-11.700747</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>90.0</td>
      <td>4189178270071217664</td>
      <td>300.085720</td>
      <td>0.0609</td>
      <td>-11.700761</td>
      <td>0.0288</td>
      <td>1.1215</td>
      <td>0.0786</td>
      <td>3.026</td>
      <td>0.107</td>
      <td>-3.222</td>
      <td>0.068</td>
      <td>False</td>
      <td>775893.0</td>
      <td>317.321</td>
      <td>10.963861</td>
      <td>325363.0</td>
      <td>343.958</td>
      <td>11.570466</td>
      <td>637267.0</td>
      <td>463.0920</td>
      <td>10.251117</td>
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
      <td>1.333245</td>
      <td>-12.631682</td>
      <td>0.729939</td>
      <td>189</td>
      <td>11.155972</td>
      <td>255966389</td>
      <td>299.70737</td>
      <td>-11.233572</td>
      <td>61844543</td>
      <td>299.70737</td>
      <td>-11.233577</td>
      <td>-4.7</td>
      <td>1.7</td>
      <td>-14.3</td>
      <td>1.2</td>
      <td>9.561</td>
      <td>9.029</td>
      <td>8.900</td>
      <td>12.642</td>
      <td>11.599</td>
      <td>12.061</td>
      <td>11.217</td>
      <td>10.785</td>
      <td>0.614944</td>
      <td>0.548874</td>
      <td>-0.611939</td>
      <td>-0.573547</td>
      <td>0.506723</td>
      <td>-0.531771</td>
      <td>-0.534322</td>
      <td>-0.840890</td>
      <td>-0.823450</td>
      <td>0.908443</td>
      <td>Alessi_10</td>
      <td>T</td>
      <td>0.0</td>
      <td>1004.409721</td>
      <td>(753.5934448242188, 4379.768371582031)</td>
      <td>1.146417</td>
      <td>299.707359</td>
      <td>-11.233559</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>90.0</td>
      <td>4189447268161556992</td>
      <td>299.707334</td>
      <td>0.0394</td>
      <td>-11.233612</td>
      <td>0.0219</td>
      <td>0.9651</td>
      <td>0.0434</td>
      <td>-5.729</td>
      <td>0.079</td>
      <td>-12.520</td>
      <td>0.045</td>
      <td>False</td>
      <td>608556.0</td>
      <td>329.498</td>
      <td>11.227615</td>
      <td>254909.0</td>
      <td>238.542</td>
      <td>11.835425</td>
      <td>500737.0</td>
      <td>287.0010</td>
      <td>10.512896</td>
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
      <td>1.296801</td>
      <td>-10.222595</td>
      <td>0.867933</td>
      <td>71</td>
      <td>9.283994</td>
      <td>254235092</td>
      <td>299.78880</td>
      <td>-11.727847</td>
      <td>61515591</td>
      <td>299.78876</td>
      <td>-11.727827</td>
      <td>47.2</td>
      <td>1.4</td>
      <td>-8.9</td>
      <td>2.0</td>
      <td>8.314</td>
      <td>8.084</td>
      <td>8.006</td>
      <td>10.075</td>
      <td>9.544</td>
      <td>9.949</td>
      <td>9.334</td>
      <td>9.167</td>
      <td>0.779595</td>
      <td>0.459145</td>
      <td>-0.261615</td>
      <td>-0.304264</td>
      <td>0.343270</td>
      <td>-0.367850</td>
      <td>-0.504861</td>
      <td>-0.625759</td>
      <td>-0.550757</td>
      <td>0.918251</td>
      <td>Alessi_10</td>
      <td>T</td>
      <td>0.0</td>
      <td>160.909330</td>
      <td>(151.23367309570312, 173.15134406089783)</td>
      <td>3.251088</td>
      <td>299.788751</td>
      <td>-11.727828</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>90.0</td>
      <td>4189344330685206912</td>
      <td>299.788962</td>
      <td>0.0352</td>
      <td>-11.727873</td>
      <td>0.0213</td>
      <td>6.0351</td>
      <td>0.0442</td>
      <td>47.836</td>
      <td>0.072</td>
      <td>-10.414</td>
      <td>0.041</td>
      <td>False</td>
      <td>3479110.0</td>
      <td>709.462</td>
      <td>9.334694</td>
      <td>1867080.0</td>
      <td>2058.930</td>
      <td>9.673481</td>
      <td>2284740.0</td>
      <td>2336.3100</td>
      <td>8.864830</td>
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
</div>


Total number of clusters in the sample:


```python
df1['Cluster'].nunique()
```




    128



So we can find out what categories exist and how many clusters belong to each category by using the value_counts() method:


```python
display(df1['Cluster'].value_counts())
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
    ASCC_112          1710
    Collinder_350     1560
    ASCC_51           1463
    ASCC_19           1436
    NGC_6405          1378
    NGC_6281          1354
    NGC_2423          1338
    NGC_6025          1311
    NGC_1901          1223
    NGC_0752          1212
    NGC_7243          1169
    NGC_6991          1168
    NGC_3330          1156
    ASCC_32           1145
    ASCC_124          1143
    Trumpler_2        1123
    NGC_1039          1118
    NGC_6087          1104
    IC_4756           1093
    ASCC_16           1081
    NGC_5662          1080
    NGC_2287          1037
    ASCC_21            959
    NGC_6152           884
    NGC_6940           884
    NGC_2447           874
    ASCC_18            850
    Alessi_12          812
    Lynga_2            812
    NGC_5316           805
    Stock_7            787
    Alessi_2           764
    NGC_1027           761
    NGC_6134           759
    NGC_5822           749
    NGC_2546           743
    Trumpler_3         724
    NGC_2281           709
    NGC_2323           708
    ASCC_23            692
    NGC_4852           681
    NGC_5460           676
    Alessi_10          641
    NGC_2264           627
    NGC_2168           581
    NGC_1662           573
    IC_4725            567
    Collinder_463      559
    NGC_1977           554
    NGC_6416           551
    NGC_6494           541
    Collinder_394      532
    Ruprecht_145       523
    NGC_6124           515
    NGC_6067           515
    NGC_2482           511
    NGC_2437           511
    NGC_6913           502
    NGC_1528           500
    NGC_2571           482
    NGC_7209           481
    NGC_1545           446
    NGC_2353           445
    NGC_6716           443
    NGC_4103           443
    NGC_1750           439
    NGC_1342           403
    NGC_2669           399
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
    Name: Cluster, dtype: int64


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
      <th>pmDE</th>
      <th>e_pmDE</th>
      <th>NobsG</th>
      <th>Gmag</th>
      <th>2MASSID</th>
      <th>RA2deg</th>
      <th>DE2deg</th>
      <th>UCAC4</th>
      <th>RAUdeg</th>
      <th>DEUdeg</th>
      <th>pmRAU</th>
      <th>e_pmRAU</th>
      <th>pmDEU</th>
      <th>e_pmDEU</th>
      <th>Jmag</th>
      <th>Hmag</th>
      <th>Kmag</th>
      <th>Bmag</th>
      <th>Vmag</th>
      <th>gmag</th>
      <th>rmag</th>
      <th>imag</th>
      <th>RADEcor</th>
      <th>RAplxcor</th>
      <th>RApmRAcor</th>
      <th>RApmDEcor</th>
      <th>DEplxcor</th>
      <th>DEpmRAcor</th>
      <th>DEpmDEcor</th>
      <th>plxpmRAcor</th>
      <th>plxpmDEcor</th>
      <th>pmRApmDEcor</th>
      <th>Pfinal</th>
      <th>distL1350</th>
      <th>MG</th>
      <th>ra_epoch2000</th>
      <th>dec_epoch2000</th>
      <th>errHalfMaj</th>
      <th>errHalfMin</th>
      <th>errPosAng</th>
      <th>source_id</th>
      <th>ra</th>
      <th>ra_error</th>
      <th>dec</th>
      <th>dec_error</th>
      <th>parallax</th>
      <th>parallax_error</th>
      <th>pmra_x</th>
      <th>pmra_error</th>
      <th>pmdec</th>
      <th>pmdec_error</th>
      <th>phot_g_mean_flux</th>
      <th>phot_g_mean_flux_error</th>
      <th>phot_g_mean_mag</th>
      <th>phot_bp_mean_flux</th>
      <th>phot_bp_mean_flux_error</th>
      <th>phot_bp_mean_mag</th>
      <th>phot_rp_mean_flux</th>
      <th>phot_rp_mean_flux_error</th>
      <th>phot_rp_mean_mag</th>
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
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>2.366200e+05</td>
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>2.366200e+05</td>
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>236620.000000</td>
      <td>236620.000000</td>
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
      <td>236541.000000</td>
      <td>236541.000000</td>
      <td>236541.000000</td>
      <td>236541.000000</td>
      <td>236541.000000</td>
      <td>236541.000000</td>
      <td>2.366200e+05</td>
      <td>2.366200e+05</td>
      <td>236620.000000</td>
      <td>2.364730e+05</td>
      <td>2.364730e+05</td>
      <td>236473.000000</td>
      <td>2.364730e+05</td>
      <td>2.364730e+05</td>
      <td>236473.000000</td>
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
      <td>-1.557729</td>
      <td>1.161047</td>
      <td>128.685179</td>
      <td>10.912579</td>
      <td>2.143364e+08</td>
      <td>180.825895</td>
      <td>-14.879133</td>
      <td>5.149631e+07</td>
      <td>180.825895</td>
      <td>-14.879135</td>
      <td>-2.523112</td>
      <td>1.574246</td>
      <td>-1.582009</td>
      <td>1.530871</td>
      <td>9.823844</td>
      <td>9.510496</td>
      <td>9.414603</td>
      <td>11.908318</td>
      <td>11.211551</td>
      <td>10.118346</td>
      <td>9.629003</td>
      <td>9.587449</td>
      <td>-0.044482</td>
      <td>-0.132229</td>
      <td>-0.658548</td>
      <td>0.018754</td>
      <td>0.193237</td>
      <td>0.087089</td>
      <td>-0.480069</td>
      <td>0.198925</td>
      <td>-0.027088</td>
      <td>-0.007671</td>
      <td>0.012961</td>
      <td>1108.863301</td>
      <td>1.364211</td>
      <td>180.825894</td>
      <td>-14.879135</td>
      <td>0.001166</td>
      <td>0.001056</td>
      <td>48.979672</td>
      <td>4.057618e+18</td>
      <td>180.825869</td>
      <td>0.031712</td>
      <td>-14.879141</td>
      <td>0.032290</td>
      <td>1.774387</td>
      <td>0.039976</td>
      <td>-3.065797</td>
      <td>0.064469</td>
      <td>-1.418307</td>
      <td>0.063055</td>
      <td>1.615962e+06</td>
      <td>9.628334e+02</td>
      <td>10.980493</td>
      <td>7.915458e+05</td>
      <td>1.406762e+03</td>
      <td>11.379729</td>
      <td>1.191315e+06</td>
      <td>2.383612e+03</td>
      <td>10.480083</td>
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
      <td>11.184024</td>
      <td>0.927231</td>
      <td>58.894432</td>
      <td>1.098175</td>
      <td>1.547625e+08</td>
      <td>90.163170</td>
      <td>45.426544</td>
      <td>3.713558e+07</td>
      <td>90.163170</td>
      <td>45.426544</td>
      <td>11.164994</td>
      <td>1.187987</td>
      <td>11.372816</td>
      <td>1.123163</td>
      <td>1.387943</td>
      <td>1.554228</td>
      <td>1.602367</td>
      <td>1.075545</td>
      <td>0.989546</td>
      <td>4.021796</td>
      <td>3.877332</td>
      <td>3.647064</td>
      <td>0.600393</td>
      <td>0.501788</td>
      <td>0.311418</td>
      <td>0.561424</td>
      <td>0.486986</td>
      <td>0.609442</td>
      <td>0.367496</td>
      <td>0.624282</td>
      <td>0.441566</td>
      <td>0.591534</td>
      <td>0.100790</td>
      <td>873.996363</td>
      <td>1.882406</td>
      <td>90.163170</td>
      <td>45.426543</td>
      <td>0.000764</td>
      <td>0.000676</td>
      <td>44.823774</td>
      <td>1.873771e+18</td>
      <td>90.163169</td>
      <td>0.026792</td>
      <td>45.426534</td>
      <td>0.026109</td>
      <td>1.590829</td>
      <td>0.028248</td>
      <td>11.108848</td>
      <td>0.050004</td>
      <td>11.172815</td>
      <td>0.046511</td>
      <td>4.792180e+06</td>
      <td>1.071588e+04</td>
      <td>1.099675</td>
      <td>2.488408e+06</td>
      <td>1.032006e+04</td>
      <td>1.050419</td>
      <td>3.753941e+06</td>
      <td>2.249080e+04</td>
      <td>1.173647</td>
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
      <td>-366.680000</td>
      <td>0.018136</td>
      <td>11.000000</td>
      <td>4.514738</td>
      <td>8.057109e+06</td>
      <td>21.903467</td>
      <td>-71.945210</td>
      <td>2.327580e+06</td>
      <td>21.903540</td>
      <td>-71.945210</td>
      <td>-302.500000</td>
      <td>0.000000</td>
      <td>-357.700000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.999914</td>
      <td>-0.995559</td>
      <td>-0.999528</td>
      <td>-0.999262</td>
      <td>-0.994550</td>
      <td>-0.998560</td>
      <td>-0.999250</td>
      <td>-0.997201</td>
      <td>-0.992578</td>
      <td>-0.999735</td>
      <td>0.000000</td>
      <td>19.375770</td>
      <td>-7.767817</td>
      <td>21.903560</td>
      <td>-71.945217</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.248838e+17</td>
      <td>21.903667</td>
      <td>0.009200</td>
      <td>-71.945207</td>
      <td>0.009700</td>
      <td>-8.661600</td>
      <td>0.012100</td>
      <td>-312.164000</td>
      <td>0.016000</td>
      <td>-365.162000</td>
      <td>0.019000</td>
      <td>6.138380e+03</td>
      <td>4.774140e+00</td>
      <td>3.964791</td>
      <td>4.698610e+03</td>
      <td>7.669640e+00</td>
      <td>4.055446</td>
      <td>6.510540e+03</td>
      <td>1.093490e+01</td>
      <td>3.701656</td>
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
      <td>-5.673700</td>
      <td>0.657286</td>
      <td>85.000000</td>
      <td>10.310032</td>
      <td>6.878748e+07</td>
      <td>114.362135</td>
      <td>-55.415619</td>
      <td>1.663808e+07</td>
      <td>114.362129</td>
      <td>-55.415614</td>
      <td>-7.000000</td>
      <td>1.000000</td>
      <td>-5.900000</td>
      <td>1.000000</td>
      <td>9.015000</td>
      <td>8.574000</td>
      <td>8.457000</td>
      <td>11.335000</td>
      <td>10.643000</td>
      <td>10.691000</td>
      <td>10.052000</td>
      <td>9.733000</td>
      <td>-0.579162</td>
      <td>-0.570199</td>
      <td>-0.867317</td>
      <td>-0.464296</td>
      <td>-0.252428</td>
      <td>-0.511960</td>
      <td>-0.775956</td>
      <td>-0.351453</td>
      <td>-0.361159</td>
      <td>-0.504510</td>
      <td>0.000000</td>
      <td>452.083326</td>
      <td>0.025793</td>
      <td>114.362133</td>
      <td>-55.415615</td>
      <td>0.001000</td>
      <td>0.001000</td>
      <td>0.000000</td>
      <td>2.026967e+18</td>
      <td>114.362129</td>
      <td>0.020900</td>
      <td>-55.415601</td>
      <td>0.023100</td>
      <td>0.797600</td>
      <td>0.028200</td>
      <td>-7.519000</td>
      <td>0.043000</td>
      <td>-5.482000</td>
      <td>0.045000</td>
      <td>3.667828e+05</td>
      <td>1.005050e+02</td>
      <td>10.382900</td>
      <td>1.949010e+05</td>
      <td>2.304200e+02</td>
      <td>10.843428</td>
      <td>2.320810e+05</td>
      <td>1.866610e+02</td>
      <td>9.813921</td>
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
      <td>-1.096704</td>
      <td>0.875991</td>
      <td>120.000000</td>
      <td>11.122927</td>
      <td>1.610887e+08</td>
      <td>156.674035</td>
      <td>-34.119026</td>
      <td>3.778163e+07</td>
      <td>156.674075</td>
      <td>-34.119026</td>
      <td>-2.500000</td>
      <td>1.300000</td>
      <td>-1.200000</td>
      <td>1.300000</td>
      <td>10.079000</td>
      <td>9.832000</td>
      <td>9.761000</td>
      <td>12.106000</td>
      <td>11.392000</td>
      <td>11.613000</td>
      <td>11.044000</td>
      <td>10.842000</td>
      <td>-0.183227</td>
      <td>-0.214951</td>
      <td>-0.753577</td>
      <td>0.077191</td>
      <td>0.283337</td>
      <td>0.237595</td>
      <td>-0.555163</td>
      <td>0.330554</td>
      <td>-0.021381</td>
      <td>-0.051683</td>
      <td>0.000000</td>
      <td>800.318496</td>
      <td>1.372408</td>
      <td>156.674082</td>
      <td>-34.119031</td>
      <td>0.001000</td>
      <td>0.001000</td>
      <td>90.000000</td>
      <td>5.236241e+18</td>
      <td>156.673922</td>
      <td>0.025400</td>
      <td>-34.119017</td>
      <td>0.026900</td>
      <td>1.304100</td>
      <td>0.033800</td>
      <td>-3.019000</td>
      <td>0.053000</td>
      <td>-1.097000</td>
      <td>0.053000</td>
      <td>6.342630e+05</td>
      <td>2.948210e+02</td>
      <td>11.182692</td>
      <td>3.173020e+05</td>
      <td>3.885180e+02</td>
      <td>11.597705</td>
      <td>4.298010e+05</td>
      <td>3.571940e+02</td>
      <td>10.678752</td>
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
      <td>3.501610</td>
      <td>1.323967</td>
      <td>160.000000</td>
      <td>11.705323</td>
      <td>3.965321e+08</td>
      <td>266.211783</td>
      <td>36.304384</td>
      <td>9.483573e+07</td>
      <td>266.211783</td>
      <td>36.304369</td>
      <td>1.800000</td>
      <td>1.800000</td>
      <td>3.600000</td>
      <td>1.800000</td>
      <td>10.864000</td>
      <td>10.677000</td>
      <td>10.617000</td>
      <td>12.666000</td>
      <td>11.924000</td>
      <td>12.194000</td>
      <td>11.704000</td>
      <td>11.589000</td>
      <td>0.541638</td>
      <td>0.259550</td>
      <td>-0.559673</td>
      <td>0.490964</td>
      <td>0.633269</td>
      <td>0.646000</td>
      <td>-0.249245</td>
      <td>0.805904</td>
      <td>0.286478</td>
      <td>0.480765</td>
      <td>0.000000</td>
      <td>1538.880925</td>
      <td>2.809844</td>
      <td>266.211794</td>
      <td>36.304366</td>
      <td>0.001000</td>
      <td>0.001000</td>
      <td>90.000000</td>
      <td>5.524290e+18</td>
      <td>266.211780</td>
      <td>0.036000</td>
      <td>36.304336</td>
      <td>0.034600</td>
      <td>2.217300</td>
      <td>0.043600</td>
      <td>1.241000</td>
      <td>0.073000</td>
      <td>3.748000</td>
      <td>0.067000</td>
      <td>1.324910e+06</td>
      <td>5.533380e+02</td>
      <td>11.777344</td>
      <td>6.356000e+05</td>
      <td>7.960760e+02</td>
      <td>12.126852</td>
      <td>9.532340e+05</td>
      <td>9.279870e+02</td>
      <td>11.347821</td>
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
      <td>68.556130</td>
      <td>16.676441</td>
      <td>1122.000000</td>
      <td>15.614060</td>
      <td>4.690414e+08</td>
      <td>358.557830</td>
      <td>77.288360</td>
      <td>1.130418e+08</td>
      <td>358.557900</td>
      <td>77.288350</td>
      <td>345.000000</td>
      <td>45.000000</td>
      <td>105.500000</td>
      <td>45.000000</td>
      <td>14.879000</td>
      <td>14.762000</td>
      <td>14.653000</td>
      <td>18.112000</td>
      <td>17.380000</td>
      <td>18.036000</td>
      <td>17.515000</td>
      <td>17.576000</td>
      <td>0.999972</td>
      <td>0.995650</td>
      <td>0.993994</td>
      <td>0.999066</td>
      <td>0.995840</td>
      <td>0.998695</td>
      <td>0.996747</td>
      <td>0.997618</td>
      <td>0.992615</td>
      <td>0.999592</td>
      <td>1.000000</td>
      <td>6306.649182</td>
      <td>8.881897</td>
      <td>358.557903</td>
      <td>77.288346</td>
      <td>0.033000</td>
      <td>0.031000</td>
      <td>90.000000</td>
      <td>6.881455e+18</td>
      <td>358.557924</td>
      <td>1.609000</td>
      <td>77.288318</td>
      <td>2.122900</td>
      <td>51.429000</td>
      <td>1.287100</td>
      <td>73.661000</td>
      <td>2.117000</td>
      <td>66.045000</td>
      <td>2.069000</td>
      <td>4.891360e+08</td>
      <td>2.050730e+06</td>
      <td>16.218231</td>
      <td>3.298960e+08</td>
      <td>1.474570e+06</td>
      <td>16.171463</td>
      <td>2.655250e+08</td>
      <td>4.506590e+06</td>
      <td>15.227877</td>
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
    PMused                          0
    Pfinal                          0
    distL1350                       0
    disterrL1350                    0
    MG                              0
    ra_epoch2000                    0
    dec_epoch2000                   0
    errHalfMaj                      0
    errHalfMin                      0
    errPosAng                       0
    source_id                       0
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
    dtype: int64


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
      <th>pmDEU</th>
      <th>e_pmDEU</th>
      <th>Jmag</th>
      <th>Hmag</th>
      <th>Kmag</th>
      <th>Bmag</th>
      <th>Vmag</th>
      <th>gmag</th>
      <th>rmag</th>
      <th>imag</th>
      <th>RADEcor</th>
      <th>RAplxcor</th>
      <th>RApmRAcor</th>
      <th>RApmDEcor</th>
      <th>DEplxcor</th>
      <th>DEpmRAcor</th>
      <th>DEpmDEcor</th>
      <th>plxpmRAcor</th>
      <th>plxpmDEcor</th>
      <th>pmRApmDEcor</th>
      <th>Cluster</th>
      <th>distL1350</th>
      <th>MG</th>
      <th>parallax</th>
      <th>parallax_error</th>
      <th>pmra_x</th>
      <th>pmra_error</th>
      <th>pmdec</th>
      <th>pmdec_error</th>
      <th>phot_g_mean_flux</th>
      <th>phot_g_mean_flux_error</th>
      <th>phot_g_mean_mag</th>
      <th>phot_bp_mean_flux</th>
      <th>phot_bp_mean_flux_error</th>
      <th>phot_bp_mean_mag</th>
      <th>phot_rp_mean_flux</th>
      <th>phot_rp_mean_flux_error</th>
      <th>phot_rp_mean_mag</th>
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
      <td>0.1</td>
      <td>1.9</td>
      <td>5.313</td>
      <td>4.645</td>
      <td>4.242</td>
      <td>10.162</td>
      <td>8.752</td>
      <td>9.820</td>
      <td>8.523</td>
      <td>7.092</td>
      <td>0.391851</td>
      <td>0.057094</td>
      <td>-0.257313</td>
      <td>-0.033846</td>
      <td>0.157448</td>
      <td>-0.171257</td>
      <td>-0.351710</td>
      <td>-0.332498</td>
      <td>-0.448010</td>
      <td>0.615296</td>
      <td>Alessi_10</td>
      <td>674.503920</td>
      <td>-1.520510</td>
      <td>2.2562</td>
      <td>0.0459</td>
      <td>7.714</td>
      <td>0.081</td>
      <td>0.637</td>
      <td>0.046</td>
      <td>14905400.0</td>
      <td>5628.460</td>
      <td>7.755004</td>
      <td>4100140.0</td>
      <td>4410.180</td>
      <td>8.819390</td>
      <td>15903200.0</td>
      <td>15970.6000</td>
      <td>6.758207</td>
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
      <td>2.0</td>
      <td>2.4</td>
      <td>11.821</td>
      <td>11.624</td>
      <td>11.559</td>
      <td>13.321</td>
      <td>12.840</td>
      <td>13.031</td>
      <td>12.722</td>
      <td>12.507</td>
      <td>0.615456</td>
      <td>0.377359</td>
      <td>-0.517338</td>
      <td>-0.433912</td>
      <td>0.427563</td>
      <td>-0.483119</td>
      <td>-0.533560</td>
      <td>-0.730528</td>
      <td>-0.748843</td>
      <td>0.928598</td>
      <td>Alessi_10</td>
      <td>786.294009</td>
      <td>3.157291</td>
      <td>1.3792</td>
      <td>0.0399</td>
      <td>1.359</td>
      <td>0.074</td>
      <td>1.534</td>
      <td>0.042</td>
      <td>155344.0</td>
      <td>29.751</td>
      <td>12.710131</td>
      <td>88844.9</td>
      <td>109.992</td>
      <td>12.979807</td>
      <td>97993.6</td>
      <td>71.8672</td>
      <td>12.283925</td>
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
      <td>-4.0</td>
      <td>1.1</td>
      <td>9.312</td>
      <td>8.731</td>
      <td>8.642</td>
      <td>12.409</td>
      <td>11.318</td>
      <td>11.808</td>
      <td>10.920</td>
      <td>10.504</td>
      <td>0.647593</td>
      <td>0.635636</td>
      <td>-0.439750</td>
      <td>-0.419539</td>
      <td>0.486089</td>
      <td>-0.441343</td>
      <td>-0.488741</td>
      <td>-0.683287</td>
      <td>-0.681095</td>
      <td>0.939302</td>
      <td>Alessi_10</td>
      <td>1858.621897</td>
      <td>-0.464074</td>
      <td>1.1215</td>
      <td>0.0786</td>
      <td>3.026</td>
      <td>0.107</td>
      <td>-3.222</td>
      <td>0.068</td>
      <td>775893.0</td>
      <td>317.321</td>
      <td>10.963861</td>
      <td>325363.0</td>
      <td>343.958</td>
      <td>11.570466</td>
      <td>637267.0</td>
      <td>463.0920</td>
      <td>10.251117</td>
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
      <td>-14.3</td>
      <td>1.2</td>
      <td>9.561</td>
      <td>9.029</td>
      <td>8.900</td>
      <td>12.642</td>
      <td>11.599</td>
      <td>12.061</td>
      <td>11.217</td>
      <td>10.785</td>
      <td>0.614944</td>
      <td>0.548874</td>
      <td>-0.611939</td>
      <td>-0.573547</td>
      <td>0.506723</td>
      <td>-0.531771</td>
      <td>-0.534322</td>
      <td>-0.840890</td>
      <td>-0.823450</td>
      <td>0.908443</td>
      <td>Alessi_10</td>
      <td>1004.409721</td>
      <td>1.146417</td>
      <td>0.9651</td>
      <td>0.0434</td>
      <td>-5.729</td>
      <td>0.079</td>
      <td>-12.520</td>
      <td>0.045</td>
      <td>608556.0</td>
      <td>329.498</td>
      <td>11.227615</td>
      <td>254909.0</td>
      <td>238.542</td>
      <td>11.835425</td>
      <td>500737.0</td>
      <td>287.0010</td>
      <td>10.512896</td>
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
      <td>-8.9</td>
      <td>2.0</td>
      <td>8.314</td>
      <td>8.084</td>
      <td>8.006</td>
      <td>10.075</td>
      <td>9.544</td>
      <td>9.949</td>
      <td>9.334</td>
      <td>9.167</td>
      <td>0.779595</td>
      <td>0.459145</td>
      <td>-0.261615</td>
      <td>-0.304264</td>
      <td>0.343270</td>
      <td>-0.367850</td>
      <td>-0.504861</td>
      <td>-0.625759</td>
      <td>-0.550757</td>
      <td>0.918251</td>
      <td>Alessi_10</td>
      <td>160.909330</td>
      <td>3.251088</td>
      <td>6.0351</td>
      <td>0.0442</td>
      <td>47.836</td>
      <td>0.072</td>
      <td>-10.414</td>
      <td>0.041</td>
      <td>3479110.0</td>
      <td>709.462</td>
      <td>9.334694</td>
      <td>1867080.0</td>
      <td>2058.930</td>
      <td>9.673481</td>
      <td>2284740.0</td>
      <td>2336.3100</td>
      <td>8.864830</td>
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


Our sample now has 88990 instances, each consist of 63 features.  This is fairly enough by Machine Learning standards, and it’s perfect to get started.  Therefore we shall proceed with the exploratory data analysis.


```python

```

# 1. Exploratory Data Analysis

A histogram for each numerical attribute:


```python
df3.hist(bins=50, figsize=(30,25))
plt.show()
```


![png](https://drive.google.com/uc?export=view&id=1g9oS0Jlw7NGrrzu67SeRcEoo1RBY8a8M)


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
    Collinder_350     708
    ASCC_19           634
    NGC_0752          596
    Alessi_21         533
    ASCC_16           524
    ASCC_124          524
    NGC_6025          507
    Stock_10          501
    NGC_1901          491
    ASCC_21           473
    NGC_1039          471
    NGC_2287          468
    ASCC_32           458
    NGC_7243          416
    Alessi_12         415
    ASCC_18           406
    NGC_3330          404
    NGC_6087          393
    NGC_2423          376
    NGC_2281          373
    NGC_6940          363
    NGC_6281          355
    Alessi_10         354
    NGC_6991          329
    NGC_5460          326
    ASCC_23           326
    IC_4756           310
    NGC_6405          307
    Trumpler_2        296
    Alessi_2          279
    NGC_5662          275
    NGC_1662          274
    NGC_6152          269
    NGC_1977          236
    NGC_3680          213
    NGC_6134          213
    NGC_5316          210
    NGC_5822          198
    Trumpler_3        198
    NGC_2264          197
    Collinder_463     196
    NGC_2447          194
    Lynga_2           190
    NGC_1027          190
    Stock_7           183
    NGC_7209          182
    NGC_1342          182
    NGC_2546          179
    NGC_6811          177
    NGC_2571          170
    NGC_1750          167
    NGC_2669          166
    NGC_6913          162
    ASCC_10           155
    NGC_2323          150
    NGC_1647          145
    NGC_6124          142
    Ruprecht_145      140
    Ruprecht_1        139
    Collinder_394     137
    Roslund_3         137
    NGC_1528          137
    NGC_2437          137
    NGC_4852          135
    NGC_2548          131
    NGC_2482          130
    NGC_6067          127
    NGC_6494          126
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
    NGC_6604           10
    NGC_6705           10
    Name: Cluster, dtype: int64



## Sample selection for Machine Learning

 #### We are going to only select clusters which contains more than 2000 members


```python
df4 = df3[df3['Cluster'].str.contains('Platais_8|Chereul_1|Alessi_13|Ruprecht_147|Platais_10|Platais_9|Alessi_9|ASCC_123|Platais_3|Alessi_3', case=True,na=False)]

#reset index
df4 = df4.reset_index()
```

Let's view the number of unique clusters by using the value_counts() method:


```python
display(df4.Cluster.value_counts())
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
display(df4.describe())
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
      <th>e_pmRAU</th>
      <th>pmDEU</th>
      <th>e_pmDEU</th>
      <th>Jmag</th>
      <th>Hmag</th>
      <th>Kmag</th>
      <th>Bmag</th>
      <th>Vmag</th>
      <th>gmag</th>
      <th>rmag</th>
      <th>imag</th>
      <th>RADEcor</th>
      <th>RAplxcor</th>
      <th>RApmRAcor</th>
      <th>RApmDEcor</th>
      <th>DEplxcor</th>
      <th>DEpmRAcor</th>
      <th>DEpmDEcor</th>
      <th>plxpmRAcor</th>
      <th>plxpmDEcor</th>
      <th>pmRApmDEcor</th>
      <th>distL1350</th>
      <th>MG</th>
      <th>parallax</th>
      <th>parallax_error</th>
      <th>pmra_x</th>
      <th>pmra_error</th>
      <th>pmdec</th>
      <th>pmdec_error</th>
      <th>phot_g_mean_flux</th>
      <th>phot_g_mean_flux_error</th>
      <th>phot_g_mean_mag</th>
      <th>phot_bp_mean_flux</th>
      <th>phot_bp_mean_flux_error</th>
      <th>phot_bp_mean_mag</th>
      <th>phot_rp_mean_flux</th>
      <th>phot_rp_mean_flux_error</th>
      <th>phot_rp_mean_mag</th>
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
      <td>44545.000000</td>
      <td>44545.000000</td>
      <td>44545.000000</td>
      <td>44545.000000</td>
      <td>44545.00000</td>
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
      <td>44545.000000</td>
      <td>44545.000000</td>
      <td>44545.000000</td>
      <td>44545.000000</td>
      <td>4.454500e+04</td>
      <td>44545.000000</td>
      <td>44545.000000</td>
      <td>4.454500e+04</td>
      <td>44545.000000</td>
      <td>44545.000000</td>
      <td>4.454500e+04</td>
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
      <td>1.484892</td>
      <td>-0.922245</td>
      <td>1.456507</td>
      <td>9.430606</td>
      <td>8.97928</td>
      <td>8.859927</td>
      <td>12.136807</td>
      <td>11.199995</td>
      <td>10.504878</td>
      <td>9.826981</td>
      <td>9.770476</td>
      <td>-0.053217</td>
      <td>-0.193378</td>
      <td>-0.583041</td>
      <td>0.042340</td>
      <td>0.244851</td>
      <td>0.125494</td>
      <td>-0.461610</td>
      <td>0.266835</td>
      <td>-0.143566</td>
      <td>0.009381</td>
      <td>930.527559</td>
      <td>1.721307</td>
      <td>2.217798</td>
      <td>0.034422</td>
      <td>-3.047656</td>
      <td>0.056526</td>
      <td>-0.886165</td>
      <td>0.056940</td>
      <td>1.607211e+06</td>
      <td>658.778042</td>
      <td>10.891191</td>
      <td>6.694851e+05</td>
      <td>901.990757</td>
      <td>11.420217</td>
      <td>1.334491e+06</td>
      <td>1905.082934</td>
      <td>10.262058</td>
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
      <td>1.014288</td>
      <td>14.550332</td>
      <td>0.945268</td>
      <td>1.336028</td>
      <td>1.48027</td>
      <td>1.528181</td>
      <td>0.939907</td>
      <td>0.910862</td>
      <td>3.684201</td>
      <td>3.500798</td>
      <td>3.103797</td>
      <td>0.612992</td>
      <td>0.465623</td>
      <td>0.349306</td>
      <td>0.581759</td>
      <td>0.467499</td>
      <td>0.628559</td>
      <td>0.378820</td>
      <td>0.579489</td>
      <td>0.454302</td>
      <td>0.583407</td>
      <td>788.889310</td>
      <td>2.142280</td>
      <td>1.991047</td>
      <td>0.020973</td>
      <td>14.415794</td>
      <td>0.037539</td>
      <td>14.466671</td>
      <td>0.035432</td>
      <td>4.011777e+06</td>
      <td>4587.269845</td>
      <td>1.044323</td>
      <td>1.491733e+06</td>
      <td>3513.308905</td>
      <td>0.971315</td>
      <td>3.639528e+06</td>
      <td>10185.315271</td>
      <td>1.127272</td>
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
      <td>0.000000</td>
      <td>-73.100000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.999432</td>
      <td>-0.993431</td>
      <td>-0.999366</td>
      <td>-0.998676</td>
      <td>-0.986662</td>
      <td>-0.998560</td>
      <td>-0.999250</td>
      <td>-0.993473</td>
      <td>-0.992349</td>
      <td>-0.993745</td>
      <td>19.375770</td>
      <td>-6.074289</td>
      <td>0.163300</td>
      <td>0.014100</td>
      <td>-66.665000</td>
      <td>0.020000</td>
      <td>-66.502000</td>
      <td>0.025000</td>
      <td>7.724700e+04</td>
      <td>12.712000</td>
      <td>4.745544</td>
      <td>2.995970e+04</td>
      <td>34.279100</td>
      <td>6.054809</td>
      <td>5.338120e+04</td>
      <td>32.553200</td>
      <td>3.737951</td>
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
      <td>1.000000</td>
      <td>-6.900000</td>
      <td>1.000000</td>
      <td>8.610000</td>
      <td>8.03200</td>
      <td>7.874000</td>
      <td>11.633000</td>
      <td>10.674000</td>
      <td>10.972000</td>
      <td>10.135000</td>
      <td>9.754000</td>
      <td>-0.580782</td>
      <td>-0.589083</td>
      <td>-0.845952</td>
      <td>-0.470247</td>
      <td>-0.144491</td>
      <td>-0.499398</td>
      <td>-0.747546</td>
      <td>-0.174879</td>
      <td>-0.501335</td>
      <td>-0.503058</td>
      <td>335.133007</td>
      <td>0.053369</td>
      <td>0.835400</td>
      <td>0.025000</td>
      <td>-9.375000</td>
      <td>0.040000</td>
      <td>-6.728000</td>
      <td>0.043000</td>
      <td>4.169440e+05</td>
      <td>123.663000</td>
      <td>10.341487</td>
      <td>1.977100e+05</td>
      <td>216.858000</td>
      <td>10.935009</td>
      <td>2.961770e+05</td>
      <td>212.458000</td>
      <td>9.631433</td>
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
      <td>1.200000</td>
      <td>-0.400000</td>
      <td>1.200000</td>
      <td>9.563000</td>
      <td>9.08500</td>
      <td>8.966000</td>
      <td>12.309000</td>
      <td>11.368000</td>
      <td>11.748000</td>
      <td>10.969000</td>
      <td>10.666000</td>
      <td>-0.233382</td>
      <td>-0.252802</td>
      <td>-0.685360</td>
      <td>0.153611</td>
      <td>0.315050</td>
      <td>0.316475</td>
      <td>-0.534316</td>
      <td>0.346520</td>
      <td>-0.166705</td>
      <td>-0.081136</td>
      <td>649.747671</td>
      <td>1.397480</td>
      <td>1.536400</td>
      <td>0.029600</td>
      <td>-3.112000</td>
      <td>0.049000</td>
      <td>-0.483000</td>
      <td>0.049000</td>
      <td>7.028910e+05</td>
      <td>294.289000</td>
      <td>11.071146</td>
      <td>3.102840e+05</td>
      <td>339.227000</td>
      <td>11.621991</td>
      <td>5.456010e+05</td>
      <td>405.186000</td>
      <td>10.419732</td>
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
      <td>1.700000</td>
      <td>6.200000</td>
      <td>1.700000</td>
      <td>10.480000</td>
      <td>10.19500</td>
      <td>10.127000</td>
      <td>12.810000</td>
      <td>11.859000</td>
      <td>12.257000</td>
      <td>11.568000</td>
      <td>11.357000</td>
      <td>0.561748</td>
      <td>0.126405</td>
      <td>-0.403074</td>
      <td>0.529229</td>
      <td>0.668170</td>
      <td>0.680774</td>
      <td>-0.234280</td>
      <td>0.834463</td>
      <td>0.172662</td>
      <td>0.492754</td>
      <td>1282.207331</td>
      <td>3.649123</td>
      <td>2.981200</td>
      <td>0.037800</td>
      <td>2.967000</td>
      <td>0.064000</td>
      <td>6.219000</td>
      <td>0.061000</td>
      <td>1.376420e+06</td>
      <td>520.268000</td>
      <td>11.638172</td>
      <td>5.841870e+05</td>
      <td>614.227000</td>
      <td>12.111315</td>
      <td>1.127700e+06</td>
      <td>998.131000</td>
      <td>11.083040</td>
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
      <td>45.000000</td>
      <td>80.900000</td>
      <td>45.000000</td>
      <td>12.370000</td>
      <td>12.27500</td>
      <td>12.182000</td>
      <td>16.767000</td>
      <td>15.639000</td>
      <td>18.036000</td>
      <td>16.659000</td>
      <td>16.383000</td>
      <td>0.999972</td>
      <td>0.987735</td>
      <td>0.986788</td>
      <td>0.998497</td>
      <td>0.988574</td>
      <td>0.998420</td>
      <td>0.994436</td>
      <td>0.997235</td>
      <td>0.985228</td>
      <td>0.999585</td>
      <td>5321.813898</td>
      <td>7.976318</td>
      <td>51.429000</td>
      <td>0.763000</td>
      <td>67.239000</td>
      <td>1.495000</td>
      <td>66.045000</td>
      <td>1.390000</td>
      <td>2.383030e+08</td>
      <td>731873.000000</td>
      <td>13.468661</td>
      <td>5.231560e+07</td>
      <td>324239.000000</td>
      <td>14.160044</td>
      <td>2.567960e+08</td>
      <td>908222.000000</td>
      <td>12.943449</td>
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




    <matplotlib.axes._subplots.AxesSubplot at 0x7fce2847b588>




![png](https://drive.google.com/uc?export=view&id=1D12zU41eM9tCeEVKNMLdpTgH98Yh4x0-)


A histogram for each numerical attribute after sample selection:


```python
df4.hist(bins=50, figsize=(30,25))
plt.show()
```


![png](https://drive.google.com/uc?export=view&id=17GeYmX6HgiJLA0OfJ6VF6a0ybJUILpA-)


# 2. Preprocessing

We are going to generate a feature matrix and target vector.

Note: Our target variable is a string, so we need to perform one-hot encouding technique to the target variable.



#### Feature matrix:


```python
x = df4.drop('Cluster', axis = 1)

display(x.head(10))
print('Dimension of feature matrix:', x.shape)
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
      <th>e_pmRAU</th>
      <th>pmDEU</th>
      <th>e_pmDEU</th>
      <th>Jmag</th>
      <th>Hmag</th>
      <th>Kmag</th>
      <th>Bmag</th>
      <th>Vmag</th>
      <th>gmag</th>
      <th>rmag</th>
      <th>imag</th>
      <th>RADEcor</th>
      <th>RAplxcor</th>
      <th>RApmRAcor</th>
      <th>RApmDEcor</th>
      <th>DEplxcor</th>
      <th>DEpmRAcor</th>
      <th>DEpmDEcor</th>
      <th>plxpmRAcor</th>
      <th>plxpmDEcor</th>
      <th>pmRApmDEcor</th>
      <th>distL1350</th>
      <th>MG</th>
      <th>parallax</th>
      <th>parallax_error</th>
      <th>pmra_x</th>
      <th>pmra_error</th>
      <th>pmdec</th>
      <th>pmdec_error</th>
      <th>phot_g_mean_flux</th>
      <th>phot_g_mean_flux_error</th>
      <th>phot_g_mean_mag</th>
      <th>phot_bp_mean_flux</th>
      <th>phot_bp_mean_flux_error</th>
      <th>phot_bp_mean_mag</th>
      <th>phot_rp_mean_flux</th>
      <th>phot_rp_mean_flux_error</th>
      <th>phot_rp_mean_mag</th>
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
      <td>2.1</td>
      <td>4.1</td>
      <td>0.9</td>
      <td>10.250</td>
      <td>9.952</td>
      <td>9.909</td>
      <td>11.996</td>
      <td>11.352</td>
      <td>11.670</td>
      <td>11.197</td>
      <td>11.095</td>
      <td>-0.272335</td>
      <td>-0.518683</td>
      <td>-0.379245</td>
      <td>0.292385</td>
      <td>0.251682</td>
      <td>0.270927</td>
      <td>-0.586840</td>
      <td>0.457989</td>
      <td>-0.587446</td>
      <td>-0.638892</td>
      <td>277.230646</td>
      <td>3.994662</td>
      <td>3.9379</td>
      <td>0.0228</td>
      <td>6.531</td>
      <td>0.040</td>
      <td>2.448</td>
      <td>0.046</td>
      <td>600016.0</td>
      <td>265.5230</td>
      <td>11.242959</td>
      <td>324976.0</td>
      <td>351.452</td>
      <td>11.571761</td>
      <td>392632.0</td>
      <td>245.0740</td>
      <td>10.776957</td>
      <td>0.794805</td>
      <td>-13.03</td>
      <td>0.57</td>
      <td>13</td>
      <td>5788.67</td>
      <td>0.1840</td>
      <td>0.0898</td>
      <td>1.23</td>
      <td>1.527</td>
      <td>0.102400</td>
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
      <td>1.2</td>
      <td>7.0</td>
      <td>0.8</td>
      <td>11.436</td>
      <td>11.203</td>
      <td>11.162</td>
      <td>12.783</td>
      <td>12.230</td>
      <td>12.490</td>
      <td>12.166</td>
      <td>12.085</td>
      <td>-0.197784</td>
      <td>-0.744938</td>
      <td>-0.444089</td>
      <td>0.408160</td>
      <td>0.120876</td>
      <td>0.228539</td>
      <td>-0.504532</td>
      <td>0.413613</td>
      <td>-0.580993</td>
      <td>-0.705289</td>
      <td>748.379977</td>
      <td>2.751269</td>
      <td>1.6554</td>
      <td>0.0233</td>
      <td>7.324</td>
      <td>0.039</td>
      <td>4.502</td>
      <td>0.046</td>
      <td>249855.0</td>
      <td>44.6974</td>
      <td>12.194143</td>
      <td>144214.0</td>
      <td>240.634</td>
      <td>12.453870</td>
      <td>155655.0</td>
      <td>93.5424</td>
      <td>11.781514</td>
      <td>0.672356</td>
      <td>46.70</td>
      <td>15.75</td>
      <td>14</td>
      <td>6371.67</td>
      <td>0.7793</td>
      <td>0.3970</td>
      <td>1.54</td>
      <td>3.531</td>
      <td>0.131510</td>
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
      <td>0.9</td>
      <td>-22.5</td>
      <td>0.8</td>
      <td>9.587</td>
      <td>9.426</td>
      <td>9.340</td>
      <td>10.759</td>
      <td>10.353</td>
      <td>10.828</td>
      <td>10.321</td>
      <td>10.248</td>
      <td>-0.000539</td>
      <td>-0.116033</td>
      <td>-0.356900</td>
      <td>0.033740</td>
      <td>-0.000838</td>
      <td>0.131725</td>
      <td>-0.579400</td>
      <td>0.275152</td>
      <td>-0.493566</td>
      <td>-0.471448</td>
      <td>309.505120</td>
      <td>2.784933</td>
      <td>3.0343</td>
      <td>0.0304</td>
      <td>-21.861</td>
      <td>0.043</td>
      <td>-24.604</td>
      <td>0.058</td>
      <td>1468280.0</td>
      <td>334.4990</td>
      <td>10.271345</td>
      <td>883632.0</td>
      <td>582.120</td>
      <td>10.485710</td>
      <td>849367.0</td>
      <td>841.3060</td>
      <td>9.939182</td>
      <td>0.546528</td>
      <td>-0.24</td>
      <td>0.74</td>
      <td>10</td>
      <td>6785.82</td>
      <td>0.8210</td>
      <td>0.4230</td>
      <td>1.80</td>
      <td>6.173</td>
      <td>0.488705</td>
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
      <td>0.8</td>
      <td>15.2</td>
      <td>1.3</td>
      <td>10.358</td>
      <td>10.020</td>
      <td>9.933</td>
      <td>11.987</td>
      <td>11.319</td>
      <td>11.634</td>
      <td>11.208</td>
      <td>11.105</td>
      <td>0.104512</td>
      <td>-0.494056</td>
      <td>-0.228263</td>
      <td>0.213509</td>
      <td>-0.123917</td>
      <td>0.062234</td>
      <td>-0.434453</td>
      <td>0.393038</td>
      <td>-0.551955</td>
      <td>-0.449832</td>
      <td>237.131056</td>
      <td>4.325154</td>
      <td>4.5112</td>
      <td>0.0239</td>
      <td>10.176</td>
      <td>0.035</td>
      <td>12.850</td>
      <td>0.052</td>
      <td>590410.0</td>
      <td>234.2820</td>
      <td>11.260481</td>
      <td>322460.0</td>
      <td>296.268</td>
      <td>11.580198</td>
      <td>382941.0</td>
      <td>245.2620</td>
      <td>10.804091</td>
      <td>0.776108</td>
      <td>15.55</td>
      <td>0.50</td>
      <td>9</td>
      <td>5832.07</td>
      <td>0.0657</td>
      <td>0.0397</td>
      <td>1.05</td>
      <td>1.142</td>
      <td>0.242504</td>
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
      <td>1.8</td>
      <td>28.8</td>
      <td>0.8</td>
      <td>10.906</td>
      <td>10.638</td>
      <td>10.576</td>
      <td>12.812</td>
      <td>12.064</td>
      <td>12.424</td>
      <td>11.932</td>
      <td>11.805</td>
      <td>-0.160921</td>
      <td>-0.046372</td>
      <td>-0.325615</td>
      <td>0.082477</td>
      <td>0.226354</td>
      <td>0.279846</td>
      <td>-0.659827</td>
      <td>0.223430</td>
      <td>-0.550915</td>
      <td>-0.549265</td>
      <td>365.584123</td>
      <td>4.100549</td>
      <td>2.9493</td>
      <td>0.0229</td>
      <td>51.117</td>
      <td>0.035</td>
      <td>26.781</td>
      <td>0.053</td>
      <td>298912.0</td>
      <td>45.5289</td>
      <td>11.999507</td>
      <td>158545.0</td>
      <td>170.960</td>
      <td>12.351009</td>
      <td>200798.0</td>
      <td>126.3880</td>
      <td>11.505020</td>
      <td>0.845989</td>
      <td>50.37</td>
      <td>0.80</td>
      <td>12</td>
      <td>5705.00</td>
      <td>0.3120</td>
      <td>0.1510</td>
      <td>1.20</td>
      <td>1.364</td>
      <td>0.868035</td>
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
      <td>1.5</td>
      <td>-14.4</td>
      <td>1.1</td>
      <td>10.887</td>
      <td>10.633</td>
      <td>10.590</td>
      <td>12.432</td>
      <td>11.862</td>
      <td>12.120</td>
      <td>11.777</td>
      <td>11.678</td>
      <td>-0.041314</td>
      <td>-0.110004</td>
      <td>-0.196055</td>
      <td>-0.033316</td>
      <td>0.131761</td>
      <td>0.155971</td>
      <td>-0.559513</td>
      <td>0.377843</td>
      <td>-0.559156</td>
      <td>-0.604976</td>
      <td>341.962496</td>
      <td>4.056083</td>
      <td>3.2523</td>
      <td>0.0226</td>
      <td>5.598</td>
      <td>0.033</td>
      <td>-12.826</td>
      <td>0.041</td>
      <td>360351.0</td>
      <td>76.7663</td>
      <td>11.796552</td>
      <td>204830.0</td>
      <td>234.704</td>
      <td>12.072903</td>
      <td>227716.0</td>
      <td>143.7040</td>
      <td>11.368437</td>
      <td>0.704466</td>
      <td>-14.52</td>
      <td>1.34</td>
      <td>15</td>
      <td>6296.00</td>
      <td>0.0477</td>
      <td>0.0250</td>
      <td>0.97</td>
      <td>1.321</td>
      <td>0.212294</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1467</td>
      <td>45.060295</td>
      <td>-42.426033</td>
      <td>0.918663</td>
      <td>168</td>
      <td>10.934841</td>
      <td>29576616</td>
      <td>45.060200</td>
      <td>-42.426030</td>
      <td>6.0</td>
      <td>0.8</td>
      <td>1.1</td>
      <td>1.1</td>
      <td>9.244</td>
      <td>8.486</td>
      <td>8.345</td>
      <td>12.704</td>
      <td>11.369</td>
      <td>12.036</td>
      <td>10.987</td>
      <td>10.608</td>
      <td>-0.093020</td>
      <td>-0.152032</td>
      <td>-0.355919</td>
      <td>0.055337</td>
      <td>0.189513</td>
      <td>0.186171</td>
      <td>-0.568360</td>
      <td>0.247772</td>
      <td>-0.511690</td>
      <td>-0.599976</td>
      <td>2994.398645</td>
      <td>-1.446707</td>
      <td>0.3056</td>
      <td>0.0235</td>
      <td>10.834</td>
      <td>0.040</td>
      <td>-1.139</td>
      <td>0.047</td>
      <td>713993.0</td>
      <td>427.9750</td>
      <td>11.054131</td>
      <td>274030.0</td>
      <td>383.870</td>
      <td>11.756893</td>
      <td>623985.0</td>
      <td>631.6300</td>
      <td>10.273984</td>
      <td>1.482909</td>
      <td>35.75</td>
      <td>0.37</td>
      <td>9</td>
      <td>4447.60</td>
      <td>0.1810</td>
      <td>0.0870</td>
      <td>33.51</td>
      <td>395.956</td>
      <td>0.165147</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1469</td>
      <td>45.441105</td>
      <td>-43.439990</td>
      <td>1.067848</td>
      <td>173</td>
      <td>11.880504</td>
      <td>28644456</td>
      <td>45.441017</td>
      <td>-43.439972</td>
      <td>13.6</td>
      <td>0.9</td>
      <td>-3.5</td>
      <td>0.9</td>
      <td>11.033</td>
      <td>10.784</td>
      <td>10.719</td>
      <td>12.624</td>
      <td>12.022</td>
      <td>12.281</td>
      <td>11.923</td>
      <td>11.824</td>
      <td>-0.150867</td>
      <td>-0.088030</td>
      <td>-0.135227</td>
      <td>-0.044645</td>
      <td>0.032308</td>
      <td>0.169340</td>
      <td>-0.512578</td>
      <td>0.151651</td>
      <td>-0.408170</td>
      <td>-0.656837</td>
      <td>515.257428</td>
      <td>3.320383</td>
      <td>2.5012</td>
      <td>0.0208</td>
      <td>14.026</td>
      <td>0.031</td>
      <td>-6.100</td>
      <td>0.043</td>
      <td>312165.0</td>
      <td>56.0052</td>
      <td>11.952406</td>
      <td>175141.0</td>
      <td>189.519</td>
      <td>12.242919</td>
      <td>198831.0</td>
      <td>185.6760</td>
      <td>11.515712</td>
      <td>0.727207</td>
      <td>16.85</td>
      <td>0.85</td>
      <td>13</td>
      <td>6080.00</td>
      <td>0.0787</td>
      <td>0.0460</td>
      <td>1.26</td>
      <td>1.944</td>
      <td>0.226176</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1470</td>
      <td>45.275190</td>
      <td>-42.892350</td>
      <td>0.813470</td>
      <td>157</td>
      <td>11.404807</td>
      <td>29156958</td>
      <td>45.274900</td>
      <td>-42.892440</td>
      <td>42.4</td>
      <td>1.9</td>
      <td>23.8</td>
      <td>1.1</td>
      <td>10.395</td>
      <td>10.084</td>
      <td>10.008</td>
      <td>12.304</td>
      <td>11.581</td>
      <td>11.909</td>
      <td>11.429</td>
      <td>11.282</td>
      <td>-0.114567</td>
      <td>-0.254760</td>
      <td>-0.181153</td>
      <td>0.089632</td>
      <td>0.047278</td>
      <td>0.187820</td>
      <td>-0.519172</td>
      <td>0.567068</td>
      <td>-0.623284</td>
      <td>-0.723338</td>
      <td>237.996584</td>
      <td>4.521954</td>
      <td>4.3537</td>
      <td>0.0222</td>
      <td>46.573</td>
      <td>0.035</td>
      <td>21.619</td>
      <td>0.043</td>
      <td>484387.0</td>
      <td>132.9040</td>
      <td>11.475386</td>
      <td>254466.0</td>
      <td>258.788</td>
      <td>11.837312</td>
      <td>328234.0</td>
      <td>190.2250</td>
      <td>10.971460</td>
      <td>0.865851</td>
      <td>7.21</td>
      <td>0.53</td>
      <td>13</td>
      <td>5590.00</td>
      <td>0.1692</td>
      <td>0.0803</td>
      <td>1.08</td>
      <td>1.023</td>
      <td>0.773307</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1471</td>
      <td>45.037582</td>
      <td>-42.733380</td>
      <td>1.005230</td>
      <td>165</td>
      <td>10.870376</td>
      <td>29299147</td>
      <td>45.037476</td>
      <td>-42.733307</td>
      <td>11.0</td>
      <td>0.8</td>
      <td>-14.7</td>
      <td>0.8</td>
      <td>9.938</td>
      <td>9.646</td>
      <td>9.549</td>
      <td>11.746</td>
      <td>11.012</td>
      <td>11.357</td>
      <td>10.868</td>
      <td>10.757</td>
      <td>-0.189408</td>
      <td>-0.118002</td>
      <td>-0.141468</td>
      <td>-0.034020</td>
      <td>0.249929</td>
      <td>0.128326</td>
      <td>-0.513406</td>
      <td>0.113214</td>
      <td>-0.453080</td>
      <td>-0.621120</td>
      <td>265.375587</td>
      <td>3.751071</td>
      <td>3.9134</td>
      <td>0.0257</td>
      <td>14.700</td>
      <td>0.038</td>
      <td>-17.029</td>
      <td>0.049</td>
      <td>798748.0</td>
      <td>462.1830</td>
      <td>10.932342</td>
      <td>423141.0</td>
      <td>483.621</td>
      <td>11.285175</td>
      <td>531507.0</td>
      <td>367.2480</td>
      <td>10.448147</td>
      <td>0.837029</td>
      <td>23.95</td>
      <td>2.09</td>
      <td>11</td>
      <td>5721.28</td>
      <td>0.5515</td>
      <td>0.2730</td>
      <td>1.46</td>
      <td>2.067</td>
      <td>0.332629</td>
    </tr>
  </tbody>
</table>
</div>


    Dimension of feature matrix: (44545, 58)


#### Target variables:

Our target variables are categorical.  This means that categorical data must be converted to a numerical form using `onehot encoding` technique.

Let's get our target variables in machine learning standard.


```python
df4['Cluster'] = df4['Cluster'].astype('category')
df4 = pd.get_dummies(df4)

```

#### Take a Quick Look at the new Data Structure

Let’s take a look at the first few rows using the DataFrame’s head() method


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
      <th>e_pmRAU</th>
      <th>pmDEU</th>
      <th>e_pmDEU</th>
      <th>Jmag</th>
      <th>Hmag</th>
      <th>Kmag</th>
      <th>Bmag</th>
      <th>Vmag</th>
      <th>gmag</th>
      <th>rmag</th>
      <th>imag</th>
      <th>RADEcor</th>
      <th>RAplxcor</th>
      <th>RApmRAcor</th>
      <th>RApmDEcor</th>
      <th>DEplxcor</th>
      <th>DEpmRAcor</th>
      <th>DEpmDEcor</th>
      <th>plxpmRAcor</th>
      <th>plxpmDEcor</th>
      <th>pmRApmDEcor</th>
      <th>distL1350</th>
      <th>MG</th>
      <th>parallax</th>
      <th>parallax_error</th>
      <th>pmra_x</th>
      <th>pmra_error</th>
      <th>pmdec</th>
      <th>pmdec_error</th>
      <th>phot_g_mean_flux</th>
      <th>phot_g_mean_flux_error</th>
      <th>phot_g_mean_mag</th>
      <th>phot_bp_mean_flux</th>
      <th>phot_bp_mean_flux_error</th>
      <th>phot_bp_mean_mag</th>
      <th>phot_rp_mean_flux</th>
      <th>phot_rp_mean_flux_error</th>
      <th>phot_rp_mean_mag</th>
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
      <td>2.1</td>
      <td>4.1</td>
      <td>0.9</td>
      <td>10.250</td>
      <td>9.952</td>
      <td>9.909</td>
      <td>11.996</td>
      <td>11.352</td>
      <td>11.670</td>
      <td>11.197</td>
      <td>11.095</td>
      <td>-0.272335</td>
      <td>-0.518683</td>
      <td>-0.379245</td>
      <td>0.292385</td>
      <td>0.251682</td>
      <td>0.270927</td>
      <td>-0.586840</td>
      <td>0.457989</td>
      <td>-0.587446</td>
      <td>-0.638892</td>
      <td>277.230646</td>
      <td>3.994662</td>
      <td>3.9379</td>
      <td>0.0228</td>
      <td>6.531</td>
      <td>0.040</td>
      <td>2.448</td>
      <td>0.046</td>
      <td>600016.0</td>
      <td>265.5230</td>
      <td>11.242959</td>
      <td>324976.0</td>
      <td>351.452</td>
      <td>11.571761</td>
      <td>392632.0</td>
      <td>245.0740</td>
      <td>10.776957</td>
      <td>0.794805</td>
      <td>-13.03</td>
      <td>0.57</td>
      <td>13</td>
      <td>5788.67</td>
      <td>0.1840</td>
      <td>0.0898</td>
      <td>1.23</td>
      <td>1.527</td>
      <td>0.102400</td>
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
      <td>1.2</td>
      <td>7.0</td>
      <td>0.8</td>
      <td>11.436</td>
      <td>11.203</td>
      <td>11.162</td>
      <td>12.783</td>
      <td>12.230</td>
      <td>12.490</td>
      <td>12.166</td>
      <td>12.085</td>
      <td>-0.197784</td>
      <td>-0.744938</td>
      <td>-0.444089</td>
      <td>0.408160</td>
      <td>0.120876</td>
      <td>0.228539</td>
      <td>-0.504532</td>
      <td>0.413613</td>
      <td>-0.580993</td>
      <td>-0.705289</td>
      <td>748.379977</td>
      <td>2.751269</td>
      <td>1.6554</td>
      <td>0.0233</td>
      <td>7.324</td>
      <td>0.039</td>
      <td>4.502</td>
      <td>0.046</td>
      <td>249855.0</td>
      <td>44.6974</td>
      <td>12.194143</td>
      <td>144214.0</td>
      <td>240.634</td>
      <td>12.453870</td>
      <td>155655.0</td>
      <td>93.5424</td>
      <td>11.781514</td>
      <td>0.672356</td>
      <td>46.70</td>
      <td>15.75</td>
      <td>14</td>
      <td>6371.67</td>
      <td>0.7793</td>
      <td>0.3970</td>
      <td>1.54</td>
      <td>3.531</td>
      <td>0.131510</td>
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
      <td>0.9</td>
      <td>-22.5</td>
      <td>0.8</td>
      <td>9.587</td>
      <td>9.426</td>
      <td>9.340</td>
      <td>10.759</td>
      <td>10.353</td>
      <td>10.828</td>
      <td>10.321</td>
      <td>10.248</td>
      <td>-0.000539</td>
      <td>-0.116033</td>
      <td>-0.356900</td>
      <td>0.033740</td>
      <td>-0.000838</td>
      <td>0.131725</td>
      <td>-0.579400</td>
      <td>0.275152</td>
      <td>-0.493566</td>
      <td>-0.471448</td>
      <td>309.505120</td>
      <td>2.784933</td>
      <td>3.0343</td>
      <td>0.0304</td>
      <td>-21.861</td>
      <td>0.043</td>
      <td>-24.604</td>
      <td>0.058</td>
      <td>1468280.0</td>
      <td>334.4990</td>
      <td>10.271345</td>
      <td>883632.0</td>
      <td>582.120</td>
      <td>10.485710</td>
      <td>849367.0</td>
      <td>841.3060</td>
      <td>9.939182</td>
      <td>0.546528</td>
      <td>-0.24</td>
      <td>0.74</td>
      <td>10</td>
      <td>6785.82</td>
      <td>0.8210</td>
      <td>0.4230</td>
      <td>1.80</td>
      <td>6.173</td>
      <td>0.488705</td>
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
      <td>0.8</td>
      <td>15.2</td>
      <td>1.3</td>
      <td>10.358</td>
      <td>10.020</td>
      <td>9.933</td>
      <td>11.987</td>
      <td>11.319</td>
      <td>11.634</td>
      <td>11.208</td>
      <td>11.105</td>
      <td>0.104512</td>
      <td>-0.494056</td>
      <td>-0.228263</td>
      <td>0.213509</td>
      <td>-0.123917</td>
      <td>0.062234</td>
      <td>-0.434453</td>
      <td>0.393038</td>
      <td>-0.551955</td>
      <td>-0.449832</td>
      <td>237.131056</td>
      <td>4.325154</td>
      <td>4.5112</td>
      <td>0.0239</td>
      <td>10.176</td>
      <td>0.035</td>
      <td>12.850</td>
      <td>0.052</td>
      <td>590410.0</td>
      <td>234.2820</td>
      <td>11.260481</td>
      <td>322460.0</td>
      <td>296.268</td>
      <td>11.580198</td>
      <td>382941.0</td>
      <td>245.2620</td>
      <td>10.804091</td>
      <td>0.776108</td>
      <td>15.55</td>
      <td>0.50</td>
      <td>9</td>
      <td>5832.07</td>
      <td>0.0657</td>
      <td>0.0397</td>
      <td>1.05</td>
      <td>1.142</td>
      <td>0.242504</td>
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
      <td>1.8</td>
      <td>28.8</td>
      <td>0.8</td>
      <td>10.906</td>
      <td>10.638</td>
      <td>10.576</td>
      <td>12.812</td>
      <td>12.064</td>
      <td>12.424</td>
      <td>11.932</td>
      <td>11.805</td>
      <td>-0.160921</td>
      <td>-0.046372</td>
      <td>-0.325615</td>
      <td>0.082477</td>
      <td>0.226354</td>
      <td>0.279846</td>
      <td>-0.659827</td>
      <td>0.223430</td>
      <td>-0.550915</td>
      <td>-0.549265</td>
      <td>365.584123</td>
      <td>4.100549</td>
      <td>2.9493</td>
      <td>0.0229</td>
      <td>51.117</td>
      <td>0.035</td>
      <td>26.781</td>
      <td>0.053</td>
      <td>298912.0</td>
      <td>45.5289</td>
      <td>11.999507</td>
      <td>158545.0</td>
      <td>170.960</td>
      <td>12.351009</td>
      <td>200798.0</td>
      <td>126.3880</td>
      <td>11.505020</td>
      <td>0.845989</td>
      <td>50.37</td>
      <td>0.80</td>
      <td>12</td>
      <td>5705.00</td>
      <td>0.3120</td>
      <td>0.1510</td>
      <td>1.20</td>
      <td>1.364</td>
      <td>0.868035</td>
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
      <td>1.5</td>
      <td>-14.4</td>
      <td>1.1</td>
      <td>10.887</td>
      <td>10.633</td>
      <td>10.590</td>
      <td>12.432</td>
      <td>11.862</td>
      <td>12.120</td>
      <td>11.777</td>
      <td>11.678</td>
      <td>-0.041314</td>
      <td>-0.110004</td>
      <td>-0.196055</td>
      <td>-0.033316</td>
      <td>0.131761</td>
      <td>0.155971</td>
      <td>-0.559513</td>
      <td>0.377843</td>
      <td>-0.559156</td>
      <td>-0.604976</td>
      <td>341.962496</td>
      <td>4.056083</td>
      <td>3.2523</td>
      <td>0.0226</td>
      <td>5.598</td>
      <td>0.033</td>
      <td>-12.826</td>
      <td>0.041</td>
      <td>360351.0</td>
      <td>76.7663</td>
      <td>11.796552</td>
      <td>204830.0</td>
      <td>234.704</td>
      <td>12.072903</td>
      <td>227716.0</td>
      <td>143.7040</td>
      <td>11.368437</td>
      <td>0.704466</td>
      <td>-14.52</td>
      <td>1.34</td>
      <td>15</td>
      <td>6296.00</td>
      <td>0.0477</td>
      <td>0.0250</td>
      <td>0.97</td>
      <td>1.321</td>
      <td>0.212294</td>
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


Let's get names of target variables:


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

But, `Decision Trees` and `Random Forest` need very little data preparation. In particular, they don’t require feature scaling or centering at all.  However `k-Nearest Neighbor` is sensitive to the feature scales.


```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_val)

x_train_std = scaler.transform(x_train)
X_val_std = scaler.transform(X_val)
```

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
RF = RandomForestClassifier(n_estimators = 100, random_state = 42)
RF.fit(x_train, y_train)

print('\n-----------------------Random Forest Classifier---------------------')
print('Accuracy in training set: {:.2f}' .format(accuracy_score(y_train, RF.predict(x_train))))
print('Accuracy in validation set: {:.2f}' .format(accuracy_score(Y_val, RF.predict(X_val))))

from sklearn.neighbors import KNeighborsClassifier
KNN = Pipeline(steps=[('preprocessor', preprocessing.StandardScaler()),
                     ('model', KNeighborsClassifier(n_neighbors = 10))])
KNN.fit(x_train, y_train)

print('\n---------------------k-Nearest Neighbor Classifier------------------')
print('Accuracy in training set: {:.2f}' .format(accuracy_score(y_train, KNN.predict(x_train))))
print('Accuracy in validation set: {:.2f}' .format(accuracy_score(Y_val, KNN.predict(X_val))))


```

    -----------------------Decision Tree Classifier---------------------
    Accuracy in training set: 1.00
    Accuracy in validation set: 1.00

    -----------------------Random Forest Classifier---------------------
    Accuracy in training set: 1.00
    Accuracy in validation set: 1.00

    ---------------------k-Nearest Neighbor Classifier------------------
    Accuracy in training set: 0.94
    Accuracy in validation set: 0.93


One might be tempted to think that, `Decision Tree` and `Random Forest` Classifier are overfitting.  `We know that accuracy works well on balanced data.  The data is imbalanced, so we cannot use accuracy to quantify model performance. So we need another perfomance measure for imbalanced data.  We shall consider using f1 score metric to quantify the perfomance.`


```python

```

# 4. Model evaluation

We shall now evaluate machine learning models in the `validation set` which consitute 10% of the training data.  We shall now show a confusion matrix showing the frequency of misclassifications by our
classifier.  


```python
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
```

### Confusion Matrix


```python
#predictions
pred1 = DT.predict(X_val)

#target_names = order

mat1 = confusion_matrix(Y_val.values.argmax(axis=1), pred1.argmax(axis=1))

#Normalise
cmat1 = mat1.astype('float')/mat1.sum(axis=1)[:, np.newaxis]

plt.figure(figsize = (15,12))
plt.subplot(2, 2, 1)

sns.set(font_scale=0.9)
sns.heatmap(cmat1, cbar = True, square=True, annot=True,yticklabels = labels,
            xticklabels = labels ,annot_kws={'size': 9}, cmap='RdPu')
plt.title('Decesion Tree Classifier')
plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.tight_layout()


#predictions
pred2 = RF.predict(X_val)

#target_names = order

mat2 = confusion_matrix(Y_val.values.argmax(axis=1), pred2.argmax(axis=1))

#Normalise
cmat2 = mat2.astype('float')/mat2.sum(axis=1)[:, np.newaxis]

plt.subplot(2, 2, 2)
sns.set(font_scale=0.9)
sns.heatmap(cmat2, cbar = True, square=True, annot=True,yticklabels = labels,
            xticklabels = labels, annot_kws={'size': 9}, cmap='RdPu')
plt.title('Random Forest Classifier')
plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.tight_layout()


#predictions
pred3 = KNN.predict(X_val)

#target_names = order

mat3 = confusion_matrix(Y_val.values.argmax(axis=1), pred3.argmax(axis=1))

#Normalise
cmat3 = mat3.astype('float')/mat3.sum(axis=1)[:, np.newaxis]

plt.subplot(2, 2, 3)
sns.set(font_scale=0.9)
sns.heatmap(cmat3, cbar = True, square=True, annot=True,yticklabels = labels,
            xticklabels = labels, annot_kws={'size': 9}, cmap='RdPu')
plt.title('k-Nearest Neighbor Classifier')
plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.tight_layout()
```


![png](https://drive.google.com/uc?export=view&id=1p3cTYJnPl28PvbZ3wRijLos5ordlf_fh)


### Classification Report


```python
summ_conf1 = classification_report(Y_val.values.argmax(axis=1), pred1.argmax(axis=1), target_names = labels)


print('\n----------------------Decision Tree Classifier-----------------------\n')
print('\n', summ_conf1)

summ_conf2 = classification_report(Y_val.values.argmax(axis=1), pred2.argmax(axis=1), target_names = labels)


print('\n----------------------Random Forest Classifier-----------------------\n')
print('\n', summ_conf2)

summ_conf3 = classification_report(Y_val.values.argmax(axis=1), pred3.argmax(axis=1), target_names = labels)

print('\n--------------------k-Nearest Neighbor Classifier---------------------\n')
print('\n', summ_conf3)


```


    ----------------------Decision Tree Classifier-----------------------


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


    ----------------------Random Forest Classifier-----------------------


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


    --------------------k-Nearest Neighbor Classifier---------------------


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



##  Fine-Tune Models

Let's fiddle with the hyperparameters manually, until you find a great combination of hyperparameter values. This can be very tedious work and time consuming to explore many combinations.

We shall use **Randomized Search**.  `RandomizedSearchCV` class can be used in much the same way as the `GridSearchCV` class, but instead of trying out all possible combinations, it evaluates a given number of random combinations by selecting a random value for each hyperparameter at every iteration.

Let's check parameters of each model:


```python
print("\033[1m"+'Decision Tree Classifier:'+"\033[10m")
display(DT.get_params)

print("\033[1m"+'Random Forest Classifier:'+"\033[10m")
display(RF.get_params)

print("\033[1m"+'K-Nearest Neighbors Classifier:'+"\033[10m")
display(KNN.get_params)
```

    [1mDecision Tree Classifier:[10m



    <bound method BaseEstimator.get_params of DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best')>


    [1mRandom Forest Classifier:[10m



    <bound method BaseEstimator.get_params of RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
                oob_score=False, random_state=42, verbose=0, warm_start=False)>


    [1mK-Nearest Neighbors Classifier:[10m



    <bound method Pipeline.get_params of Pipeline(memory=None,
         steps=[('preprocessor', StandardScaler(copy=True, with_mean=True, with_std=True)), ('model', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=None, n_neighbors=10, p=2,
               weights='uniform'))])>



```python
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV
```

#### Decision Tree


```python
params1 = {"max_depth": [3, 32, None], "min_samples_leaf": [np.random.randint(1,9)],
               "criterion": ["gini","entropy"],'max_depth': [1,32], "max_features": [2, 4, 6] }


DT_search = RandomizedSearchCV(DT, param_distributions=params1, random_state=42,
                               n_iter = 200, cv = 5, verbose = 1, return_train_score=True)
DT_search.fit(X_val, Y_val)

print('\n-----------------------Decision Tree Classifier---------------------')

```

    Fitting 5 folds for each of 12 candidates, totalling 60 fits


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.



    -----------------------Decision Tree Classifier---------------------


    [Parallel(n_jobs=1)]: Done  60 out of  60 | elapsed:    5.0s finished


#### Random Forest

**Warning**: the next cell may take time to run, depending on your hardware.


```python
#parameter settings
params2 = {"max_depth": [3, 32, None],'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8],
           'bootstrap': [False, True], 'n_estimators': [3, 10, 15], 'max_features': [2, 3, 4],
           "criterion": ["gini","entropy"]}


RF_search = RandomizedSearchCV(RF, param_distributions=params2, random_state=42,
                               cv = 5, verbose = 1, return_train_score=True)

RF_search.fit(X_val, Y_val)


print('\n-----------------------Random Forest Classifier---------------------')

```

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.


    Fitting 5 folds for each of 10 candidates, totalling 50 fits


    [Parallel(n_jobs=1)]: Done  50 out of  50 | elapsed:   17.6s finished



    -----------------------Random Forest Classifier---------------------


#### k-Nearest Neighbors

**Warning**: the next cell may take time to run, depending on your hardware.


```python
params3 = {'weights':('uniform', 'distance'), 'n_neighbors':[6, 10, 12]}

KNN = KNeighborsClassifier(n_neighbors = 10)

KNN_search = RandomizedSearchCV(KNN, param_distributions=params3, random_state=42,
                                    cv = 5, verbose=1, return_train_score=True)

KNN_search.fit(X_val_std, Y_val)


print('\n-----------------------k-Nearest Neighbor Classifier---------------------\n')

```

    Fitting 5 folds for each of 6 candidates, totalling 30 fits


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.



    -----------------------k-Nearest Neighbor Classifier---------------------



    [Parallel(n_jobs=1)]: Done  30 out of  30 | elapsed:  1.5min finished


### Confusion Matrix (after parameter tunning)


```python
#predictions
pred1 = DT_search.predict(X_val)

#target_names = order

mat1 = confusion_matrix(Y_val.values.argmax(axis=1), pred1.argmax(axis=1))

#Normalise
cmat1 = mat1.astype('float')/mat1.sum(axis=1)[:, np.newaxis]

plt.figure(figsize = (20,15))
plt.subplot(2, 2, 1)

sns.set(font_scale=0.9)
sns.heatmap(cmat1, cbar = True, square=True, annot=True,yticklabels = labels,
            xticklabels = labels ,annot_kws={'size': 9}, cmap='RdPu')
plt.title('Decision Tree Classifier')
plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.tight_layout()


#predictions
pred2 = RF_search.predict(X_val)

#target_names = order

mat2 = confusion_matrix(Y_val.values.argmax(axis=1), pred2.argmax(axis=1))

#Normalise
cmat2 = mat2.astype('float')/mat2.sum(axis=1)[:, np.newaxis]

plt.subplot(2, 2, 2)
sns.set(font_scale=0.9)
sns.heatmap(cmat2, cbar = True, square=True, annot=True,yticklabels = labels,
            xticklabels = labels, annot_kws={'size': 9}, cmap='RdPu')
plt.title('Random Forest Classifier')
plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.tight_layout()


#predictions
pred3 = KNN_search.predict(X_val)

#target_names = order

mat3 = confusion_matrix(Y_val.values.argmax(axis=1), pred3.argmax(axis=1))

#Normalise
cmat3 = mat3.astype('float')/mat3.sum(axis=1)[:, np.newaxis]

plt.subplot(2, 2, 3)
sns.set(font_scale=0.9)
sns.heatmap(cmat3, cbar = True, square=True, annot=True,yticklabels = labels,
            xticklabels = labels, annot_kws={'size': 9}, cmap='RdPu')
plt.title('k-Nearest Neighbor Classifier')
plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.tight_layout()
```


![png](https://drive.google.com/uc?export=view&id=1f7cIgZXPR4nEEGdEgRlsfpVe13edP8aK)


### Classification Report (after parameter tuning)


```python
summ_conf1 = classification_report(Y_val.values.argmax(axis=1), pred1.argmax(axis=1), target_names = labels)


print('\n----------------------Decision Tree Classifier-----------------------')
print('\n', summ_conf1)

summ_conf2 = classification_report(Y_val.values.argmax(axis=1), pred2.argmax(axis=1), target_names = labels)


print('\n----------------------Random Forest Classifier-----------------------')
print('\n', summ_conf2)

summ_conf3 = classification_report(Y_val.values.argmax(axis=1), pred3.argmax(axis=1),target_names = labels)


print('\n-------------------k-Nearest Neighbor Classifier----------------------')
print('\n', summ_conf3)
print('--'*33)
```


    ----------------------Decision Tree Classifier-----------------------

                           precision    recall  f1-score   support

        Cluster_ASCC_123       0.92      1.00      0.96       337
       Cluster_Alessi_13       0.99      0.99      0.99       506
        Cluster_Alessi_3       0.97      0.95      0.96       229
        Cluster_Alessi_9       0.99      0.98      0.99       362
       Cluster_Chereul_1       0.99      0.99      0.99       556
      Cluster_Platais_10       0.99      0.98      0.99       379
       Cluster_Platais_3       1.00      0.99      1.00       328
       Cluster_Platais_8       0.99      0.99      0.99       949
       Cluster_Platais_9       0.98      0.96      0.97       382
    Cluster_Ruprecht_147       1.00      1.00      1.00       427

               micro avg       0.99      0.99      0.99      4455
               macro avg       0.98      0.98      0.98      4455
            weighted avg       0.99      0.99      0.99      4455


    ----------------------Random Forest Classifier-----------------------

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


    -------------------k-Nearest Neighbor Classifier----------------------

                           precision    recall  f1-score   support

        Cluster_ASCC_123       0.07      0.76      0.13       337
       Cluster_Alessi_13       0.00      0.00      0.00       506
        Cluster_Alessi_3       0.00      0.00      0.00       229
        Cluster_Alessi_9       0.00      0.00      0.00       362
       Cluster_Chereul_1       0.03      0.02      0.02       556
      Cluster_Platais_10       0.00      0.00      0.00       379
       Cluster_Platais_3       0.28      0.33      0.30       328
       Cluster_Platais_8       0.00      0.00      0.00       949
       Cluster_Platais_9       0.00      0.00      0.00       382
    Cluster_Ruprecht_147       0.00      0.00      0.00       427

               micro avg       0.08      0.08      0.08      4455
               macro avg       0.04      0.11      0.04      4455
            weighted avg       0.03      0.08      0.03      4455

    ------------------------------------------------------------------



```python

```

**Conclusion from hyperparameter tunning:**  we see that `Decision Tree` classifier is performed well, but `Random Forest` classifier performed the best!.  More sophisticated approach may be required for parameter tunning. So, Decision Tree model is performing better than kNN model.

# 7. Predictions

We shall now make predictions on the remainin `test set` which consitute 10% of the training data


```python
#predictions in the new dataset
pred_test = RF_search.predict(X_test)

#new data frame with column features label
New_df = pd.DataFrame(data = pred_test, columns = labels)

#reverse onehot encoding to actual values
New_df['Cluster_new'] = (New_df.iloc[:, :] == 1).idxmax(1)

New_df.Cluster_new.unique()
```




    array(['Cluster_Alessi_13', 'Cluster_Ruprecht_147', 'Cluster_Chereul_1',
           'Cluster_ASCC_123', 'Cluster_Platais_8', 'Cluster_Platais_10',
           'Cluster_Platais_9', 'Cluster_Platais_3', 'Cluster_Alessi_9',
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
      <th>e_pmRAU</th>
      <th>pmDEU</th>
      <th>e_pmDEU</th>
      <th>Jmag</th>
      <th>Hmag</th>
      <th>Kmag</th>
      <th>Bmag</th>
      <th>Vmag</th>
      <th>gmag</th>
      <th>rmag</th>
      <th>imag</th>
      <th>RADEcor</th>
      <th>RAplxcor</th>
      <th>RApmRAcor</th>
      <th>RApmDEcor</th>
      <th>DEplxcor</th>
      <th>DEpmRAcor</th>
      <th>DEpmDEcor</th>
      <th>plxpmRAcor</th>
      <th>plxpmDEcor</th>
      <th>pmRApmDEcor</th>
      <th>distL1350</th>
      <th>MG</th>
      <th>parallax</th>
      <th>parallax_error</th>
      <th>pmra_x</th>
      <th>pmra_error</th>
      <th>pmdec</th>
      <th>pmdec_error</th>
      <th>phot_g_mean_flux</th>
      <th>phot_g_mean_flux_error</th>
      <th>phot_g_mean_mag</th>
      <th>phot_bp_mean_flux</th>
      <th>phot_bp_mean_flux_error</th>
      <th>phot_bp_mean_mag</th>
      <th>phot_rp_mean_flux</th>
      <th>phot_rp_mean_flux_error</th>
      <th>phot_rp_mean_mag</th>
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
      <td>1.0</td>
      <td>11.5</td>
      <td>1.7</td>
      <td>11.078</td>
      <td>10.790</td>
      <td>10.732</td>
      <td>12.614</td>
      <td>12.100</td>
      <td>12.305</td>
      <td>11.974</td>
      <td>11.852</td>
      <td>-0.387298</td>
      <td>-0.049264</td>
      <td>-0.564733</td>
      <td>0.368948</td>
      <td>-0.026722</td>
      <td>0.422826</td>
      <td>-0.738487</td>
      <td>0.252334</td>
      <td>-0.360260</td>
      <td>-0.583724</td>
      <td>587.709798</td>
      <td>3.060165</td>
      <td>1.9432</td>
      <td>0.0282</td>
      <td>-0.376</td>
      <td>0.041</td>
      <td>10.083</td>
      <td>0.052</td>
      <td>304675.0</td>
      <td>58.6869</td>
      <td>11.978774</td>
      <td>173108.0</td>
      <td>149.605</td>
      <td>12.255595</td>
      <td>193427.0</td>
      <td>112.806</td>
      <td>11.545630</td>
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
      <td>1.9</td>
      <td>-4.2</td>
      <td>1.2</td>
      <td>11.280</td>
      <td>11.166</td>
      <td>11.138</td>
      <td>12.240</td>
      <td>12.045</td>
      <td>12.034</td>
      <td>11.984</td>
      <td>11.940</td>
      <td>0.886495</td>
      <td>0.073546</td>
      <td>-0.780962</td>
      <td>-0.802752</td>
      <td>-0.147211</td>
      <td>-0.725856</td>
      <td>-0.814565</td>
      <td>-0.302497</td>
      <td>-0.207878</td>
      <td>0.941000</td>
      <td>2624.117449</td>
      <td>-0.218638</td>
      <td>1.1617</td>
      <td>0.0490</td>
      <td>5.439</td>
      <td>0.076</td>
      <td>-5.362</td>
      <td>0.066</td>
      <td>320248.0</td>
      <td>99.0433</td>
      <td>11.924648</td>
      <td>202767.0</td>
      <td>454.768</td>
      <td>12.083894</td>
      <td>176190.0</td>
      <td>234.842</td>
      <td>11.646969</td>
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
      <td>0.6</td>
      <td>-21.6</td>
      <td>1.0</td>
      <td>10.790</td>
      <td>10.542</td>
      <td>10.468</td>
      <td>12.255</td>
      <td>11.820</td>
      <td>11.909</td>
      <td>11.711</td>
      <td>11.675</td>
      <td>0.005486</td>
      <td>-0.171000</td>
      <td>0.183401</td>
      <td>0.496688</td>
      <td>0.130026</td>
      <td>0.447393</td>
      <td>-0.044589</td>
      <td>-0.232999</td>
      <td>-0.692398</td>
      <td>0.128810</td>
      <td>453.486546</td>
      <td>3.324279</td>
      <td>1.7463</td>
      <td>0.0268</td>
      <td>4.457</td>
      <td>0.049</td>
      <td>-23.371</td>
      <td>0.052</td>
      <td>404206.0</td>
      <td>103.3940</td>
      <td>11.671858</td>
      <td>231232.0</td>
      <td>222.921</td>
      <td>11.941268</td>
      <td>253312.0</td>
      <td>145.084</td>
      <td>11.252779</td>
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
      <td>1.3</td>
      <td>-6.1</td>
      <td>1.5</td>
      <td>9.132</td>
      <td>8.446</td>
      <td>8.247</td>
      <td>13.489</td>
      <td>11.917</td>
      <td>12.639</td>
      <td>11.509</td>
      <td>10.844</td>
      <td>0.571819</td>
      <td>0.705016</td>
      <td>-0.869864</td>
      <td>-0.040045</td>
      <td>0.668792</td>
      <td>-0.669302</td>
      <td>-0.062474</td>
      <td>-0.830229</td>
      <td>0.251392</td>
      <td>-0.085561</td>
      <td>1696.514860</td>
      <td>0.007233</td>
      <td>0.5592</td>
      <td>0.0284</td>
      <td>-2.663</td>
      <td>0.046</td>
      <td>-3.549</td>
      <td>0.041</td>
      <td>573516.0</td>
      <td>322.5090</td>
      <td>11.292003</td>
      <td>194531.0</td>
      <td>266.098</td>
      <td>12.128914</td>
      <td>546710.0</td>
      <td>433.669</td>
      <td>10.417527</td>
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
      <td>1.5</td>
      <td>16.0</td>
      <td>1.9</td>
      <td>10.108</td>
      <td>9.761</td>
      <td>9.679</td>
      <td>11.839</td>
      <td>11.245</td>
      <td>11.495</td>
      <td>11.081</td>
      <td>10.914</td>
      <td>-0.654065</td>
      <td>-0.486941</td>
      <td>-0.024495</td>
      <td>0.130091</td>
      <td>0.124945</td>
      <td>0.005613</td>
      <td>-0.020655</td>
      <td>0.076219</td>
      <td>-0.275376</td>
      <td>-0.732063</td>
      <td>278.395434</td>
      <td>3.851732</td>
      <td>3.4975</td>
      <td>0.0387</td>
      <td>22.712</td>
      <td>0.041</td>
      <td>14.263</td>
      <td>0.054</td>
      <td>686586.0</td>
      <td>822.4990</td>
      <td>11.096628</td>
      <td>365626.0</td>
      <td>1105.410</td>
      <td>11.443795</td>
      <td>449116.0</td>
      <td>1012.570</td>
      <td>10.631023</td>
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
      <th>pmRAU</th>
      <th>e_pmRAU</th>
      <th>pmDEU</th>
      <th>e_pmDEU</th>
      <th>Jmag</th>
      <th>Hmag</th>
      <th>Kmag</th>
      <th>Bmag</th>
      <th>Vmag</th>
      <th>gmag</th>
      <th>rmag</th>
      <th>imag</th>
      <th>RADEcor</th>
      <th>RAplxcor</th>
      <th>RApmRAcor</th>
      <th>RApmDEcor</th>
      <th>DEplxcor</th>
      <th>DEpmRAcor</th>
      <th>DEpmDEcor</th>
      <th>plxpmRAcor</th>
      <th>plxpmDEcor</th>
      <th>pmRApmDEcor</th>
      <th>distL1350</th>
      <th>MG</th>
      <th>parallax</th>
      <th>parallax_error</th>
      <th>pmra_x</th>
      <th>pmra_error</th>
      <th>pmdec</th>
      <th>pmdec_error</th>
      <th>phot_g_mean_flux</th>
      <th>phot_g_mean_flux_error</th>
      <th>phot_g_mean_mag</th>
      <th>phot_bp_mean_flux</th>
      <th>phot_bp_mean_flux_error</th>
      <th>phot_bp_mean_mag</th>
      <th>phot_rp_mean_flux</th>
      <th>phot_rp_mean_flux_error</th>
      <th>phot_rp_mean_mag</th>
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
      <td>-2.5</td>
      <td>1.0</td>
      <td>11.5</td>
      <td>1.7</td>
      <td>11.078</td>
      <td>10.790</td>
      <td>10.732</td>
      <td>12.614</td>
      <td>12.100</td>
      <td>12.305</td>
      <td>11.974</td>
      <td>11.852</td>
      <td>-0.387298</td>
      <td>-0.049264</td>
      <td>-0.564733</td>
      <td>0.368948</td>
      <td>-0.026722</td>
      <td>0.422826</td>
      <td>-0.738487</td>
      <td>0.252334</td>
      <td>-0.360260</td>
      <td>-0.583724</td>
      <td>587.709798</td>
      <td>3.060165</td>
      <td>1.9432</td>
      <td>0.0282</td>
      <td>-0.376</td>
      <td>0.041</td>
      <td>10.083</td>
      <td>0.052</td>
      <td>304675.0</td>
      <td>58.6869</td>
      <td>11.978774</td>
      <td>173108.0</td>
      <td>149.605</td>
      <td>12.255595</td>
      <td>193427.0</td>
      <td>112.806</td>
      <td>11.545630</td>
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
      <td>4.3</td>
      <td>1.9</td>
      <td>-4.2</td>
      <td>1.2</td>
      <td>11.280</td>
      <td>11.166</td>
      <td>11.138</td>
      <td>12.240</td>
      <td>12.045</td>
      <td>12.034</td>
      <td>11.984</td>
      <td>11.940</td>
      <td>0.886495</td>
      <td>0.073546</td>
      <td>-0.780962</td>
      <td>-0.802752</td>
      <td>-0.147211</td>
      <td>-0.725856</td>
      <td>-0.814565</td>
      <td>-0.302497</td>
      <td>-0.207878</td>
      <td>0.941000</td>
      <td>2624.117449</td>
      <td>-0.218638</td>
      <td>1.1617</td>
      <td>0.0490</td>
      <td>5.439</td>
      <td>0.076</td>
      <td>-5.362</td>
      <td>0.066</td>
      <td>320248.0</td>
      <td>99.0433</td>
      <td>11.924648</td>
      <td>202767.0</td>
      <td>454.768</td>
      <td>12.083894</td>
      <td>176190.0</td>
      <td>234.842</td>
      <td>11.646969</td>
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
      <td>3.4</td>
      <td>0.6</td>
      <td>-21.6</td>
      <td>1.0</td>
      <td>10.790</td>
      <td>10.542</td>
      <td>10.468</td>
      <td>12.255</td>
      <td>11.820</td>
      <td>11.909</td>
      <td>11.711</td>
      <td>11.675</td>
      <td>0.005486</td>
      <td>-0.171000</td>
      <td>0.183401</td>
      <td>0.496688</td>
      <td>0.130026</td>
      <td>0.447393</td>
      <td>-0.044589</td>
      <td>-0.232999</td>
      <td>-0.692398</td>
      <td>0.128810</td>
      <td>453.486546</td>
      <td>3.324279</td>
      <td>1.7463</td>
      <td>0.0268</td>
      <td>4.457</td>
      <td>0.049</td>
      <td>-23.371</td>
      <td>0.052</td>
      <td>404206.0</td>
      <td>103.3940</td>
      <td>11.671858</td>
      <td>231232.0</td>
      <td>222.921</td>
      <td>11.941268</td>
      <td>253312.0</td>
      <td>145.084</td>
      <td>11.252779</td>
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
      <td>-1.1</td>
      <td>1.3</td>
      <td>-6.1</td>
      <td>1.5</td>
      <td>9.132</td>
      <td>8.446</td>
      <td>8.247</td>
      <td>13.489</td>
      <td>11.917</td>
      <td>12.639</td>
      <td>11.509</td>
      <td>10.844</td>
      <td>0.571819</td>
      <td>0.705016</td>
      <td>-0.869864</td>
      <td>-0.040045</td>
      <td>0.668792</td>
      <td>-0.669302</td>
      <td>-0.062474</td>
      <td>-0.830229</td>
      <td>0.251392</td>
      <td>-0.085561</td>
      <td>1696.514860</td>
      <td>0.007233</td>
      <td>0.5592</td>
      <td>0.0284</td>
      <td>-2.663</td>
      <td>0.046</td>
      <td>-3.549</td>
      <td>0.041</td>
      <td>573516.0</td>
      <td>322.5090</td>
      <td>11.292003</td>
      <td>194531.0</td>
      <td>266.098</td>
      <td>12.128914</td>
      <td>546710.0</td>
      <td>433.669</td>
      <td>10.417527</td>
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
      <td>17.2</td>
      <td>1.5</td>
      <td>16.0</td>
      <td>1.9</td>
      <td>10.108</td>
      <td>9.761</td>
      <td>9.679</td>
      <td>11.839</td>
      <td>11.245</td>
      <td>11.495</td>
      <td>11.081</td>
      <td>10.914</td>
      <td>-0.654065</td>
      <td>-0.486941</td>
      <td>-0.024495</td>
      <td>0.130091</td>
      <td>0.124945</td>
      <td>0.005613</td>
      <td>-0.020655</td>
      <td>0.076219</td>
      <td>-0.275376</td>
      <td>-0.732063</td>
      <td>278.395434</td>
      <td>3.851732</td>
      <td>3.4975</td>
      <td>0.0387</td>
      <td>22.712</td>
      <td>0.041</td>
      <td>14.263</td>
      <td>0.054</td>
      <td>686586.0</td>
      <td>822.4990</td>
      <td>11.096628</td>
      <td>365626.0</td>
      <td>1105.410</td>
      <td>11.443795</td>
      <td>449116.0</td>
      <td>1012.570</td>
      <td>10.631023</td>
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
      <td>-1.8</td>
      <td>1.0</td>
      <td>1.5</td>
      <td>1.5</td>
      <td>8.479</td>
      <td>7.779</td>
      <td>7.548</td>
      <td>12.763</td>
      <td>11.199</td>
      <td>11.990</td>
      <td>10.673</td>
      <td>10.115</td>
      <td>-0.756910</td>
      <td>-0.593463</td>
      <td>-0.717732</td>
      <td>0.715924</td>
      <td>0.617395</td>
      <td>0.792752</td>
      <td>-0.598597</td>
      <td>0.895885</td>
      <td>-0.359961</td>
      <td>-0.575337</td>
      <td>763.562363</td>
      <td>1.106224</td>
      <td>0.7438</td>
      <td>0.0213</td>
      <td>-3.922</td>
      <td>0.039</td>
      <td>7.767</td>
      <td>0.042</td>
      <td>1030030.0</td>
      <td>306.6560</td>
      <td>10.656243</td>
      <td>342546.0</td>
      <td>382.599</td>
      <td>11.514592</td>
      <td>989685.0</td>
      <td>674.163</td>
      <td>9.773177</td>
      <td>1.741415</td>
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
      <td>-15.4</td>
      <td>1.9</td>
      <td>-1.0</td>
      <td>1.3</td>
      <td>10.176</td>
      <td>9.939</td>
      <td>9.900</td>
      <td>11.696</td>
      <td>11.235</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>10.965</td>
      <td>-0.584801</td>
      <td>-0.460041</td>
      <td>-0.550289</td>
      <td>0.603027</td>
      <td>0.813794</td>
      <td>0.859893</td>
      <td>-0.450925</td>
      <td>0.877456</td>
      <td>-0.247045</td>
      <td>-0.529922</td>
      <td>278.347396</td>
      <td>3.855244</td>
      <td>3.4514</td>
      <td>0.0304</td>
      <td>-13.982</td>
      <td>0.054</td>
      <td>2.070</td>
      <td>0.047</td>
      <td>663114.0</td>
      <td>219.2700</td>
      <td>11.134396</td>
      <td>370974.0</td>
      <td>343.082</td>
      <td>11.428028</td>
      <td>422887.0</td>
      <td>192.359</td>
      <td>10.696359</td>
      <td>0.731669</td>
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
      <td>3.6</td>
      <td>1.3</td>
      <td>2.7</td>
      <td>1.4</td>
      <td>9.282</td>
      <td>8.751</td>
      <td>8.555</td>
      <td>12.559</td>
      <td>11.356</td>
      <td>11.931</td>
      <td>10.954</td>
      <td>10.595</td>
      <td>0.896046</td>
      <td>0.954616</td>
      <td>-0.957382</td>
      <td>-0.892057</td>
      <td>0.935341</td>
      <td>-0.805175</td>
      <td>-0.933926</td>
      <td>-0.849995</td>
      <td>-0.900884</td>
      <td>0.869873</td>
      <td>3603.351920</td>
      <td>-1.887825</td>
      <td>0.9443</td>
      <td>0.0266</td>
      <td>3.220</td>
      <td>0.051</td>
      <td>3.063</td>
      <td>0.049</td>
      <td>734290.0</td>
      <td>632.9090</td>
      <td>11.023697</td>
      <td>298741.0</td>
      <td>420.886</td>
      <td>11.663151</td>
      <td>612793.0</td>
      <td>763.710</td>
      <td>10.293636</td>
      <td>1.369515</td>
      <td>22.10</td>
      <td>0.32</td>
      <td>5</td>
      <td>4640.43</td>
      <td>0.2993</td>
      <td>0.1487</td>
      <td>9.74</td>
      <td>39.638</td>
      <td>0.065104</td>
      <td>Cluster_ASCC_123</td>
    </tr>
  </tbody>
</table>
</div>


    ---------------------------------------End--------------------------------



```python

```


```python

```
