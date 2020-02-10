---
layout: "post"
author: "Unarine Tshiwawa"
---

$$\mathrm{\textbf{Credit Card Fraud Detection}}$$


Organisation that pride themselves to protect customers or businesses against fraudulent activities have to take the necessary measures to reduce monetary loss, keep customer and brand reputation high while keeping organizational efficiencies on track. Hence, staying ahead of fraudsters and bots impersonating humans.  Organizations from hospitals to banks to government agencies have to manage all of this while meeting strict compliance guidelines, managing data and IT cybersecurity and more. How can this be?  In this work, I will use machine learning method for cradit card fraud detection

I shall layout, as a data scientist, an approach that could be based on two main risk drivers of credit card transaction default prediction:. 1) Non-fraudulent transaction and 2) Fraudulent transaction. I shall demonstrate how to build robust models to effectively predict the odds of transactions.  The system is shown mostly normal instances during training, so it learns to recognize them and when it sees a new instance it can tell whether it looks like a normal one or whether it is likely an anomaly

Clearly, we see that this is a machine leaning - binary classification task - type of a problem where the machine learning algorithm learns a set of rules in order to distinguish between two possible classes: non-fraudulent and fraudulent transactions in credit cards.  The main goal in supervised learning is to learn a model (based on past observations) from labeled training data that allows us to make predictions about unseen or future data of new instances.

**Let's get started:**


```python
#get libraries
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
sns.color_palette()
%matplotlib inline

from IPython.display import display
```


```python
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
```

# Data

The dataset in this work comes from [kaggle](https://www.kaggle.com/).  The data sets contains transactions made by credit cards by cardholders.  For this credit card fraud dataset, we do not have the original features.  It contains only numerical input variables which are the result of a principal component analysis (PCA) transformation. Features V1, V2, ... V28 are the principal components obtained with PCA, features which have not been transformed with PCA is `Time` and `Amount`.  For more infomation about the data, the reader is referred to [details](https://www.kaggle.com/shayannaveed/credit-card-fraud-detection).

Read dataset:


```python
name = 'credit-card-fraud-detection.zip'

df = pd.read_csv(name, compression = 'zip')

df1 = df.copy()
```

### Features of each instance in the dataset:


```python
display(df1.columns)
```


    Index(['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
           'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
           'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
           'Class'],
          dtype='object')



```python
df1.shape
```




    (284807, 31)



The dataset contains 284807 instances, each contains 31 features.

### Take a Quick Look at the Data Structure

Let’s take a look at the first few rows using the DataFrame’s head() method


```python
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
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>V11</th>
      <th>V12</th>
      <th>V13</th>
      <th>V14</th>
      <th>V15</th>
      <th>V16</th>
      <th>V17</th>
      <th>V18</th>
      <th>V19</th>
      <th>V20</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>0.090794</td>
      <td>-0.551600</td>
      <td>-0.617801</td>
      <td>-0.991390</td>
      <td>-0.311169</td>
      <td>1.468177</td>
      <td>-0.470401</td>
      <td>0.207971</td>
      <td>0.025791</td>
      <td>0.403993</td>
      <td>0.251412</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>-0.166974</td>
      <td>1.612727</td>
      <td>1.065235</td>
      <td>0.489095</td>
      <td>-0.143772</td>
      <td>0.635558</td>
      <td>0.463917</td>
      <td>-0.114805</td>
      <td>-0.183361</td>
      <td>-0.145783</td>
      <td>-0.069083</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>0.207643</td>
      <td>0.624501</td>
      <td>0.066084</td>
      <td>0.717293</td>
      <td>-0.165946</td>
      <td>2.345865</td>
      <td>-2.890083</td>
      <td>1.109969</td>
      <td>-0.121359</td>
      <td>-2.261857</td>
      <td>0.524980</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>378.66</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>-1.387024</td>
      <td>-0.054952</td>
      <td>-0.226487</td>
      <td>0.178228</td>
      <td>0.507757</td>
      <td>-0.287924</td>
      <td>-0.631418</td>
      <td>-1.059647</td>
      <td>-0.684093</td>
      <td>1.965775</td>
      <td>-1.232622</td>
      <td>-0.208038</td>
      <td>-0.108300</td>
      <td>0.005274</td>
      <td>-0.190321</td>
      <td>-1.175575</td>
      <td>0.647376</td>
      <td>-0.221929</td>
      <td>0.062723</td>
      <td>0.061458</td>
      <td>123.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>0.817739</td>
      <td>0.753074</td>
      <td>-0.822843</td>
      <td>0.538196</td>
      <td>1.345852</td>
      <td>-1.119670</td>
      <td>0.175121</td>
      <td>-0.451449</td>
      <td>-0.237033</td>
      <td>-0.038195</td>
      <td>0.803487</td>
      <td>0.408542</td>
      <td>-0.009431</td>
      <td>0.798278</td>
      <td>-0.137458</td>
      <td>0.141267</td>
      <td>-0.206010</td>
      <td>0.502292</td>
      <td>0.219422</td>
      <td>0.215153</td>
      <td>69.99</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.0</td>
      <td>-0.425966</td>
      <td>0.960523</td>
      <td>1.141109</td>
      <td>-0.168252</td>
      <td>0.420987</td>
      <td>-0.029728</td>
      <td>0.476201</td>
      <td>0.260314</td>
      <td>-0.568671</td>
      <td>-0.371407</td>
      <td>1.341262</td>
      <td>0.359894</td>
      <td>-0.358091</td>
      <td>-0.137134</td>
      <td>0.517617</td>
      <td>0.401726</td>
      <td>-0.058133</td>
      <td>0.068653</td>
      <td>-0.033194</td>
      <td>0.084968</td>
      <td>-0.208254</td>
      <td>-0.559825</td>
      <td>-0.026398</td>
      <td>-0.371427</td>
      <td>-0.232794</td>
      <td>0.105915</td>
      <td>0.253844</td>
      <td>0.081080</td>
      <td>3.67</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4.0</td>
      <td>1.229658</td>
      <td>0.141004</td>
      <td>0.045371</td>
      <td>1.202613</td>
      <td>0.191881</td>
      <td>0.272708</td>
      <td>-0.005159</td>
      <td>0.081213</td>
      <td>0.464960</td>
      <td>-0.099254</td>
      <td>-1.416907</td>
      <td>-0.153826</td>
      <td>-0.751063</td>
      <td>0.167372</td>
      <td>0.050144</td>
      <td>-0.443587</td>
      <td>0.002821</td>
      <td>-0.611987</td>
      <td>-0.045575</td>
      <td>-0.219633</td>
      <td>-0.167716</td>
      <td>-0.270710</td>
      <td>-0.154104</td>
      <td>-0.780055</td>
      <td>0.750137</td>
      <td>-0.257237</td>
      <td>0.034507</td>
      <td>0.005168</td>
      <td>4.99</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7.0</td>
      <td>-0.644269</td>
      <td>1.417964</td>
      <td>1.074380</td>
      <td>-0.492199</td>
      <td>0.948934</td>
      <td>0.428118</td>
      <td>1.120631</td>
      <td>-3.807864</td>
      <td>0.615375</td>
      <td>1.249376</td>
      <td>-0.619468</td>
      <td>0.291474</td>
      <td>1.757964</td>
      <td>-1.323865</td>
      <td>0.686133</td>
      <td>-0.076127</td>
      <td>-1.222127</td>
      <td>-0.358222</td>
      <td>0.324505</td>
      <td>-0.156742</td>
      <td>1.943465</td>
      <td>-1.015455</td>
      <td>0.057504</td>
      <td>-0.649709</td>
      <td>-0.415267</td>
      <td>-0.051634</td>
      <td>-1.206921</td>
      <td>-1.085339</td>
      <td>40.80</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


### Dataset information:

The info() method is useful to get a quick description of the data


```python
display(df1.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 284807 entries, 0 to 284806
    Data columns (total 31 columns):
    Time      284807 non-null float64
    V1        284807 non-null float64
    V2        284807 non-null float64
    V3        284807 non-null float64
    V4        284807 non-null float64
    V5        284807 non-null float64
    V6        284807 non-null float64
    V7        284807 non-null float64
    V8        284807 non-null float64
    V9        284807 non-null float64
    V10       284807 non-null float64
    V11       284807 non-null float64
    V12       284807 non-null float64
    V13       284807 non-null float64
    V14       284807 non-null float64
    V15       284807 non-null float64
    V16       284807 non-null float64
    V17       284807 non-null float64
    V18       284807 non-null float64
    V19       284807 non-null float64
    V20       284807 non-null float64
    V21       284807 non-null float64
    V22       284807 non-null float64
    V23       284807 non-null float64
    V24       284807 non-null float64
    V25       284807 non-null float64
    V26       284807 non-null float64
    V27       284807 non-null float64
    V28       284807 non-null float64
    Amount    284807 non-null float64
    Class     284807 non-null int64
    dtypes: float64(30), int64(1)
    memory usage: 67.4 MB



    None


There are 284807 instances in the dataset, which means that it is sufficient by Machine Learning standards, and it’s perfect to get started with.  There are no null values in the dataset - that's beautiful :)

The feature - `Amount` - is the transaction amount, this feature can be used for example-dependant cost-senstive learning. Feature - `Class` - is the response variable and **it takes value 1 in case of fraud and 0 otherwise**. `Time` is the number of seconds elapsed between this transaction and the first transaction in the dataset.

PCA dimensionality reduction may be as a result to protect user identities and sensitive features(v1-v28)

### Basic statistics: Summary of each numerical attribute.


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
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>V11</th>
      <th>V12</th>
      <th>V13</th>
      <th>V14</th>
      <th>V15</th>
      <th>V16</th>
      <th>V17</th>
      <th>V18</th>
      <th>V19</th>
      <th>V20</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>284807.000000</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>284807.000000</td>
      <td>284807.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>94813.859575</td>
      <td>3.919560e-15</td>
      <td>5.688174e-16</td>
      <td>-8.769071e-15</td>
      <td>2.782312e-15</td>
      <td>-1.552563e-15</td>
      <td>2.010663e-15</td>
      <td>-1.694249e-15</td>
      <td>-1.927028e-16</td>
      <td>-3.137024e-15</td>
      <td>1.768627e-15</td>
      <td>9.170318e-16</td>
      <td>-1.810658e-15</td>
      <td>1.693438e-15</td>
      <td>1.479045e-15</td>
      <td>3.482336e-15</td>
      <td>1.392007e-15</td>
      <td>-7.528491e-16</td>
      <td>4.328772e-16</td>
      <td>9.049732e-16</td>
      <td>5.085503e-16</td>
      <td>1.537294e-16</td>
      <td>7.959909e-16</td>
      <td>5.367590e-16</td>
      <td>4.458112e-15</td>
      <td>1.453003e-15</td>
      <td>1.699104e-15</td>
      <td>-3.660161e-16</td>
      <td>-1.206049e-16</td>
      <td>88.349619</td>
      <td>0.001727</td>
    </tr>
    <tr>
      <th>std</th>
      <td>47488.145955</td>
      <td>1.958696e+00</td>
      <td>1.651309e+00</td>
      <td>1.516255e+00</td>
      <td>1.415869e+00</td>
      <td>1.380247e+00</td>
      <td>1.332271e+00</td>
      <td>1.237094e+00</td>
      <td>1.194353e+00</td>
      <td>1.098632e+00</td>
      <td>1.088850e+00</td>
      <td>1.020713e+00</td>
      <td>9.992014e-01</td>
      <td>9.952742e-01</td>
      <td>9.585956e-01</td>
      <td>9.153160e-01</td>
      <td>8.762529e-01</td>
      <td>8.493371e-01</td>
      <td>8.381762e-01</td>
      <td>8.140405e-01</td>
      <td>7.709250e-01</td>
      <td>7.345240e-01</td>
      <td>7.257016e-01</td>
      <td>6.244603e-01</td>
      <td>6.056471e-01</td>
      <td>5.212781e-01</td>
      <td>4.822270e-01</td>
      <td>4.036325e-01</td>
      <td>3.300833e-01</td>
      <td>250.120109</td>
      <td>0.041527</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>-5.640751e+01</td>
      <td>-7.271573e+01</td>
      <td>-4.832559e+01</td>
      <td>-5.683171e+00</td>
      <td>-1.137433e+02</td>
      <td>-2.616051e+01</td>
      <td>-4.355724e+01</td>
      <td>-7.321672e+01</td>
      <td>-1.343407e+01</td>
      <td>-2.458826e+01</td>
      <td>-4.797473e+00</td>
      <td>-1.868371e+01</td>
      <td>-5.791881e+00</td>
      <td>-1.921433e+01</td>
      <td>-4.498945e+00</td>
      <td>-1.412985e+01</td>
      <td>-2.516280e+01</td>
      <td>-9.498746e+00</td>
      <td>-7.213527e+00</td>
      <td>-5.449772e+01</td>
      <td>-3.483038e+01</td>
      <td>-1.093314e+01</td>
      <td>-4.480774e+01</td>
      <td>-2.836627e+00</td>
      <td>-1.029540e+01</td>
      <td>-2.604551e+00</td>
      <td>-2.256568e+01</td>
      <td>-1.543008e+01</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>54201.500000</td>
      <td>-9.203734e-01</td>
      <td>-5.985499e-01</td>
      <td>-8.903648e-01</td>
      <td>-8.486401e-01</td>
      <td>-6.915971e-01</td>
      <td>-7.682956e-01</td>
      <td>-5.540759e-01</td>
      <td>-2.086297e-01</td>
      <td>-6.430976e-01</td>
      <td>-5.354257e-01</td>
      <td>-7.624942e-01</td>
      <td>-4.055715e-01</td>
      <td>-6.485393e-01</td>
      <td>-4.255740e-01</td>
      <td>-5.828843e-01</td>
      <td>-4.680368e-01</td>
      <td>-4.837483e-01</td>
      <td>-4.988498e-01</td>
      <td>-4.562989e-01</td>
      <td>-2.117214e-01</td>
      <td>-2.283949e-01</td>
      <td>-5.423504e-01</td>
      <td>-1.618463e-01</td>
      <td>-3.545861e-01</td>
      <td>-3.171451e-01</td>
      <td>-3.269839e-01</td>
      <td>-7.083953e-02</td>
      <td>-5.295979e-02</td>
      <td>5.600000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>84692.000000</td>
      <td>1.810880e-02</td>
      <td>6.548556e-02</td>
      <td>1.798463e-01</td>
      <td>-1.984653e-02</td>
      <td>-5.433583e-02</td>
      <td>-2.741871e-01</td>
      <td>4.010308e-02</td>
      <td>2.235804e-02</td>
      <td>-5.142873e-02</td>
      <td>-9.291738e-02</td>
      <td>-3.275735e-02</td>
      <td>1.400326e-01</td>
      <td>-1.356806e-02</td>
      <td>5.060132e-02</td>
      <td>4.807155e-02</td>
      <td>6.641332e-02</td>
      <td>-6.567575e-02</td>
      <td>-3.636312e-03</td>
      <td>3.734823e-03</td>
      <td>-6.248109e-02</td>
      <td>-2.945017e-02</td>
      <td>6.781943e-03</td>
      <td>-1.119293e-02</td>
      <td>4.097606e-02</td>
      <td>1.659350e-02</td>
      <td>-5.213911e-02</td>
      <td>1.342146e-03</td>
      <td>1.124383e-02</td>
      <td>22.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>139320.500000</td>
      <td>1.315642e+00</td>
      <td>8.037239e-01</td>
      <td>1.027196e+00</td>
      <td>7.433413e-01</td>
      <td>6.119264e-01</td>
      <td>3.985649e-01</td>
      <td>5.704361e-01</td>
      <td>3.273459e-01</td>
      <td>5.971390e-01</td>
      <td>4.539234e-01</td>
      <td>7.395934e-01</td>
      <td>6.182380e-01</td>
      <td>6.625050e-01</td>
      <td>4.931498e-01</td>
      <td>6.488208e-01</td>
      <td>5.232963e-01</td>
      <td>3.996750e-01</td>
      <td>5.008067e-01</td>
      <td>4.589494e-01</td>
      <td>1.330408e-01</td>
      <td>1.863772e-01</td>
      <td>5.285536e-01</td>
      <td>1.476421e-01</td>
      <td>4.395266e-01</td>
      <td>3.507156e-01</td>
      <td>2.409522e-01</td>
      <td>9.104512e-02</td>
      <td>7.827995e-02</td>
      <td>77.165000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>172792.000000</td>
      <td>2.454930e+00</td>
      <td>2.205773e+01</td>
      <td>9.382558e+00</td>
      <td>1.687534e+01</td>
      <td>3.480167e+01</td>
      <td>7.330163e+01</td>
      <td>1.205895e+02</td>
      <td>2.000721e+01</td>
      <td>1.559499e+01</td>
      <td>2.374514e+01</td>
      <td>1.201891e+01</td>
      <td>7.848392e+00</td>
      <td>7.126883e+00</td>
      <td>1.052677e+01</td>
      <td>8.877742e+00</td>
      <td>1.731511e+01</td>
      <td>9.253526e+00</td>
      <td>5.041069e+00</td>
      <td>5.591971e+00</td>
      <td>3.942090e+01</td>
      <td>2.720284e+01</td>
      <td>1.050309e+01</td>
      <td>2.252841e+01</td>
      <td>4.584549e+00</td>
      <td>7.519589e+00</td>
      <td>3.517346e+00</td>
      <td>3.161220e+01</td>
      <td>3.384781e+01</td>
      <td>25691.160000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>


First thing to notice, the dataset has no null values.  The average amount used is 88.34$\pm$250.12

# 1. Exploratory Data Analasis

Histogram all numerical features in the dataset:


```python
df2 = df1.copy()
```


```python
df2.hist(figsize=(20,20))
plt.show()
```


![png](https://drive.google.com/uc?export=view&id=1PaQxWI4c1G73L4Ix0KRZhrFGHTvBQhvB)

Mapping numerical feature (`Class`) to categorical fearure.  O and 1 indicates all transactions that are non-fradulent and fradulent, respectively.  We are doing this bacause we want to visualise numerical features.


```python
df2['Class'] = df2['Class'].map({0:'Non-fraudulent', 1:'Fraudulent'})
df2['Class'].unique()
```




    array(['Non-fraudulent', 'Fraudulent'], dtype=object)



data viz: pie plot


```python
df2['Class'].value_counts().plot(kind = 'pie', explode=(0,0.1), autopct='%1.2f%%',
                                         shadow = True, legend = True, labels = None)

df2['Class'].value_counts()
```




    Non-fraudulent    284315
    Fraudulent           492
    Name: Class, dtype: int64





![png](https://drive.google.com/uc?export=view&id=1TQbEyHrxRNggh0J5CvWfBp2WjbCFw398)

Note: `We see that we have huge disparity between Fraudelent and non-fraudulent cases.`

PCA was already performed on this credit card dataset, to removing the redundancy for us.  Feature selection is not necessary either since the number of observations (284,807) vastly outnumbers the number of features (30), which dramatically reduces the chances of overfitting.


```python

```

## Check correlation of features

The correlation coefficient only measures linear correlations.  Let's check for linearity amoung features.  So we can easily compute the standard correlation coefficient (also called Pearson’s r) between every pair of attributes using the corr() method:


```python
corr_matrix = df1.corr()

corr_matrix['Class'].sort_values(ascending=False)

plt.figure(figsize=(12,7))
corr_mat = sns.heatmap(corr_matrix, square=True)
```

![png](https://drive.google.com/uc?export=view&id=1eHocF_Nl37tuWgLk7QAZC3IIWW84Od8P)

The correlation coefficient ranges from –1 to 1. When it is close to 1, it means that there is a strong positive correlation.  When the coefficient is close to –1, it means that there is a strong negative correlation.  Finally, coefficients close to zero mean that there is no linear correlation.

Info: We do not see any strong correlation among features, so there is no strong correlation among features.  So linear model will not be a good fit here!

# 2. Preprocessing - getting data into shape

To determine whether our machine learning algorithm not only performs well on the training set but also generalizes well to new data, we also want to randomly divide the dataset into a separate `training` and `test set`. `We use the training set to train and optimize our machine learning model`, while we keep the `test set until the very end to evaluate the final model`.


```python
#feature matrix
x = df1.drop('Class', axis = 1)

#Target vector/labels
y = df1['Class']

print(x.shape)
print(y.shape)
```

    (284807, 30)
    (284807,)


### Data Splicing

We are going to split the data into three sets -  80%, 20% for training, validation, respectively.  This means, 80% of the data will be set to train and optimize our machine learning model and 20% will be used to validate the model.


```python
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify = y)
```

Let's verify data partitioning:


```python
for dataset in [Y_train, Y_val]:

    print(round((len(dataset)/len(y))*100, 2), '%')

print('>>>>>>>>>>>> Done!!<<<<<<<<<<<<<<')
```

    80.0 %
    20.0 %
    >>>>>>>>>>>> Done!!<<<<<<<<<<<<<<


### Feature scaling

`Support vector machines` is sensitive to the feature scales.  Data are not usually presented to the machine learning algorithm in exactly the same raw form as it is found. Usually data are scaled to a specific range in a process called normalization for optimal performance.

We are going to make selected features to the same scale for optimal performance, which is often achieved by transforming the features in the range [0, 1]: standardize features by removing the mean and scaling to unit variance.





```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train_std = scaler.transform(X_train)
X_val_std = scaler.transform(X_val)

```

# 3. Machine Learning model

Now, the feature matrix and labels are ready for training. This is the part where different classifier will be instantiated (for training) and a model which performs better in both training and test set will be used for further predictions.

It is important to note that the parameters for the previously mentioned procedures, such as feature scaling and dimensionality reduction, are solely obtained from the training dataset, and the same parameters are later reapplied to transform the test dataset, as well as any new data samples—the performance measured on the test data may be overly optimistic otherwise.


```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
```

### Let's discuss different class of classifiers that will be used to train the model:

1. `LocalOutlierFactor`: Is an unsupervised outlier detection method.  It calculates an anomaly score of each sample - that's called the local outlier factor.  It measure the local deviation of density for a given sample relative to its neighbors.  In other word, this model pin-point how isolated an object is with respect to its surrounding. This method is similar to k-NN, however, we are calcu;ating an anomaly scores based on those neighbors [LocalOutlierFactor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html)

2. `IsolationForest`: Is also an unsupervised detection method.  This is a bit different from LocalOutlierFactor, it isolate the observation by randomly selecting a feature and a split value between maximum and minimum values of the selected feature [IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).

3. `C-Support Vector Classification` is a supervised machine learning algorithm capable of performing classification, regression and even outlier detection [C-Support Vector Classification](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).  

 ### The proportion of outliers in the dataset:

 This define the threshold on the scores of the samples


```python
Fraud = df1[df1['Class'] == 1] #training instance that are fraudulent
Valid = df1[df1['Class'] == 0] #training instances that are valid

#ratio of fradulent to valid cases
Outlier_frac = len(Fraud)/float(len(Valid))
Outlier_frac


print('\nFraud counts:',Fraud.shape[0], '\nNon-fraud counts:', Valid.shape[0])
print('\nProportion of outliers in the dataset:', Outlier_frac)
```


    Fraud counts: 492
    Non-fraud counts: 284315

    Proportion of outliers in the dataset: 0.0017304750013189597


### Instantiates different classes for training

Each classification algorithm has its inherent biases, and no single classification model enjoys superiority if we don't make any assumptions about the task.  It is therefore expedient to compare at least a handful of different algorithms in order to train and select the best performing model - that's what we are going to do here.  All selected models will be trained on the training data, the learning algorithm searched for the model parameter values that minimize a cost function.

`Let's first consider unsupervised learning algorithms:`


```python
#Instantiate IsolationForest class
IF_clf = IsolationForest(max_samples = len(X_train_std), contamination = Outlier_frac, random_state = 42)

#fit and predicting labels
y_pred = IF_clf.fit_predict(X_train_std)

#reshape the prediction value to 0 for valid, 1 for fraud
y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1

#Instantiate LocalOutlierFactor class
LOF_clf = LocalOutlierFactor(contamination = Outlier_frac)

#fit and predicting labels
y_pred2 = LOF_clf.fit_predict(X_train_std)

y_pred2[y_pred2 == 1] = 0
y_pred2[y_pred2 == -1] = 1
```


**Note**:
Unsupervised Pretraining, both models are more computationally expensive!!!


`Additional: let's look at this powerful supervised learning algorithm`

A C-Support Vector Classification, is part of Support Vector Machines (SVMs), is a very powerful and versatile Machine Learning model, capable of performing linear or nonlinear classification, regression, and even outlier detection.  This Machine Learning algorithm is computationally expensive to compute all the additional features, especially when training sets is large.


Parameters to adjust:

- The optimal value of the C parameter will depend on your dataset, and should be tuned via cross-validation.  If your SVC model is overfitting, we can always try regularizing it by reducing C.

- Since the training set is not too large, we shall try the Gaussian Radial Basis Function (RBF) kernel; it works well in most cases.

- So $\gamma$ acts like a regularization hyperparameter: if the model is overfitting, it must be reduced, and if it is underfitting, it must be increased (similar to the C hyperparameter).


```python
from sklearn.svm import SVC #C-Support Vector Classification

svc_clf = SVC(kernel = 'rbf', gamma = 'auto', C = 8) #instantiate C-Support Vector class


svc_clf.fit(X_train_std, Y_train) #training the model
```




    SVC(C=8, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
y_pred3 = svc_clf.predict(X_val_std)

```

# 4. Model Evaluation - selecting optimal predictive model

We shall now show a `confusion matrix` showing the frequency of misclassifications by our
classifier. to measure performance, `classification accuracy` which is defined as the proportion of correctly classified instances will be used, to see the performance of each classifier.  In this part, the `test set` will be used to evaluate each model performance

### Accuracy score


```python


print('\n\n--------------------Isolation Forest-------------------')
#print('score prediction: {:.2f}' .format(score_pred))
print('\n Accuracy score (training set): ', accuracy_score(Y_train, y_pred))
print('\n Accuracy score (test set): ', accuracy_score(Y_val, y_pred_IF))
print('\n Number of error :', (Y_val != y_pred_IF).sum())

print('\n\n--------------------LocalOutlierFactor-------------------')
#print('score prediction: {:.2f}' .format(score_pred))
print('\n Accuracy score (training set): ', accuracy_score(Y_train, y_pred2))
print('\n Accuracy score (test set): ', accuracy_score(Y_val, y_pred_LOF))
print('\n Number of error :', (Y_val != y_pred_LOF).sum())

print('\n\n--------------------Support Vector Class-------------------')
print('\n Accuracy score (training set): ', accuracy_score(Y_train, svc_clf.predict(X_train_std)))
print('\n Accuracy score (validation set): ', accuracy_score(Y_val, y_pred3))
print('\n Number of error :', (Y_val != y_pred3).sum())

```

    --------------------Isolation Forest-------------------

     Accuracy score (training set):  0.9976694682788738

     Accuracy score (test set):  0.997559776693234

     Number of error : 139


    --------------------LocalOutlierFactor-------------------

     Accuracy score (training set):  0.9965371195330158

     Accuracy score (test set):  0.9966117762719006

     Number of error : 193


    --------------------Support Vector Class-------------------


We know that accuracy works well on balanced datasets.  The dataset is highly imbalanced, so we cannot use accuracy to quantify model performance. So we need another perfomance measure for imbalanced datasets.  We shall consider using `classificication accuracy` to quantify the perfomance of each model.

### Confusion matrix

Some terms used in a confusion matrix are:

- True positives (TPs): True positives are cases when we predict the transactions as fraudulent when the transaction actually is fraudulent.

- True negatives (TNs): Cases when we predict transections as non-fradulent when transactions actually are non-fraudulant.

- False positives (FPs): When we predict the transactions as non-fraudulent  when the transactions are actually fraudulent. FPs are also considered to be type I errors.

- False negatives (FNs): When we predict the transactions as fraudulent when the transactions actually are non-fraudulent. FNs are also considered to be type II errors.


```python
names = ['Non-fradulent', 'Fradulent']

mat1 = confusion_matrix(Y_val, y_pred_IF)

plt.figure(figsize = (15,10))
plt.subplot(2, 2, 1)
sns.set(font_scale = 1.2)
sns.heatmap(mat1, cbar = True, square = True, annot = True, yticklabels = names,
            annot_kws={'size': 15}, xticklabels = names, cmap = 'RdPu')
plt.title('Isolation Forest model')
plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.tight_layout()


mat2 = confusion_matrix(Y_val, y_pred_LOF)

plt.subplot(2, 2, 2)
sns.set(font_scale = 1.2)
sns.heatmap(mat2, cbar = True, square = True, annot = True, yticklabels = names,
            annot_kws={'size': 15}, xticklabels = names, cmap = 'RdPu')
plt.title('LocalOutlierFactor model')
plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.tight_layout()

mat3 = confusion_matrix(Y_val, y_pred3)

plt.subplot(2, 2, 3)
sns.set(font_scale = 1.2)
sns.heatmap(mat3, cbar = True, square = True, annot = True, yticklabels = names,
            annot_kws={'size': 15}, xticklabels = names, cmap = 'RdPu')
plt.title('C-Support Vector model')
plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.tight_layout()
plt.show()
```


![png](https://drive.google.com/uc?export=view&id=1DT0_WUD29rWrhLswtDAjnsYzZJsg9N66)


### Classification report


```python
print('\n\n ---------------------Isolation Forest model--------------------\n')
print(classification_report(Y_val, y_pred_IF, target_names = names))

print('\n\n --------------------LocalOutlierFactor model-------------------\n')
print(classification_report(Y_val, y_pred_LOF, target_names = names))

print('\n\n -----------------------C-Support Vector model-------------------\n')
print(classification_report(Y_val, y_pred3, target_names = names))
```



     ---------------------Isolation Forest model--------------------

                   precision    recall  f1-score   support

    Non-fradulent       1.00      1.00      1.00     56864
        Fradulent       0.29      0.30      0.29        98

        micro avg       1.00      1.00      1.00     56962
        macro avg       0.65      0.65      0.65     56962
     weighted avg       1.00      1.00      1.00     56962



     --------------------LocalOutlierFactor model-------------------

                   precision    recall  f1-score   support

    Non-fradulent       1.00      1.00      1.00     56864
        Fradulent       0.02      0.02      0.02        98

        micro avg       1.00      1.00      1.00     56962
        macro avg       0.51      0.51      0.51     56962
     weighted avg       1.00      1.00      1.00     56962



     -----------------------C-Support Vector model-------------------

                   precision    recall  f1-score   support

    Non-fradulent       1.00      1.00      1.00     56864
        Fradulent       0.95      0.72      0.82        98

        micro avg       1.00      1.00      1.00     56962
        macro avg       0.97      0.86      0.91     56962
     weighted avg       1.00      1.00      1.00     56962



`LocalOutlierFactor Model`: For class 0 (Non-fraudulent transaction), we have precision of ~ 100% and for class 1 (Fraudulent transaction), we have 0.2% - that's not good.  Precision counts for false-positives and recall counts for false-negatives.

`Isolation Forest Model`: We have a precision of ~ 30%, that's better than ~ 2% obtained in `LocalOutlierFactor Model`.  But, still, we are only correctly identifying 30% of transactions that are actual fraudulent cases.  We do also have a lot of false-positives (this could frustrate customer with false alarm), and we also have a recall (false-Negative) of ~ 30%

`C-Support Vector Model`: We have achieved a precision of ~ 95%, that's better than ~ 30% obtained in `Isolation Forest Model`.  But, still, we are only correctly identifying ~ 82% of transactions that are actual fraudulent cases - that's much better.

# Conclusion



From unsupervised models and the `f1-score`, `IsolationForest` model performs better than `LocalOutlierFactor` model. The dataset used in this work was sufficiently complex, so random forest method was able to produce better results.  So 30% of the time we are going to detect fraudulent transaction, unless the culprit made 4 transactions statistically, we will find them everytime! However, `C-Support Vector` model, which is supervised performs, was able to produce best results!


```python

```
