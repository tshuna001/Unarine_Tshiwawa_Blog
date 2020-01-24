---
layout: post
author: "Unarine Tshiwawa"
---

$$\mathrm{\textbf{Credit Risk}}$$

For organisation that pride themselves in their effective use of credit risk models to deliver profitable and high-impact loan alternative. I shall layout, as a data scientist, an approach that could be based on two main risk drivers of loan default prediction:. 1) willingness to pay and 2) ability to pay. I shall demonstrate how to build robust models to effectively predict the odds of repayment.

In this task, it is my objective to predict if a loan is good or bad, i.e. accurately predict binary outcome variable, where Good is 1 and Bad is 0.


```python
from IPython.display import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling
sns.set('notebook')

```

## Dataset:

We are using data from [zindi](https://zindi.africa/competitions/data-science-nigeria-challenge-1-loan-default-prediction/data). For more information about datasets, the reader is  referred to [info](https://zindi.africa/competitions/data-science-nigeria-challenge-1-loan-default-prediction/data). The objective in this work is to create a binary classification model that predicts whether or not an individual is a good payer or not based on several indicators. The target variable is given as `good_bad_flag` and takes on a value of 1 if the client is  a good payer and 0 otherwise.

We will do some data exploratory data analysis, and then focus on learning a model. The second step will involve merging datasets and then impute the missing values. We will then split the dataset into a training and testing set randomly.  Then, we simply predict the most common class in the training set for all observations in the validation set.


Read data:


```python
train_demo = pd.read_csv('traindemographics.csv')
train_perf = pd.read_csv('trainperf.csv')
train_prev = pd.read_csv('trainprevloans.zip', compression='zip')

```

To avoid data corruption, we shall make copy of all three sets:


```python
train_demo1 = train_demo.copy()
train_perf1 = train_perf.copy()
train_prev1 = train_prev.copy()
```

### Let's consider each dataset:

### 1. Train demographics dataset

Take a Quick Look at the Data Structure. Let’s take a look at the top five rows using the DataFrame’s head() method


```python
print("\033[1m"+'Train demographics dataset:'+"\033[10m")
display(train_demo1.head(5))

print("\033[1m"+'Train performance dataset:'+"\033[10m")
display(train_perf1.head(5))

print("\033[1m"+'Train performance dataset:'+"\033[10m")
display(train_prev1.head(5))


```

    Train demographics dataset:



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
      <th>customerid</th>
      <th>birthdate</th>
      <th>bank_account_type</th>
      <th>longitude_gps</th>
      <th>latitude_gps</th>
      <th>bank_name_clients</th>
      <th>bank_branch_clients</th>
      <th>employment_status_clients</th>
      <th>level_of_education_clients</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8a858e135cb22031015cbafc76964ebd</td>
      <td>1973-10-10 00:00:00.000000</td>
      <td>Savings</td>
      <td>3.319219</td>
      <td>6.528604</td>
      <td>GT Bank</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8a858e275c7ea5ec015c82482d7c3996</td>
      <td>1986-01-21 00:00:00.000000</td>
      <td>Savings</td>
      <td>3.325598</td>
      <td>7.119403</td>
      <td>Sterling Bank</td>
      <td>NaN</td>
      <td>Permanent</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8a858e5b5bd99460015bdc95cd485634</td>
      <td>1987-04-01 00:00:00.000000</td>
      <td>Savings</td>
      <td>5.746100</td>
      <td>5.563174</td>
      <td>Fidelity Bank</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8a858efd5ca70688015cabd1f1e94b55</td>
      <td>1991-07-19 00:00:00.000000</td>
      <td>Savings</td>
      <td>3.362850</td>
      <td>6.642485</td>
      <td>GT Bank</td>
      <td>NaN</td>
      <td>Permanent</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8a858e785acd3412015acd48f4920d04</td>
      <td>1982-11-22 00:00:00.000000</td>
      <td>Savings</td>
      <td>8.455332</td>
      <td>11.971410</td>
      <td>GT Bank</td>
      <td>NaN</td>
      <td>Permanent</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    Train performance dataset:



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
      <th>customerid</th>
      <th>systemloanid</th>
      <th>loannumber</th>
      <th>approveddate</th>
      <th>creationdate</th>
      <th>loanamount</th>
      <th>totaldue</th>
      <th>termdays</th>
      <th>referredby</th>
      <th>good_bad_flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8a2a81a74ce8c05d014cfb32a0da1049</td>
      <td>301994762</td>
      <td>12</td>
      <td>2017-07-25 08:22:56.000000</td>
      <td>2017-07-25 07:22:47.000000</td>
      <td>30000.0</td>
      <td>34500.0</td>
      <td>30</td>
      <td>NaN</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8a85886e54beabf90154c0a29ae757c0</td>
      <td>301965204</td>
      <td>2</td>
      <td>2017-07-05 17:04:41.000000</td>
      <td>2017-07-05 16:04:18.000000</td>
      <td>15000.0</td>
      <td>17250.0</td>
      <td>30</td>
      <td>NaN</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8a8588f35438fe12015444567666018e</td>
      <td>301966580</td>
      <td>7</td>
      <td>2017-07-06 14:52:57.000000</td>
      <td>2017-07-06 13:52:51.000000</td>
      <td>20000.0</td>
      <td>22250.0</td>
      <td>15</td>
      <td>NaN</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8a85890754145ace015429211b513e16</td>
      <td>301999343</td>
      <td>3</td>
      <td>2017-07-27 19:00:41.000000</td>
      <td>2017-07-27 18:00:35.000000</td>
      <td>10000.0</td>
      <td>11500.0</td>
      <td>15</td>
      <td>NaN</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8a858970548359cc0154883481981866</td>
      <td>301962360</td>
      <td>9</td>
      <td>2017-07-03 23:42:45.000000</td>
      <td>2017-07-03 22:42:39.000000</td>
      <td>40000.0</td>
      <td>44000.0</td>
      <td>30</td>
      <td>NaN</td>
      <td>Good</td>
    </tr>
  </tbody>
</table>
</div>


    Train performance dataset:



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
      <th>customerid</th>
      <th>systemloanid</th>
      <th>loannumber</th>
      <th>approveddate</th>
      <th>creationdate</th>
      <th>loanamount</th>
      <th>totaldue</th>
      <th>termdays</th>
      <th>closeddate</th>
      <th>referredby</th>
      <th>firstduedate</th>
      <th>firstrepaiddate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8a2a81a74ce8c05d014cfb32a0da1049</td>
      <td>301682320</td>
      <td>2</td>
      <td>2016-08-15 18:22:40.000000</td>
      <td>2016-08-15 17:22:32.000000</td>
      <td>10000.0</td>
      <td>13000.0</td>
      <td>30</td>
      <td>2016-09-01 16:06:48.000000</td>
      <td>NaN</td>
      <td>2016-09-14 00:00:00.000000</td>
      <td>2016-09-01 15:51:43.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8a2a81a74ce8c05d014cfb32a0da1049</td>
      <td>301883808</td>
      <td>9</td>
      <td>2017-04-28 18:39:07.000000</td>
      <td>2017-04-28 17:38:53.000000</td>
      <td>10000.0</td>
      <td>13000.0</td>
      <td>30</td>
      <td>2017-05-28 14:44:49.000000</td>
      <td>NaN</td>
      <td>2017-05-30 00:00:00.000000</td>
      <td>2017-05-26 00:00:00.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8a2a81a74ce8c05d014cfb32a0da1049</td>
      <td>301831714</td>
      <td>8</td>
      <td>2017-03-05 10:56:25.000000</td>
      <td>2017-03-05 09:56:19.000000</td>
      <td>20000.0</td>
      <td>23800.0</td>
      <td>30</td>
      <td>2017-04-26 22:18:56.000000</td>
      <td>NaN</td>
      <td>2017-04-04 00:00:00.000000</td>
      <td>2017-04-26 22:03:47.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8a8588f35438fe12015444567666018e</td>
      <td>301861541</td>
      <td>5</td>
      <td>2017-04-09 18:25:55.000000</td>
      <td>2017-04-09 17:25:42.000000</td>
      <td>10000.0</td>
      <td>11500.0</td>
      <td>15</td>
      <td>2017-04-24 01:35:52.000000</td>
      <td>NaN</td>
      <td>2017-04-24 00:00:00.000000</td>
      <td>2017-04-24 00:48:43.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8a85890754145ace015429211b513e16</td>
      <td>301941754</td>
      <td>2</td>
      <td>2017-06-17 09:29:57.000000</td>
      <td>2017-06-17 08:29:50.000000</td>
      <td>10000.0</td>
      <td>11500.0</td>
      <td>15</td>
      <td>2017-07-14 21:18:43.000000</td>
      <td>NaN</td>
      <td>2017-07-03 00:00:00.000000</td>
      <td>2017-07-14 21:08:35.000000</td>
    </tr>
  </tbody>
</table>
</div>

```python

```


```python
Train demographics information:

The info() method is useful to get a quick description of the data
```

```python

print('\n--------------------Train dermographics--------------------\n')
display(train_demo1.info())
print('\n----------------------Train performance--------------------\n')
display(train_perf1.info())
print('\n-----------------------Train previous----------------------\n')
display(train_perf1.info())
```


    --------------------Train dermographics--------------------

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4346 entries, 0 to 4345
    Data columns (total 9 columns):
    customerid                    4346 non-null object
    birthdate                     4346 non-null object
    bank_account_type             4346 non-null object
    longitude_gps                 4346 non-null float64
    latitude_gps                  4346 non-null float64
    bank_name_clients             4346 non-null object
    bank_branch_clients           51 non-null object
    employment_status_clients     3698 non-null object
    level_of_education_clients    587 non-null object
    dtypes: float64(2), object(7)
    memory usage: 305.7+ KB



    None



    ----------------------Train performance--------------------

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4368 entries, 0 to 4367
    Data columns (total 10 columns):
    customerid       4368 non-null object
    systemloanid     4368 non-null int64
    loannumber       4368 non-null int64
    approveddate     4368 non-null object
    creationdate     4368 non-null object
    loanamount       4368 non-null float64
    totaldue         4368 non-null float64
    termdays         4368 non-null int64
    referredby       587 non-null object
    good_bad_flag    4368 non-null object
    dtypes: float64(2), int64(3), object(5)
    memory usage: 341.3+ KB



    None



    -----------------------Train previous----------------------

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4368 entries, 0 to 4367
    Data columns (total 10 columns):
    customerid       4368 non-null object
    systemloanid     4368 non-null int64
    loannumber       4368 non-null int64
    approveddate     4368 non-null object
    creationdate     4368 non-null object
    loanamount       4368 non-null float64
    totaldue         4368 non-null float64
    termdays         4368 non-null int64
    referredby       587 non-null object
    good_bad_flag    4368 non-null object
    dtypes: float64(2), int64(3), object(5)
    memory usage: 341.3+ KB



    None


Train demographics: there are 4346 instances in the dataset, that it is fairly small by Machine Learning standards, but it’s perfect to get started.  All attributes are categorical, except gps coordinates (`longitude_gps`, `latitude_gps`) which are numerical.


Train performance: there are 4368 instances in the dataset.  It is also fairly small by Machine Learning standards, but it’s perfect to get started. Attributes are categorical (`customerid`,`approveddate`,`creationdate`,`referredby`,`good_bad_flag`) and numerical (`systemloanid`,`loannumber`,`loanamount`,`totaldue`,`termdays`).


Train previous: We also have 4386 instances in the dataset, 5 categorical (`customerid`,`approveddate`,`creationdate`,`referredby`,`good_bad_flag`) and 5 numerical variables (`systemloanid`,`loannumber`,`loanamount`,`totaldue`, `termdays`).

```IPython

```

#### Basic statistics: Summary of each numerical attribute:


```python

print('\n--------------------Train dermographics--------------------\n')
display(train_demo1.describe())
print('\n----------------------Train performance--------------------\n')
display(train_perf1.describe())
print('\n-----------------------Train previous----------------------\n')
display(train_perf1.describe())

```


    --------------------Train dermographics--------------------




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
      <th>longitude_gps</th>
      <th>latitude_gps</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4346.000000</td>
      <td>4346.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.626189</td>
      <td>7.251356</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.184832</td>
      <td>3.055052</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-118.247009</td>
      <td>-33.868818</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.354953</td>
      <td>6.470610</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.593302</td>
      <td>6.621888</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.545220</td>
      <td>7.425052</td>
    </tr>
    <tr>
      <th>max</th>
      <td>151.209290</td>
      <td>71.228069</td>
    </tr>
  </tbody>
</table>
</div>



    ----------------------Train performance--------------------




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
      <th>systemloanid</th>
      <th>loannumber</th>
      <th>loanamount</th>
      <th>totaldue</th>
      <th>termdays</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4.368000e+03</td>
      <td>4368.000000</td>
      <td>4368.000000</td>
      <td>4368.000000</td>
      <td>4368.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.019810e+08</td>
      <td>5.172390</td>
      <td>17809.065934</td>
      <td>21257.377679</td>
      <td>29.261676</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.343115e+04</td>
      <td>3.653569</td>
      <td>10749.694571</td>
      <td>11943.510416</td>
      <td>11.512519</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.019585e+08</td>
      <td>2.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>15.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.019691e+08</td>
      <td>2.000000</td>
      <td>10000.000000</td>
      <td>13000.000000</td>
      <td>30.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.019801e+08</td>
      <td>4.000000</td>
      <td>10000.000000</td>
      <td>13000.000000</td>
      <td>30.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.019935e+08</td>
      <td>7.000000</td>
      <td>20000.000000</td>
      <td>24500.000000</td>
      <td>30.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.020040e+08</td>
      <td>27.000000</td>
      <td>60000.000000</td>
      <td>68100.000000</td>
      <td>90.000000</td>
    </tr>
  </tbody>
</table>
</div>



    -----------------------Train previous----------------------




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
      <th>systemloanid</th>
      <th>loannumber</th>
      <th>loanamount</th>
      <th>totaldue</th>
      <th>termdays</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4.368000e+03</td>
      <td>4368.000000</td>
      <td>4368.000000</td>
      <td>4368.000000</td>
      <td>4368.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.019810e+08</td>
      <td>5.172390</td>
      <td>17809.065934</td>
      <td>21257.377679</td>
      <td>29.261676</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.343115e+04</td>
      <td>3.653569</td>
      <td>10749.694571</td>
      <td>11943.510416</td>
      <td>11.512519</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.019585e+08</td>
      <td>2.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>15.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.019691e+08</td>
      <td>2.000000</td>
      <td>10000.000000</td>
      <td>13000.000000</td>
      <td>30.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.019801e+08</td>
      <td>4.000000</td>
      <td>10000.000000</td>
      <td>13000.000000</td>
      <td>30.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.019935e+08</td>
      <td>7.000000</td>
      <td>20000.000000</td>
      <td>24500.000000</td>
      <td>30.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.020040e+08</td>
      <td>27.000000</td>
      <td>60000.000000</td>
      <td>68100.000000</td>
      <td>90.000000</td>
    </tr>
  </tbody>
</table>
</div>

```python

```
The first thing to notice about these three datasets is that `Train dermographics dataset` differ with  the other two dataset by size and attributes.

The count , mean , min , and max rows are self-explanatory.  Note that the null values are ignored. The std row shows the standard deviation, which measures how dispersed the values are. The 25%, 50%, and 75% rows show the corresponding percentiles: a percentile indicates the value below which a given percentage of observations in a group of observations falls

### Impute for missing values


```python
print('\n--------------------Train dermographics--------------------\n')
display(train_demo1.isnull().sum())
print('\n----------------------Train performance--------------------\n')
display(train_perf1.isnull().sum())
print('\n-----------------------Train previous----------------------\n')
display(train_prev1.isnull().sum())
```


    --------------------Train dermographics--------------------




    customerid                       0
    birthdate                        0
    bank_account_type                0
    longitude_gps                    0
    latitude_gps                     0
    bank_name_clients                0
    bank_branch_clients           4295
    employment_status_clients      648
    level_of_education_clients    3759
    dtype: int64



    ----------------------Train performance--------------------




    customerid          0
    systemloanid        0
    loannumber          0
    approveddate        0
    creationdate        0
    loanamount          0
    totaldue            0
    termdays            0
    referredby       3781
    good_bad_flag       0
    dtype: int64



    -----------------------Train previous----------------------




    customerid             0
    systemloanid           0
    loannumber             0
    approveddate           0
    creationdate           0
    loanamount             0
    totaldue               0
    termdays               0
    closeddate             0
    referredby         17157
    firstduedate           0
    firstrepaiddate        0
    dtype: int64


## EXPLORATORY DATA ANALYSIS

For numerical attributes:


```python
print("\033[1m"+'Train demographics dataset:'+"\033[10m")
display(train_demo1.hist(bins=50, figsize=(12,3)))
```

    [1mTrain demographics dataset:[10m



    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x7fda11b6e668>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fda11354c18>]],
          dtype=object)




![png](https://drive.google.com/uc?export=view&id=1wusAhL6vcHf0EKOraDSVjuwSKpeXrMDl)

#### Geographical location:
The location columns are the only numeric variables in our demographics dataset. It also has no missing data, which makes sense as proof of address is a major requirement when dealing with financial institutions.


```python
import folium as fl

#base map
gps_map = fl.Map(location=[(train_demo1['latitude_gps']).mean(), (train_demo1['longitude_gps']).mean()],
                 zoom_start = 2)


for i, j in zip(train_demo1['latitude_gps'], train_demo1['longitude_gps']):

    fl.CircleMarker(radius = 1, location = [i, j],
                    color = 'red',
                    fill = True,
                    fill_opacity = 1
                   ).add_to(gps_map)
gps_map
```







```python

```

<font color=green>Note: for better view, the map above can be zoomed in and out.</font>

Setting the alpha option to 0.3 makes it much easier to visualize the places where there is a high density of data points


```python
train_demo1.plot(kind="scatter", x="longitude_gps", y="latitude_gps",figsize=(10,7) ,alpha=0.3)
```

    'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.


    <matplotlib.axes._subplots.AxesSubplot at 0x7fda0f88de48>

![png](https://drive.google.com/uc?export=view&id=1NKOKE3XZ6jpUNhdUZYlErwlrXt8NFAcT)

Insight: It appears that many clients are mainly scattered across Nigeria (Africa), Lagos and Abuja host two noticeable clusters, and the geographical concentration of clients gradually becomes sparsely dense with distance from the Western coast.


```python
print("\033[1m"+'Train performance dataset:'+"\033[10m")
display(train_perf1.hist(bins=50, figsize=(12,10)))
```

    Train performance dataset:


![png](https://drive.google.com/uc?export=view&id=1K-uz8njK1Q4dlK6VIHn7-sNNJHHp7jol)


```python
print("\033[1m"+'Train previous dataset:'+"\033[10m")
display(train_prev1.hist(bins=30, figsize=(12,10)))
```

    Train previous dataset:

![png](https://drive.google.com/uc?export=view&id=1AhGxIpygHRtxUZiSi2YgHq4ejUHXHKNM)

### Merging datasets:

Approach:  we shall merge `demographics dataset` and `train performance dataset` using customerid.  Then, the resultant dataset will be merged with `train previous dataset` using customerid.


```python
df = pd.merge(train_demo1, train_perf1, how = 'inner', on = 'customerid')

print(train_demo.shape, train_perf.shape)
display(df.head(5))
```

    (4346, 9) (4368, 10)



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
      <th>customerid</th>
      <th>birthdate</th>
      <th>bank_account_type</th>
      <th>longitude_gps</th>
      <th>latitude_gps</th>
      <th>bank_name_clients</th>
      <th>bank_branch_clients</th>
      <th>employment_status_clients</th>
      <th>level_of_education_clients</th>
      <th>systemloanid</th>
      <th>loannumber</th>
      <th>approveddate</th>
      <th>creationdate</th>
      <th>loanamount</th>
      <th>totaldue</th>
      <th>termdays</th>
      <th>referredby</th>
      <th>good_bad_flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8a858e135cb22031015cbafc76964ebd</td>
      <td>1973-10-10 00:00:00.000000</td>
      <td>Savings</td>
      <td>3.319219</td>
      <td>6.528604</td>
      <td>GT Bank</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>301964962</td>
      <td>2</td>
      <td>2017-07-05 14:29:48.000000</td>
      <td>2017-07-05 13:29:42.000000</td>
      <td>10000.0</td>
      <td>13000.0</td>
      <td>30</td>
      <td>8a858899538ddb8e0153a780c56e34bb</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8a858e275c7ea5ec015c82482d7c3996</td>
      <td>1986-01-21 00:00:00.000000</td>
      <td>Savings</td>
      <td>3.325598</td>
      <td>7.119403</td>
      <td>Sterling Bank</td>
      <td>NaN</td>
      <td>Permanent</td>
      <td>NaN</td>
      <td>301972172</td>
      <td>2</td>
      <td>2017-07-10 21:21:46.000000</td>
      <td>2017-07-10 20:21:40.000000</td>
      <td>10000.0</td>
      <td>13000.0</td>
      <td>30</td>
      <td>NaN</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8a858e5b5bd99460015bdc95cd485634</td>
      <td>1987-04-01 00:00:00.000000</td>
      <td>Savings</td>
      <td>5.746100</td>
      <td>5.563174</td>
      <td>Fidelity Bank</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>301976271</td>
      <td>4</td>
      <td>2017-07-13 15:40:27.000000</td>
      <td>2017-07-13 14:40:19.000000</td>
      <td>10000.0</td>
      <td>13000.0</td>
      <td>30</td>
      <td>NaN</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8a858efd5ca70688015cabd1f1e94b55</td>
      <td>1991-07-19 00:00:00.000000</td>
      <td>Savings</td>
      <td>3.362850</td>
      <td>6.642485</td>
      <td>GT Bank</td>
      <td>NaN</td>
      <td>Permanent</td>
      <td>NaN</td>
      <td>301997763</td>
      <td>2</td>
      <td>2017-07-26 21:03:17.000000</td>
      <td>2017-07-26 20:03:09.000000</td>
      <td>10000.0</td>
      <td>11500.0</td>
      <td>15</td>
      <td>NaN</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8a858ea05a859123015a8892914d15b7</td>
      <td>1990-07-21 00:00:00.000000</td>
      <td>Savings</td>
      <td>3.365935</td>
      <td>6.564823</td>
      <td>Access Bank</td>
      <td>NaN</td>
      <td>Permanent</td>
      <td>NaN</td>
      <td>301992494</td>
      <td>6</td>
      <td>2017-07-23 21:44:43.000000</td>
      <td>2017-07-23 20:44:36.000000</td>
      <td>20000.0</td>
      <td>24500.0</td>
      <td>30</td>
      <td>NaN</td>
      <td>Good</td>
    </tr>
  </tbody>
</table>
</div>



```python
df1 = pd.merge(df, train_prev, how = 'inner', on = 'customerid')
df1.shape
```




    (13693, 29)




```python
display(df1.head(5))
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
      <th>customerid</th>
      <th>birthdate</th>
      <th>bank_account_type</th>
      <th>longitude_gps</th>
      <th>latitude_gps</th>
      <th>bank_name_clients</th>
      <th>bank_branch_clients</th>
      <th>employment_status_clients</th>
      <th>level_of_education_clients</th>
      <th>systemloanid_x</th>
      <th>...</th>
      <th>loannumber_y</th>
      <th>approveddate_y</th>
      <th>creationdate_y</th>
      <th>loanamount_y</th>
      <th>totaldue_y</th>
      <th>termdays_y</th>
      <th>closeddate</th>
      <th>referredby_y</th>
      <th>firstduedate</th>
      <th>firstrepaiddate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8a858e135cb22031015cbafc76964ebd</td>
      <td>1973-10-10 00:00:00.000000</td>
      <td>Savings</td>
      <td>3.319219</td>
      <td>6.528604</td>
      <td>GT Bank</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>301964962</td>
      <td>...</td>
      <td>1</td>
      <td>2017-06-19 17:55:26.000000</td>
      <td>2017-06-19 16:54:19.000000</td>
      <td>10000.0</td>
      <td>11500.0</td>
      <td>15</td>
      <td>2017-07-04 18:09:47.000000</td>
      <td>8a858899538ddb8e0153a780c56e34bb</td>
      <td>2017-07-05 00:00:00.000000</td>
      <td>2017-07-04 17:59:36.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8a858e275c7ea5ec015c82482d7c3996</td>
      <td>1986-01-21 00:00:00.000000</td>
      <td>Savings</td>
      <td>3.325598</td>
      <td>7.119403</td>
      <td>Sterling Bank</td>
      <td>NaN</td>
      <td>Permanent</td>
      <td>NaN</td>
      <td>301972172</td>
      <td>...</td>
      <td>1</td>
      <td>2017-06-07 12:47:30.000000</td>
      <td>2017-06-07 11:46:22.000000</td>
      <td>10000.0</td>
      <td>13000.0</td>
      <td>30</td>
      <td>2017-07-10 08:52:54.000000</td>
      <td>NaN</td>
      <td>2017-07-07 00:00:00.000000</td>
      <td>2017-07-10 08:42:44.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8a858e5b5bd99460015bdc95cd485634</td>
      <td>1987-04-01 00:00:00.000000</td>
      <td>Savings</td>
      <td>5.746100</td>
      <td>5.563174</td>
      <td>Fidelity Bank</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>301976271</td>
      <td>...</td>
      <td>3</td>
      <td>2017-06-08 11:49:34.000000</td>
      <td>2017-06-08 10:49:27.000000</td>
      <td>10000.0</td>
      <td>13000.0</td>
      <td>30</td>
      <td>2017-07-11 10:12:20.000000</td>
      <td>NaN</td>
      <td>2017-07-10 00:00:00.000000</td>
      <td>2017-07-11 10:02:11.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8a858e5b5bd99460015bdc95cd485634</td>
      <td>1987-04-01 00:00:00.000000</td>
      <td>Savings</td>
      <td>5.746100</td>
      <td>5.563174</td>
      <td>Fidelity Bank</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>301976271</td>
      <td>...</td>
      <td>1</td>
      <td>2017-05-08 11:07:01.000000</td>
      <td>2017-05-08 10:06:40.000000</td>
      <td>10000.0</td>
      <td>11500.0</td>
      <td>15</td>
      <td>2017-05-27 13:02:53.000000</td>
      <td>NaN</td>
      <td>2017-05-23 00:00:00.000000</td>
      <td>2017-05-27 12:52:45.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8a858e5b5bd99460015bdc95cd485634</td>
      <td>1987-04-01 00:00:00.000000</td>
      <td>Savings</td>
      <td>5.746100</td>
      <td>5.563174</td>
      <td>Fidelity Bank</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>301976271</td>
      <td>...</td>
      <td>2</td>
      <td>2017-05-27 17:10:41.000000</td>
      <td>2017-05-27 16:10:34.000000</td>
      <td>10000.0</td>
      <td>11500.0</td>
      <td>15</td>
      <td>2017-06-08 11:13:50.000000</td>
      <td>NaN</td>
      <td>2017-06-12 00:00:00.000000</td>
      <td>2017-06-08 11:03:40.000000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 29 columns</p>
</div>


Let's remove redundant column:


```python
df1 = df1.drop(['bank_account_type','bank_name_clients','bank_branch_clients',
                'level_of_education_clients','systemloanid_x','loannumber_x',
               'referredby_x','referredby_y', 'approveddate_x', 'creationdate_x',
               'approveddate_y','creationdate_y','closeddate','firstduedate','firstrepaiddate',
                'systemloanid_y','loannumber_y','loanamount_y','totaldue_y'], axis = 1)
```


```python
df1.head()
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
      <th>customerid</th>
      <th>birthdate</th>
      <th>longitude_gps</th>
      <th>latitude_gps</th>
      <th>employment_status_clients</th>
      <th>loanamount_x</th>
      <th>totaldue_x</th>
      <th>termdays_x</th>
      <th>good_bad_flag</th>
      <th>termdays_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8a858e135cb22031015cbafc76964ebd</td>
      <td>1973-10-10 00:00:00.000000</td>
      <td>3.319219</td>
      <td>6.528604</td>
      <td>NaN</td>
      <td>10000.0</td>
      <td>13000.0</td>
      <td>30</td>
      <td>Good</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8a858e275c7ea5ec015c82482d7c3996</td>
      <td>1986-01-21 00:00:00.000000</td>
      <td>3.325598</td>
      <td>7.119403</td>
      <td>Permanent</td>
      <td>10000.0</td>
      <td>13000.0</td>
      <td>30</td>
      <td>Bad</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8a858e5b5bd99460015bdc95cd485634</td>
      <td>1987-04-01 00:00:00.000000</td>
      <td>5.746100</td>
      <td>5.563174</td>
      <td>NaN</td>
      <td>10000.0</td>
      <td>13000.0</td>
      <td>30</td>
      <td>Good</td>
      <td>30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8a858e5b5bd99460015bdc95cd485634</td>
      <td>1987-04-01 00:00:00.000000</td>
      <td>5.746100</td>
      <td>5.563174</td>
      <td>NaN</td>
      <td>10000.0</td>
      <td>13000.0</td>
      <td>30</td>
      <td>Good</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8a858e5b5bd99460015bdc95cd485634</td>
      <td>1987-04-01 00:00:00.000000</td>
      <td>5.746100</td>
      <td>5.563174</td>
      <td>NaN</td>
      <td>10000.0</td>
      <td>13000.0</td>
      <td>30</td>
      <td>Good</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>



### Imputing missing values:


```python
display(df1.isna().sum())
```


    customerid                      0
    birthdate                       0
    longitude_gps                   0
    latitude_gps                    0
    employment_status_clients    1363
    loanamount_x                    0
    totaldue_x                      0
    termdays_x                      0
    good_bad_flag                   0
    termdays_y                      0
    dtype: int64


We see that `employment_status_clients` columns contains 1363 instances that are nans, so we will drop those instances -- that means, more training data will be lost!


```python
df1 = df1.dropna()

df1['employment_status_clients'].value_counts()
```




    Permanent        9138
    Self-Employed    2319
    Student           615
    Unemployed        219
    Retired            29
    Contract           10
    Name: employment_status_clients, dtype: int64



Let's make a pie plot for our categorical feature after imputing nan values


```python
pie1 = df1['employment_status_clients'].value_counts().plot(kind = 'pie', explode=(0,0,0,0,0,0.1), autopct='%1.2f%%',
                                                shadow = True, legend = "upper left", labels = None)
pie1
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fda0d78e710>




![png](https://drive.google.com/uc?export=view&id=1U-U7oOkZROEoNm2Q8aFYCQqybfn1y0Vr)

Replace binary variables to numerical features in `employment_status_clients`:


```python
df1['employment_status_clients'] = df1['employment_status_clients'].replace({'Permanent':5, 'Self-Employed':4,
                                                                             'Student':3, 'Unemployed':2,
                                                                            'Retired':1, 'Contract':0})
```

Let's make a quick view of a new dataset:


```python
display(df1.head(5))
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
      <th>customerid</th>
      <th>birthdate</th>
      <th>longitude_gps</th>
      <th>latitude_gps</th>
      <th>employment_status_clients</th>
      <th>loanamount_x</th>
      <th>totaldue_x</th>
      <th>termdays_x</th>
      <th>good_bad_flag</th>
      <th>termdays_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>8a858e275c7ea5ec015c82482d7c3996</td>
      <td>1986-01-21 00:00:00.000000</td>
      <td>3.325598</td>
      <td>7.119403</td>
      <td>5</td>
      <td>10000.0</td>
      <td>13000.0</td>
      <td>30</td>
      <td>Bad</td>
      <td>30</td>
    </tr>
    <tr>
      <th>5</th>
      <td>8a858efd5ca70688015cabd1f1e94b55</td>
      <td>1991-07-19 00:00:00.000000</td>
      <td>3.362850</td>
      <td>6.642485</td>
      <td>5</td>
      <td>10000.0</td>
      <td>11500.0</td>
      <td>15</td>
      <td>Good</td>
      <td>15</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8a858ea05a859123015a8892914d15b7</td>
      <td>1990-07-21 00:00:00.000000</td>
      <td>3.365935</td>
      <td>6.564823</td>
      <td>5</td>
      <td>20000.0</td>
      <td>24500.0</td>
      <td>30</td>
      <td>Good</td>
      <td>30</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8a858ea05a859123015a8892914d15b7</td>
      <td>1990-07-21 00:00:00.000000</td>
      <td>3.365935</td>
      <td>6.564823</td>
      <td>5</td>
      <td>20000.0</td>
      <td>24500.0</td>
      <td>30</td>
      <td>Good</td>
      <td>30</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8a858ea05a859123015a8892914d15b7</td>
      <td>1990-07-21 00:00:00.000000</td>
      <td>3.365935</td>
      <td>6.564823</td>
      <td>5</td>
      <td>20000.0</td>
      <td>24500.0</td>
      <td>30</td>
      <td>Good</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>


We have duplicate columns `termdays`, we shall remove `termdays_y`


```python
df1 = df1.drop(['termdays_y'], axis = 1)
```


```python
display(df1.info())
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 12330 entries, 1 to 13691
    Data columns (total 9 columns):
    customerid                   12330 non-null object
    birthdate                    12330 non-null object
    longitude_gps                12330 non-null float64
    latitude_gps                 12330 non-null float64
    employment_status_clients    12330 non-null int64
    loanamount_x                 12330 non-null float64
    totaldue_x                   12330 non-null float64
    termdays_x                   12330 non-null int64
    good_bad_flag                12330 non-null object
    dtypes: float64(4), int64(2), object(3)
    memory usage: 963.3+ KB



    None


<font color=green> For a new dataset: there are now 12330 instances and it is fairly reasonable by Machine Learning standards.</font>

All attributes are numerical, except `customerid`, `birthdate` and `good_bad_flag` which are categorical features.


Let's view a pie chart for our target variable:


```python
pie2 = df1['good_bad_flag'].value_counts().plot(kind = 'pie', explode=(0,0.1), autopct='%1.2f%%',
                                                shadow = True, legend = True, labels = None)
pie2
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fda0d7b02e8>



![png](https://drive.google.com/uc?export=view&id=1H3BGVTfNHw7w16OwSroCCKLvspnbbYzT)

NB: - One important thing to note, the dataset is imbalanced. This is an imbalanced class problem because there are significantly more clients who are good payers than bad payers!

## Feature engineering (for a new dataset)

Convert categorical variables to numerical variables for machine learning:


```python
df1['good_bad_flag'] = df1['good_bad_flag'].replace({'Good':1, 'Bad':0})
df1['good_bad_flag'].unique()
```




    array([0, 1])



### Converting birthdate to age:



```python
now = pd.Timestamp('now')

df1['birthdate'] = pd.to_datetime(df1['birthdate'])
df1['birthdate'] = df1['birthdate'].where(df1['birthdate'] < now, df1['birthdate'] -  np.timedelta64(100, 'Y'))
df1['age'] = (now - df1['birthdate']).astype('<m8[Y]')

df1 = df1.drop('birthdate', axis = 1)

```

Let's plot the distribution of age:


```python
#sns.distplot(df1['age'], bins = 30)
plt.figure()
sns.distplot(df1['age'], bins = 30)
plt.xlabel('Age [years]')
plt.show()
```
![png](https://drive.google.com/uc?export=view&id=15-xj31dWZDfja218uNSwWIlD4wqFms3k)

Insight: The average age is around 35.


```python
print('\n-------------------------------------------Basic statistics--------------------------------------------')
display(df1.describe())
```


    -------------------------------------------Basic statistics--------------------------------------------



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
      <th>longitude_gps</th>
      <th>latitude_gps</th>
      <th>employment_status_clients</th>
      <th>loanamount_x</th>
      <th>totaldue_x</th>
      <th>termdays_x</th>
      <th>good_bad_flag</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>12330.000000</td>
      <td>12330.000000</td>
      <td>12330.000000</td>
      <td>12330.000000</td>
      <td>12330.000000</td>
      <td>12330.000000</td>
      <td>12330.000000</td>
      <td>12330.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.536353</td>
      <td>7.307844</td>
      <td>4.645418</td>
      <td>26337.388483</td>
      <td>30507.083698</td>
      <td>33.736010</td>
      <td>0.823682</td>
      <td>34.765288</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.769082</td>
      <td>3.563097</td>
      <td>0.692577</td>
      <td>12768.636498</td>
      <td>13916.967496</td>
      <td>14.801045</td>
      <td>0.381106</td>
      <td>6.122289</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-118.247009</td>
      <td>-33.868818</td>
      <td>0.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>15.000000</td>
      <td>0.000000</td>
      <td>23.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.355216</td>
      <td>6.466717</td>
      <td>4.000000</td>
      <td>15000.000000</td>
      <td>16500.000000</td>
      <td>30.000000</td>
      <td>1.000000</td>
      <td>30.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.579151</td>
      <td>6.615100</td>
      <td>5.000000</td>
      <td>30000.000000</td>
      <td>34500.000000</td>
      <td>30.000000</td>
      <td>1.000000</td>
      <td>34.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.661640</td>
      <td>7.422043</td>
      <td>5.000000</td>
      <td>40000.000000</td>
      <td>44000.000000</td>
      <td>30.000000</td>
      <td>1.000000</td>
      <td>38.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>151.209290</td>
      <td>71.228069</td>
      <td>5.000000</td>
      <td>60000.000000</td>
      <td>68100.000000</td>
      <td>90.000000</td>
      <td>1.000000</td>
      <td>58.000000</td>
    </tr>
  </tbody>
</table>
</div>


We could use a correlation threshold for removing variables, if there exist some linearity. In this case, we will probably want to keep all of the variables and let the model decide which are relevant.


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
      <th>customerid</th>
      <th>longitude_gps</th>
      <th>latitude_gps</th>
      <th>employment_status_clients</th>
      <th>loanamount_x</th>
      <th>totaldue_x</th>
      <th>termdays_x</th>
      <th>good_bad_flag</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>8a858e275c7ea5ec015c82482d7c3996</td>
      <td>3.325598</td>
      <td>7.119403</td>
      <td>5</td>
      <td>10000.0</td>
      <td>13000.0</td>
      <td>30</td>
      <td>0</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>8a858efd5ca70688015cabd1f1e94b55</td>
      <td>3.362850</td>
      <td>6.642485</td>
      <td>5</td>
      <td>10000.0</td>
      <td>11500.0</td>
      <td>15</td>
      <td>1</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8a858ea05a859123015a8892914d15b7</td>
      <td>3.365935</td>
      <td>6.564823</td>
      <td>5</td>
      <td>20000.0</td>
      <td>24500.0</td>
      <td>30</td>
      <td>1</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8a858ea05a859123015a8892914d15b7</td>
      <td>3.365935</td>
      <td>6.564823</td>
      <td>5</td>
      <td>20000.0</td>
      <td>24500.0</td>
      <td>30</td>
      <td>1</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8a858ea05a859123015a8892914d15b7</td>
      <td>3.365935</td>
      <td>6.564823</td>
      <td>5</td>
      <td>20000.0</td>
      <td>24500.0</td>
      <td>30</td>
      <td>1</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>8a858ea05a859123015a8892914d15b7</td>
      <td>3.365935</td>
      <td>6.564823</td>
      <td>5</td>
      <td>20000.0</td>
      <td>24500.0</td>
      <td>30</td>
      <td>1</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>8a858ea05a859123015a8892914d15b7</td>
      <td>3.365935</td>
      <td>6.564823</td>
      <td>5</td>
      <td>20000.0</td>
      <td>24500.0</td>
      <td>30</td>
      <td>1</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>8a858f405d13c45f015d13dd93ec0c1c</td>
      <td>3.290590</td>
      <td>6.612075</td>
      <td>5</td>
      <td>10000.0</td>
      <td>13000.0</td>
      <td>30</td>
      <td>1</td>
      <td>27.0</td>
    </tr>
  </tbody>
</table>
</div>


## Correlations

We can calculate correlation values to see how the features are related to the outcome--`good_bad_flag`. Correlation does not of course imply causation, but because we are building a model, the correlated features are likely useful for learning a mapping between the clients information and whether or not they are good payers.


```python
df1.corr()['good_bad_flag'].sort_values(ascending=False)
```




    good_bad_flag                1.000000
    loanamount_x                 0.095799
    totaldue_x                   0.085127
    age                          0.046017
    employment_status_clients   -0.007333
    termdays_x                  -0.007663
    longitude_gps               -0.011899
    latitude_gps                -0.014939
    Name: good_bad_flag, dtype: float64



<font color=red>Clearly, we can see that linear model will not be a good fit here!</font>

# Preprocesing:

- In this step, we shall prepare the dataset for machine learning algorithms.


```python
from sklearn.model_selection import train_test_split


#feature matrix
x = df1.drop(['good_bad_flag','customerid'], axis = 1)

#target vector/s
y = df1['good_bad_flag']

```

Let's view feature matrix and target vector:


```python
print('\n------------------------------------------------Matrix feature------------------------------------------')
display(x.head(5))

print('\n------------------------------------------------Target vector------------------------------------------')
display(y.head(5))
print('\n--------------------------------------------------------------------------------------------------------|')
```


    ------------------------------------------------Matrix feature------------------------------------------



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
      <th>longitude_gps</th>
      <th>latitude_gps</th>
      <th>employment_status_clients</th>
      <th>loanamount_x</th>
      <th>totaldue_x</th>
      <th>termdays_x</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>3.325598</td>
      <td>7.119403</td>
      <td>5</td>
      <td>10000.0</td>
      <td>13000.0</td>
      <td>30</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3.362850</td>
      <td>6.642485</td>
      <td>5</td>
      <td>10000.0</td>
      <td>11500.0</td>
      <td>15</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3.365935</td>
      <td>6.564823</td>
      <td>5</td>
      <td>20000.0</td>
      <td>24500.0</td>
      <td>30</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3.365935</td>
      <td>6.564823</td>
      <td>5</td>
      <td>20000.0</td>
      <td>24500.0</td>
      <td>30</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3.365935</td>
      <td>6.564823</td>
      <td>5</td>
      <td>20000.0</td>
      <td>24500.0</td>
      <td>30</td>
      <td>29.0</td>
    </tr>
  </tbody>
</table>
</div>



    ------------------------------------------------Target vector------------------------------------------



    1    0
    5    1
    6    1
    7    1
    8    1
    Name: good_bad_flag, dtype: int64



    --------------------------------------------------------------------------------------------------------|


### Data splicing:

We are going to split the data into two sets -  80% train set and 20% validation set.  This means, 80% of the data will be set to train and optimize our machine learning model, and the remaining 20% will be used to validate the model.


```python
from sklearn.model_selection import train_test_split


#data splicing
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.20, random_state = 1, stratify = y)

print(x_train.shape)
print(x_val.shape)
```

    (9864, 7)
    (2466, 7)


Let's confirm again if data partition criteria was archieved!


```python
for dataset in [y_train, y_val]:

    print(round((len(dataset)/len(y))*100, 2))

print('>>>>>>>>>>>> Done!!<<<<<<<<<<<<<<')

```

    80.0
    20.0
    >>>>>>>>>>>> Done!!<<<<<<<<<<<<<<


## Machine Learning models

Data are not usually presented to the machine learning algorithm in exactly the same raw form as it is found. Usually data are scaled to a specific range in a process called normalization.

For linear model, we are going to make selected features to the same scale for optimal performance, which is often achieved by transforming the features in the range [0, 1]: standardize features by removing the mean and scaling to unit variance  using `StandardScaler()` class.  This will be done in the pipeline to run multiple processes in the order that they are listed. The purpose and advantage of using pipeline is to assemble several steps that can be cross-validated together while setting different parameters.


<font color=red>However, one of the many qualities of `Decision Trees` and `Random Forest` is that they require very little data preparation. In particular, they don’t require feature scaling or centering at all.</font>


```python
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score
#from sklearn.model_selection import GridSearchCV

#Logistic regression
from sklearn.linear_model import LogisticRegression

LR = Pipeline(steps = [('preprocessor', preprocessing.StandardScaler()),
                     ('model', LogisticRegression(random_state = 0))])
LR.fit(x_train, y_train)

print('\n--------------------LogisticRegression Classifier-------------------')
print('Accuracy in training set: {:.2f}' .format(accuracy_score(y_train, LR.predict(x_train))))
print('Accuracy in validation set: {:.2f}' .format(accuracy_score(y_val, LR.predict(x_val))))



#Decision Tree
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(x_train, y_train)

print('\n-----------------------Decision Tree Classifier---------------------')
print('Accuracy in training set: {:.2f}' .format(accuracy_score(y_train, DT.predict(x_train))))
print('Accuracy in validation set: {:.2f}' .format(accuracy_score(y_val, DT.predict(x_val))))



#Random forest
from sklearn.ensemble import RandomForestClassifier

#instantiate random forest class
RF = RandomForestClassifier(random_state = 0, n_jobs=-1)

RF.fit(x_train, y_train)

print('\n-----------------------Random Forest Classifier---------------------')
print('Accuracy in training set: {:.2f}' .format(accuracy_score(y_train, RF.predict(x_train))))
print('Accuracy in validation set: {:.2f}' .format(accuracy_score(y_val, RF.predict(x_val))))


```


    --------------------LogisticRegression Classifier-------------------
    Accuracy in training set: 0.82
    Accuracy in validation set: 0.82

    -----------------------Decision Tree Classifier---------------------
    Accuracy in training set: 1.00
    Accuracy in validation set: 0.97

    -----------------------Random Forest Classifier---------------------
    Accuracy in training set: 1.00
    Accuracy in validation set: 0.97


<font color=green>We know that accuracy works well on balanced datasets.  The dataset is imbalanced, so we cannot use accuracy to quantify model performance. So we need another perfomance measure for imbalanced datasets.  We shall consider using `f1 score metric` to quantify the perfomance.</font>

# Model evaluation

We shall now show a `confusion matrix` showing the frequency of misclassifications by our
classifier. We shall also look at `receiver operating characteristics (ROC)` & `area under curve (AUC)` to see the performance of each classifier accross various thresholds.

For a perfect model, the classifier performance, has AUC = 1.  Of course that's not achievable in real-world situations.
So we want the classifier that will achieve AUC value as close as possible to 1.

### 1. Logistic Regression


```python
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

#predictions
pred1 = LR.predict(x_val)

mat1 = confusion_matrix(y_val, pred1)

target_names =  ['Bad payer', 'Good payer']

plt.figure(figsize = (7,6))
sns.set(font_scale = 1.2)
sns.heatmap(mat1, cbar = True, square = True, annot = True, yticklabels = target_names,
            annot_kws={'size': 15}, xticklabels=target_names, cmap = 'RdPu')

plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.tight_layout()
plt.show()

```

![png](https://drive.google.com/uc?export=view&id=1sdBcNp7RpLO1f7_gHTeaY3KxU5Nqoyej)

```python
summ_conf1 = classification_report(y_val, pred1, target_names = target_names)


print('\n------------Logistic Regression Classification Report----------------')
print('\n', summ_conf1)
print('--'*33)
```


    ------------Logistic Regression Classification Report----------------

                   precision    recall  f1-score   support

       Bad payer       0.00      0.00      0.00       435
      Good payer       0.82      1.00      0.90      2031

       micro avg       0.82      0.82      0.82      2466
       macro avg       0.41      0.50      0.45      2466
    weighted avg       0.68      0.82      0.74      2466

    ------------------------------------------------------------------


### ROC & AUC - Logistic Regression:

The receiver operating characteristic (ROC) curve is another common tool used with binary classifiers. The higher the recall (TPR), the more false positives (FPR) the classifier produces. The dotted line represents the ROC curve of a purely random classifier; a good classifier stays as far away from that line as possible (toward the top-left corner).


```python
#predicting class probabilities for input x_val
probs1 = LR.predict_proba(x_val)[:, 1]

#Calculate the area under the roc curve
auc1 = roc_auc_score(y_val, probs1)

```

Plot receiver operating characteristic curve:


```python
fpr1, tpr1, thresholds1 = roc_curve(y_val, probs1)

plt.style.use('bmh')
plt.figure(figsize = (5, 5))

# Plot the roc curve
plt.plot(fpr1, tpr1, 'b')
plt.plot([0,1], [0,1], linestyle = '--', color = 'r')
plt.xlabel('False Positive Rate \n (100%-specificity)', size = 16)
plt.ylabel('True Positive Rate \n (Sensitivity)', size = 16)
plt.title('Receiver Operating Characteristic Curve, AUC = %0.4f' % auc1, size = 18)
```




    Text(0.5, 1.0, 'Receiver Operating Characteristic Curve, AUC = 0.5812')



![png](https://drive.google.com/uc?export=view&id=1OnwEm6HzTLTp9VP0-E1hzY9t5ruKfOv7)

<font color=red> Recall: there was no correlation among features, consequently, Linear Regression model is performing very bad when looking at `f1 score` and `ROC & AUC` </font>

### 2. Decision Tree


```python
pred2 = DT.predict(x_val)


mat2 = confusion_matrix(y_val, pred2)

plt.figure(figsize = (7,6))
sns.set(font_scale = 1.2)
sns.heatmap(mat2, cbar = True, square = True, annot = True, yticklabels = target_names,
            annot_kws={'size': 15}, xticklabels=target_names, cmap = 'RdPu')

plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.tight_layout()
plt.show()
```

![png](https://drive.google.com/uc?export=view&id=1Dzj8cnnwqutg8KuE4hvlqUysXrkLvBoS)


```python
summ_conf2 = classification_report(y_val, pred2, target_names = target_names)


print('\n------------Decision Trees Classification Report----------------')
print('\n', summ_conf2)
print('--'*35)
```


    ------------Decision Trees Classification Report----------------

                   precision    recall  f1-score   support

       Bad payer       0.90      0.92      0.91       435
      Good payer       0.98      0.98      0.98      2031

       micro avg       0.97      0.97      0.97      2466
       macro avg       0.94      0.95      0.94      2466
    weighted avg       0.97      0.97      0.97      2466

    ----------------------------------------------------------------------


### ROC & AUC - Decision Tree:


```python
probs2 = DT.predict_proba(x_val)[:, 1]

auc2 = roc_auc_score(y_val, probs2)
```

Plot receiver operating characteristic curve for Decision Tree Classifier:


```python
fpr2, tpr2, thresholds2 = roc_curve(y_val, probs2)

plt.style.use('bmh')
plt.figure(figsize = (5, 5))

# Plot the roc curve
plt.plot(fpr2, tpr2, 'b')
plt.plot([0,1], [0,1], linestyle = '--', color = 'r')
plt.xlabel('False Positive Rate \n (100%-specificity)', size = 16)
plt.ylabel('True Positive Rate \n (Sensitivity)', size = 16)
plt.title('Receiver Operating Characteristic Curve, AUC = %0.4f' % auc2, size = 18)
```




    Text(0.5, 1.0, 'Receiver Operating Characteristic Curve, AUC = 0.9484')

![png](https://drive.google.com/uc?export=view&id=129K2o0aBGMyMRmDzbcvrWBTByKwF83r5)

We see that Decision Tree model is performing fairly well when looking at `f1 score` and `ROC & AUC`.

## 3. Random Forest


```python
pred3 = RF.predict(x_val)

mat3 = confusion_matrix(y_val, pred3)

plt.figure(figsize = (7,6))
sns.set(font_scale = 1.2)
sns.heatmap(mat3, cbar = True, square = True, annot = True,yticklabels = target_names,
            annot_kws={'size': 15}, xticklabels=target_names, cmap = 'RdPu')

plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.tight_layout()
plt.show()
```

![png](https://drive.google.com/uc?export=view&id=1RfrSmp9w5T1_Si38ipdfEgCCH-nZxQej)


```python
summ_conf3 = classification_report(y_val, pred3, target_names = target_names)


print('\n------------Random Forest Classification Report----------------')
print('\n', summ_conf3)
print('--'*33)
```


    ------------Random Forest Classification Report----------------

                   precision    recall  f1-score   support

       Bad payer       0.91      0.92      0.92       435
      Good payer       0.98      0.98      0.98      2031

       micro avg       0.97      0.97      0.97      2466
       macro avg       0.95      0.95      0.95      2466
    weighted avg       0.97      0.97      0.97      2466

    ------------------------------------------------------------------


### ROC & AUC - Random Forest:


```python
probs3 = RF.predict_proba(x_val)[:, 1]

auc3 = roc_auc_score(y_val, probs3)
```

Plot receiver operating characteristic curve for Random Forest Classifier:


```python
fpr3, tpr3, thresholds3 = roc_curve(y_val, probs3)

plt.style.use('bmh')
plt.figure(figsize = (5, 5))

# Plot the roc curve
plt.plot(fpr3, tpr3, 'b')
plt.plot([0,1], [0,1], linestyle = '--', color = 'r')
plt.xlabel('False Positive Rate \n (100%-specificity)', size = 16)
plt.ylabel('True Positive Rate \n (Sensitivity)', size = 16)
plt.title('Receiver Operating Characteristic Curve, AUC = %0.4f' % auc3, size = 18)
```




    Text(0.5, 1.0, 'Receiver Operating Characteristic Curve, AUC = 0.9857')

![png](https://drive.google.com/uc?export=view&id=1fxTnCdjiAmLPjEOrfkxaz_DKtwuGrnO6)


We see that Random Forest model is performing pretty good when looking at `f1 score` and `ROC & AUC`.


### Conclusions from Machine Learning Models

From the `f1 score` and the `Area Under the Receiver Operating Characteristic Curve`, `Random Forest` performs the best.  


```python

```