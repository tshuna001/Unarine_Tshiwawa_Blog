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

We are using data from [zindi](https://zindi.africa/competitions/data-science-nigeria-challenge-1-loan-default-prediction/data). For more information about datasets, the reader is  referred to [data](https://zindi.africa/competitions/data-science-nigeria-challenge-1-loan-default-prediction/data). The objective in this work is to create a binary classification model that predicts whether or not an individual is a good payer or not based on several indicators. The target variable is given as `good_bad_flag` and takes on a value of 1 if the client is  a good payer and 0 otherwise.

We will do some data exploratory data analysis, and then focus on learning a model. The second step will involve merging datasets and then impute the missing values. We will then split the dataset into a training and testing set randomly.  Then, we simply predict the most common class in the training data for all observations in the testing dataset



```python
train_demo = pd.read_csv('traindemographics.csv')
train_perf = pd.read_csv('trainperf.csv')
train_prev = pd.read_csv('trainprevloans.zip', compression='zip')

```

Let's view few columns for these datasets:


```python
print('\n--------------------------------Train demo dataset-----------------------------')
display(train_demo.head(5))

print('\n-----------------------------Train performance dataset-------------------------')
display(train_perf.head(5))

print('\n----------------------------Train previous loans dataset-----------------------')
display(train_perf.head(5))
```


    --------------------------------Train demo dataset-----------------------------



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



    -----------------------------Train performance dataset-------------------------



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



    ----------------------------Train previous loans dataset-----------------------



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


### Merging datasets:

Approach:  we shall merge `demographics dataset` and `train performance dataset` using customerid.  Then, the resultant dataset will be merged with `train previous dataset` using customerid.


```python
df = pd.merge(train_demo, train_perf, how = 'inner', on = 'customerid')

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
               'approveddate_y','creationdate_y','closeddate','firstduedate','firstrepaiddate'], axis = 1)
```

### Imputing missing values:


```python
df1.isna().sum()
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
    systemloanid_y                  0
    loannumber_y                    0
    loanamount_y                    0
    totaldue_y                      0
    termdays_y                      0
    dtype: int64



We see that `employment_status_clients` columns contains 1363 nans, so we will drop those rows:


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



Replace binary variables to numerical features in `employment_status_clients`:


```python
df1['employment_status_clients'] = df1['employment_status_clients'].replace({'Permanent':5, 'Self-Employed':4,
                                                                             'Student':3, 'Unemployed':2,
                                                                            'Retired':1, 'Contract':0})
```


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
      <th>systemloanid_y</th>
      <th>loannumber_y</th>
      <th>loanamount_y</th>
      <th>totaldue_y</th>
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
      <td>301929966</td>
      <td>1</td>
      <td>10000.0</td>
      <td>13000.0</td>
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
      <td>301939781</td>
      <td>1</td>
      <td>10000.0</td>
      <td>11500.0</td>
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
      <td>301876300</td>
      <td>3</td>
      <td>10000.0</td>
      <td>13000.0</td>
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
      <td>301953421</td>
      <td>5</td>
      <td>20000.0</td>
      <td>24500.0</td>
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
      <td>301916071</td>
      <td>4</td>
      <td>20000.0</td>
      <td>24500.0</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>


Let's view a pie chart for our target variable:


```python

pie = df1['good_bad_flag'].value_counts().plot(kind = 'pie',explode=(0,0.1), autopct='%1.2f%%',
                                         shadow = True, legend = True, labels = None)
pie
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f44ae69bba8>




![](https://raw.githubusercontent.com/tshuna001/images/master/loan_model_20_1.png?token=AE5UC7254R5MVS3DT27EJ3S55FBFG)


NB: - One important thing to note, the dataset is imbalanced. This is an imbalanced class problem because there are significantly more clients who are good payers than bad payers!

Replacing binary variables to numerical variables:


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


```python

```


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
      <th>systemloanid_y</th>
      <th>loannumber_y</th>
      <th>loanamount_y</th>
      <th>totaldue_y</th>
      <th>termdays_y</th>
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
      <td>1.233000e+04</td>
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
      <td>3.018333e+08</td>
      <td>4.353041</td>
      <td>16938.605028</td>
      <td>20079.213252</td>
      <td>27.172749</td>
      <td>34.631306</td>
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
      <td>9.456062e+04</td>
      <td>3.333520</td>
      <td>9599.644649</td>
      <td>10756.897338</td>
      <td>11.201963</td>
      <td>6.119540</td>
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
      <td>3.016001e+08</td>
      <td>1.000000</td>
      <td>3000.000000</td>
      <td>3900.000000</td>
      <td>15.000000</td>
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
      <td>3.017701e+08</td>
      <td>2.000000</td>
      <td>10000.000000</td>
      <td>11750.000000</td>
      <td>15.000000</td>
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
      <td>3.018465e+08</td>
      <td>3.000000</td>
      <td>10000.000000</td>
      <td>13000.000000</td>
      <td>30.000000</td>
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
      <td>3.019167e+08</td>
      <td>6.000000</td>
      <td>20000.000000</td>
      <td>24500.000000</td>
      <td>30.000000</td>
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
      <td>3.019934e+08</td>
      <td>26.000000</td>
      <td>60000.000000</td>
      <td>68100.000000</td>
      <td>90.000000</td>
      <td>58.000000</td>
    </tr>
  </tbody>
</table>
</div>


We could use a correlation threshold for removing variables. In this case, we will probably want to keep all of the variables and let the model decide which are relevant.

## Correlations

We can calculate correlation values to see how the features are related to the outcome--`good_bad_flag`. Correlation does not of course imply causation, but because we are building a model, the correlated features are likely useful for learning a mapping between the clients information and whether or not they are good payers.


```python
df1.corr()['good_bad_flag'].sort_values(ascending=False)
```




    good_bad_flag                1.000000
    loanamount_x                 0.095799
    totaldue_x                   0.085127
    loanamount_y                 0.059628
    totaldue_y                   0.059166
    age                          0.046062
    loannumber_y                 0.038659
    termdays_y                   0.033126
    employment_status_clients   -0.007333
    termdays_x                  -0.007663
    longitude_gps               -0.011899
    latitude_gps                -0.014939
    systemloanid_y              -0.024939
    Name: good_bad_flag, dtype: float64



# Preprocesing:

- In this step, we shall prepare data for machine learning algorithms.


```python
from sklearn.model_selection import train_test_split


#feature matrix
x = df1.drop(['good_bad_flag','customerid'], axis = 1)

#target
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
      <th>systemloanid_y</th>
      <th>loannumber_y</th>
      <th>loanamount_y</th>
      <th>totaldue_y</th>
      <th>termdays_y</th>
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
      <td>301929966</td>
      <td>1</td>
      <td>10000.0</td>
      <td>13000.0</td>
      <td>30</td>
      <td>33.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3.362850</td>
      <td>6.642485</td>
      <td>5</td>
      <td>10000.0</td>
      <td>11500.0</td>
      <td>15</td>
      <td>301939781</td>
      <td>1</td>
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
      <td>301876300</td>
      <td>3</td>
      <td>10000.0</td>
      <td>13000.0</td>
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
      <td>301953421</td>
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
      <td>301916071</td>
      <td>4</td>
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


#### Data splicing:

We are going to split the data into two sets -  75$\%$ train set and 25$\%$ test set.  This means, 75$\%$ of the data will be set to train and optimize our machine learning model, and the remaining 25$\%$ will be used to test the model.


```python
from sklearn.model_selection import train_test_split


#data splicing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 10, stratify = y)

print(x_train.shape)
print(x_test.shape)
```

    (9247, 12)
    (3083, 12)


## Machine Learning models

Data are not usually presented to the machine learning algorithm in exactly the same raw form as it is found. Usually data are scaled to a specific range in a process called normalization.

We are going to make selected features to the same scale for optimal performance, which is often achieved by transforming the features in the range [0, 1]: standardize features by removing the mean and scaling to unit variance  using `StandardScaler()` class.  This will be done in the pipeline to run multiple processes in the order that they are listed. The purpose and advantage of using pipeline is to assemble several steps that can be cross-validated together while setting different parameters.


```python
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')
#from sklearn.model_selection import GridSearchCV

#Logistic regression
from sklearn.linear_model import LogisticRegression

LR = Pipeline(steps = [('preprocessor', preprocessing.StandardScaler()),
                     ('model', LogisticRegression(random_state = 42))])
LR.fit(x_train, y_train)

print('\n--------------------LogisticRegression Classifier-------------------')
print('Accuraccy in training set: {:.2f}' .format(LR.score(x_train, y_train)))
print('Accuraccy in test set: {:.2f}' .format(LR.score(x_test, y_test)))


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

DT = Pipeline(steps=[('preprocessor', preprocessing.StandardScaler()),
                     ('model', DecisionTreeClassifier())])
DT.fit(x_train, y_train)

print('\n-----------------------Decision Tree Classifier---------------------')
print('Accuraccy in training set: {:.2f}' .format(DT.score(x_train, y_train)))
print('Accuraccy in test set: {:.2f}' .format(DT.score(x_test, y_test)))


#Random forest
from sklearn.ensemble import RandomForestClassifier

#instantiate random forest class
RF = Pipeline(steps=[('preprocessor', preprocessing.StandardScaler()),
                     ('model', RandomForestClassifier(n_estimators = 1000, random_state = 1, n_jobs = -1))])

RF.fit(x_train, y_train)

print('\n-----------------------Random Forest Classifier---------------------')
print('Accuraccy in training set: {:.2f}' .format(RF.score(x_train, y_train)))
print('Accuraccy in test set: {:.2f}' .format(RF.score(x_test, y_test)))


```


    --------------------LogisticRegression Classifier-------------------
    Accuraccy in training set: 0.82
    Accuraccy in test set: 0.82

    -----------------------Decision Tree Classifier---------------------
    Accuraccy in training set: 1.00
    Accuraccy in test set: 0.94

    -----------------------Random Forest Classifier---------------------
    Accuraccy in training set: 1.00
    Accuraccy in test set: 0.94


`We know that accuracy works well on balanced data.  The data is imbalanced, so we cannot use accuracy to quantify model performance. So we need another perfomance measure for imbalanced data.  We shall consider using f1 score metric to quantify the perfomance.`

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
pred1 = LR.predict(x_test)

mat1 = confusion_matrix(y_test, pred1)

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


![](https://raw.githubusercontent.com/tshuna001/images/master/loan_model_41_0.png?token=AE5UC7ZNVXUGJN7QUXYINW255FBNG)



```python
summ_conf1 = classification_report(y_test, pred1, target_names = target_names)


print('\n------------Logistic Regression Classification Report----------------')
print('\n', summ_conf1)
print('--'*33)
```


    ------------Logistic Regression Classification Report----------------

                   precision    recall  f1-score   support

       Bad payer       0.00      0.00      0.00       544
      Good payer       0.82      1.00      0.90      2539

       micro avg       0.82      0.82      0.82      3083
       macro avg       0.41      0.50      0.45      3083
    weighted avg       0.68      0.82      0.74      3083

    ------------------------------------------------------------------


### ROC & AUC - Logistic Regression:


```python
#predicting class probabilities for input x_test
probs1 = LR.predict_proba(x_test)[:, 1]

#Calculate the area under the roc curve
auc1 = roc_auc_score(y_test, probs1)

```

Plot receiver operating characteristic curve:


```python
fpr1, tpr1, thresholds1 = roc_curve(y_test, probs1)

plt.style.use('bmh')
plt.figure(figsize = (5, 5))

# Plot the roc curve
plt.plot(fpr1, tpr1, 'b')
plt.plot([0,1], [0,1], linestyle = '--', color = 'r')
plt.xlabel('False Positive Rate', size = 16)
plt.ylabel('True Positive Rate', size = 16)
plt.title('Receiver Operating Characteristic Curve, AUC = %0.4f' % auc1, size = 18)
```




    Text(0.5, 1.0, 'Receiver Operating Characteristic Curve, AUC = 0.5657')




![](https://raw.githubusercontent.com/tshuna001/images/master/loan_model_46_1.png?token=AE5UC7YQZOSPSUYBIFQ4AD255FBQ6)


### 2. Decision Tree


```python
pred2 = DT.predict(x_test)


mat2 = confusion_matrix(y_test, pred2)

plt.figure(figsize = (7,6))
sns.set(font_scale = 1.2)
sns.heatmap(mat2, cbar = True, square = True, annot = True, yticklabels = target_names,
            annot_kws={'size': 15}, xticklabels=target_names, cmap = 'RdPu')

plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.tight_layout()
plt.show()
```


![](https://raw.githubusercontent.com/tshuna001/images/master/loan_model_48_0.png?token=AE5UC76IPUIKRZDH55AQN3255FBTO)



```python
summ_conf2 = classification_report(y_test, pred2, target_names = target_names)


print('\n------------Logistic Regression Classification Report----------------')
print('\n', summ_conf2)
print('--'*35)
```


    ------------Logistic Regression Classification Report----------------

                   precision    recall  f1-score   support

       Bad payer       0.82      0.84      0.83       544
      Good payer       0.97      0.96      0.96      2539

       micro avg       0.94      0.94      0.94      3083
       macro avg       0.89      0.90      0.90      3083
    weighted avg       0.94      0.94      0.94      3083

    ----------------------------------------------------------------------


### ROC & AUC - Decision Tree:


```python
probs2 = DT.predict_proba(x_test)[:, 1]

auc2 = roc_auc_score(y_test, probs2)
```

Plot receiver operating characteristic curve for Decision Tree Classifier:


```python
fpr2, tpr2, thresholds2 = roc_curve(y_test, probs2)

plt.style.use('bmh')
plt.figure(figsize = (5, 5))

# Plot the roc curve
plt.plot(fpr2, tpr2, 'b')
plt.plot([0,1], [0,1], linestyle = '--', color = 'r')
plt.xlabel('False Positive Rate', size = 16)
plt.ylabel('True Positive Rate', size = 16)
plt.title('Receiver Operating Characteristic Curve, AUC = %0.4f' % auc2, size = 18)
```




    Text(0.5, 1.0, 'Receiver Operating Characteristic Curve, AUC = 0.9015')




![](https://raw.githubusercontent.com/tshuna001/images/master/loan_model_53_1.png?token=AE5UC75IDCILLQU7LHQWK5C55FBWQ)


## 3. Random Forest


```python
pred3 = RF.predict(x_test)

mat3 = confusion_matrix(y_test, pred3)

plt.figure(figsize = (7,6))
sns.set(font_scale = 1.2)
sns.heatmap(mat3, cbar = True, square = True, annot = True,yticklabels = target_names,
            annot_kws={'size': 15}, xticklabels=target_names, cmap = 'RdPu')

plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.tight_layout()
plt.show()
```


![](https://raw.githubusercontent.com/tshuna001/images/master/loan_model_55_0.png?token=AE5UC77PLONH2DY6O33N3D255FBZQ)



```python
summ_conf3 = classification_report(y_test, pred3, target_names = target_names)


print('\n------------Random Forest Classification Report----------------')
print('\n', summ_conf3)
print('--'*33)
```


    ------------Random Forest Classification Report----------------

                   precision    recall  f1-score   support

       Bad payer       0.95      0.67      0.79       544
      Good payer       0.93      0.99      0.96      2539

       micro avg       0.94      0.94      0.94      3083
       macro avg       0.94      0.83      0.88      3083
    weighted avg       0.94      0.94      0.93      3083

    ------------------------------------------------------------------


### ROC & AUC - Random Forest:


```python
probs3 = RF.predict_proba(x_test)[:, 1]

auc3 = roc_auc_score(y_test, probs3)
```

Plot receiver operating characteristic curve for Random Forest Classifier:


```python
fpr3, tpr3, thresholds3 = roc_curve(y_test, probs3)

plt.style.use('bmh')
plt.figure(figsize = (5, 5))

# Plot the roc curve
plt.plot(fpr3, tpr3, 'b')
plt.plot([0,1], [0,1], linestyle = '--', color = 'r')
plt.xlabel('False Positive Rate', size = 16)
plt.ylabel('True Positive Rate', size = 16)
plt.title('Receiver Operating Characteristic Curve, AUC = %0.4f' % auc3, size = 18)
```




    Text(0.5, 1.0, 'Receiver Operating Characteristic Curve, AUC = 0.9745')




![](https://raw.githubusercontent.com/tshuna001/images/master/loan_model_60_1.png?token=AE5UC77C3KG6XVGISUNAAH255FB4C)


### Conclusions from Machine Learning Models

From the f1 score and the Area Under the Receiver Operating Characteristic Curve, the `Random Forest` performs the best.  


```python

```
