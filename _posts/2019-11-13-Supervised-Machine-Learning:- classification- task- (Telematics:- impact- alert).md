---
layout: post
author: "Unarine Tshiwawa"
---

# 1. We shall start by defining the problem:

We are given a dataset containing number of false accidents impacts that are labelled (by time, season, and location) and a test dataset that contain impacts that are not labelled.  In this exercise, we will use a model trained on the dataset containing false accident impacts to classify different kinds of impacts cause in the test dataset.

#### Main point: we are trying to build a model that will predict labels for a new data.

Clearly, we see that this is a machine leaning - classification - type of a problem.  We shall also label each impact by `Year`, `Season`, `Month`, `Week`, `Day`, and `Time`.  An output to this problem is a categorical quantity.


Consider that the only classification of impacts alert will be:

* Speedbumps
* Potholes
* Car Wash and
* Gravel Road

#### Objectives to demonstrate:

    - Multi-classification
    - Feature Engineering
    - Pre-processing  
    - Hyperparameter searching
    - Evaluations
    - Predictions




```python

```


```python
#Import libraries
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns
sns.set()
%matplotlib inline
from IPython.display import display
import pandas_profiling
```


```python
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
```

Read data:


```python
#false impacts (we shall call it a training dataset)
dataset1 = pd.read_csv('impacts_dataset_ch.csv')

#testset
dataset2 = pd.read_csv('impacts_test_dataset_ch.csv')

df1 = dataset1.copy()
df2 = dataset2.copy()
```

Datasets information:

We shall use the info() method to get a quick description of the data


```python
print('\n-----------------------------Training dataset----------------------\n')
display(df1.info())
print('\n-------------------------------Test dataset------------------------\n')
display(df2.info())

```


    -----------------------------Training dataset----------------------

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2975 entries, 0 to 2974
    Data columns (total 5 columns):
    cause              2975 non-null object
    gps_lat            2975 non-null float64
    gps_lon            2975 non-null float64
    alert_timestamp    2975 non-null object
    impact_num         2975 non-null int64
    dtypes: float64(2), int64(1), object(2)
    memory usage: 116.3+ KB



    None



    -------------------------------Test dataset------------------------

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 595 entries, 0 to 594
    Data columns (total 4 columns):
    impact_num         595 non-null int64
    gps_lat            595 non-null float64
    gps_lon            595 non-null float64
    alert_timestamp    595 non-null object
    dtypes: float64(2), int64(1), object(1)
    memory usage: 18.7+ KB



    None


- NB: There are 2975 and 595 instances in both `training` and `test dataset`, respectively.  The size of `training dataset` is small by Machine Learning standards, but it’s perfect to get started.

- All attributes in both datasets are numerical, except that `cause` and `alert_timestamp` fields are categorical.

- We do not have null values in both datasets.


### Take a Quick Look at the Data Structure

Let’s take a look at the top five rows using the DataFrame’s head() method


```python
print('\n------------------------Training data set--------------------')
display(df1.head(5))
print('\n---------------------------Test dataset----------------------')
display(df2.head(5))
```


    ------------------------Training data set--------------------



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
      <th>cause</th>
      <th>gps_lat</th>
      <th>gps_lon</th>
      <th>alert_timestamp</th>
      <th>impact_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Speedbump</td>
      <td>-26.268468</td>
      <td>28.060555</td>
      <td>2018-03-04 16:03:40</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Potholes</td>
      <td>-26.074818</td>
      <td>28.066700</td>
      <td>2018-04-08 10:12:19</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Speedbump</td>
      <td>-26.040653</td>
      <td>28.107715</td>
      <td>2018-05-12 17:23:51</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Speedbump</td>
      <td>-26.194437</td>
      <td>28.037294</td>
      <td>2018-01-26 15:10:58</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Speedbump</td>
      <td>-25.934343</td>
      <td>28.194816</td>
      <td>2018-06-16 06:07:04</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



    ---------------------------Test dataset----------------------



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
      <th>impact_num</th>
      <th>gps_lat</th>
      <th>gps_lon</th>
      <th>alert_timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2975</td>
      <td>-26.105012</td>
      <td>28.154925</td>
      <td>2018-05-22 09:22:06</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2976</td>
      <td>-26.069402</td>
      <td>27.851743</td>
      <td>2018-02-13 08:06:28</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2977</td>
      <td>-26.111047</td>
      <td>28.255036</td>
      <td>2018-05-03 12:59:05</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2978</td>
      <td>-26.167354</td>
      <td>28.080104</td>
      <td>2018-01-22 09:08:51</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2979</td>
      <td>-25.974256</td>
      <td>27.965016</td>
      <td>2018-04-17 09:54:55</td>
    </tr>
  </tbody>
</table>
</div>


Note: - We see that we do not have target variable in the `test dataset`.  In other words, we ony have matrix feature ready to be classified.



### Imputing Null/Nan values (In both datasets)


```python

print('\n------------------------Training data set--------------------')
display((df1.isnull().sum()/df1.shape[0])*100)
print('\n---------------------------Test dataset----------------------')
display((df2.isnull().sum()/df2.shape[0])*100)
```


    ------------------------Training data set--------------------



    cause              0.0
    gps_lat            0.0
    gps_lon            0.0
    alert_timestamp    0.0
    impact_num         0.0
    dtype: float64



    ---------------------------Test dataset----------------------



    impact_num         0.0
    gps_lat            0.0
    gps_lon            0.0
    alert_timestamp    0.0
    dtype: float64


All datasets have no null/nan values! We are good to proceed :)

### Basic statistics:

Summary of each numerical attribute in both datasets.


```python
print('\n------------------------Training data set--------------------')
display(df1.describe())
print('\n---------------------------Test dataset----------------------')
display(df2.describe())
```


    ------------------------Training data set--------------------



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
      <th>gps_lat</th>
      <th>gps_lon</th>
      <th>impact_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2975.000000</td>
      <td>2975.000000</td>
      <td>2975.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-26.112202</td>
      <td>28.050153</td>
      <td>1487.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.085944</td>
      <td>0.102115</td>
      <td>858.952851</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-26.575409</td>
      <td>27.730628</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-26.159565</td>
      <td>27.991750</td>
      <td>743.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-26.114688</td>
      <td>28.046570</td>
      <td>1487.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-26.055884</td>
      <td>28.115476</td>
      <td>2230.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>-25.907032</td>
      <td>28.291150</td>
      <td>2974.000000</td>
    </tr>
  </tbody>
</table>
</div>



    ---------------------------Test dataset----------------------



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
      <th>impact_num</th>
      <th>gps_lat</th>
      <th>gps_lon</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>595.000000</td>
      <td>595.000000</td>
      <td>595.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3272.001681</td>
      <td>-26.116536</td>
      <td>28.047017</td>
    </tr>
    <tr>
      <th>std</th>
      <td>171.908895</td>
      <td>0.096329</td>
      <td>0.104359</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2975.000000</td>
      <td>-26.575198</td>
      <td>27.788814</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3123.500000</td>
      <td>-26.160641</td>
      <td>27.991177</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3272.000000</td>
      <td>-26.116125</td>
      <td>28.046705</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3420.500000</td>
      <td>-26.056709</td>
      <td>28.109730</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3570.000000</td>
      <td>-25.909180</td>
      <td>28.287567</td>
    </tr>
  </tbody>
</table>
</div>


The count, mean, min, and max rows are self-explanatory.  Note that the null values are ignored, but we do not have to worry about that in this case -- both datasets have no null values. The std row shows the standard deviation, which measures how dispersed the values are. The 25%, 50%, and 75% rows show the corresponding percentiles: a percentile indicates the value below which a given percentage of observations in a group of observations falls

## Let's first consider `Training dataset`


```python

```

# 1. Exploratory Data Analysis:

A histogram for each numerical attribute (note that `impact_num` is a redundant feature):


```python
df1.drop('impact_num', axis = 1).hist(bins=50, figsize=(15,5))
plt.show()
```

![png](https://drive.google.com/uc?export=view&id=1iPTt6sPN1u-8An0gE-6k2Ujy1ZZ8H31x)


We see that demographics by location are the only numerical features to consider.

Let's plot categorical features:


```python
plt.figure(figsize = (15,5))
plt.subplot(1, 2, 1)
df1['cause'].value_counts().plot(kind = 'bar', title = 'Number of false impacts per cause', rot = 0)

plt.subplot(1, 2, 2)
df1['cause'].value_counts().plot(kind = 'pie',
                                 explode=(0,0,0,0.1), autopct='%1.2f%%',
                                 shadow=True,
                                 title = 'Number of false impacts per cause', labels = None, legend = True)

plt.show()
```

![png](https://drive.google.com/uc?export=view&id=1ify_Oa1DNDNOBrH4iMcyswYv01VnmbQ2)


For each accident impact, we want to know how many accident happened per season, month, week, day and hour in our dataset within stipulated time (2018-01-01 to 2018-08-13).  In general, there are higher accidents contribution from speedbump, potholes, car wash, and gravel road impact, respectively.


```python

```

### Time Series & Feature engineering:

Let's convert date (`alert_timestamp`) to datetime object so that we can easily extract some useful information from it, such as `time`, `day` of the week, and `month`.




```python
df1.alert_timestamp.describe()
```

    count                    2975
    unique                   2975
    top       2018-03-25 19:19:20
    freq                        1
    first     2018-01-01 19:44:54
    last      2018-08-13 11:49:50
    Name: alert_timestamp, dtype: object



Clearly, train dataset samples does not cover the entire year!

We would like to know how many accidents happens per hour and month for each impact cause in dataset provided.


**Approach**: We shall define a function that will convert the feature column `alert_timestamp` into new feature columns comprises of `Hour`, `Day`, `Month`, `Week`, `Season` and `Year` in both training and test set.

If we considering that all routes are in South Africa, Southern Hemisphere, we can refer to the standard season cycles and create a new season feature with values of `Spring`, `Summer`, `Autumn`, and `Winter`. Pandas provides easy-to-use functions to extract date-related features.


```python
def date_property(df):

    "extract all date properties from a datetime datatype by hour, month, quarter, year, day and week"

    df['alert_timestamp'] = pd.to_datetime(df['alert_timestamp'], infer_datetime_format=True)


    H = [] #Hours

    for i, j in enumerate(df.alert_timestamp):


        k = j.hour #extracting hours
        if k == 0:
            H.append(24) #24 hours
        else:
            H.append(k)

    df['Hour'] = H

    df["Month"] = df["alert_timestamp"].dt.month
    df["Quarter"] = df["alert_timestamp"].dt.quarter
    df["Year"] = df["alert_timestamp"].dt.year
    df["Day"] = df["alert_timestamp"].dt.day
    df["Week"] = df["alert_timestamp"].dt.week


    df["Season"] = np.where(df["Month"].isin([3,4,5]),"Autumn",
                np.where(df["Month"].isin([6,7,8]), "Winter",
                np.where(df["Month"].isin([9,10,11]),"Spring",
                np.where(df["Month"].isin([12,1,2]), "Summer","None"))))

    df_1 = df.copy()


    '''mapping months from jan -> Aug'''

    df_1.Month = df_1.Month.map({1:'January', 2:'February', 3:'March', 4:'April', 5:'May', 6:'June',
                                 7:'July', 8:'August'})
    df_1.Month = pd.Categorical(df_1.Month, ['January', 'February', 'March', 'April', 'May', 'June',
                                             'July', 'August'])

    '''extracting days of the week'''

    df_1.Day = [j.day_name()[0:3] for j in df_1['alert_timestamp']]
    df_1.Day = pd.Categorical(df_1.Day, ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])


    '''dividing a day into four parts.'''

    df_1.Hour = pd.cut(df_1.Hour, bins = [0, 6, 12, 18, 24], labels = ['Early morning \n (00h00 - 06h00)',
                                                                 'Morning \n (06h00 - 12h00)',
                                                                'Afternoon \n (12h00 - 18h00)',
                                                                'Evening \n (18h00 - 24h00)'])

    return df, df_1





```

Let's view few columns in both new training and test set with new feature columns comprises of "Month" and "Hour":


```python
train_data = date_property(df1)[1]
train_data = train_data.drop('impact_num', axis = 1)

test_data = date_property(df2)[1]
test_data = test_data.drop('impact_num', axis = 1)

print('\n----------------------Train set------------------')
display(train_data.head(5))

print('\n--------------------Test set-------------------')
display(test_data.head(5))
print('-------------------------------------------------')
```


    ----------------------Train set------------------



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
      <th>cause</th>
      <th>gps_lat</th>
      <th>gps_lon</th>
      <th>alert_timestamp</th>
      <th>Hour</th>
      <th>Month</th>
      <th>Quarter</th>
      <th>Year</th>
      <th>Day</th>
      <th>Week</th>
      <th>Season</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Speedbump</td>
      <td>-26.268468</td>
      <td>28.060555</td>
      <td>2018-03-04 16:03:40</td>
      <td>Afternoon \n (12h00 - 18h00)</td>
      <td>March</td>
      <td>1</td>
      <td>2018</td>
      <td>Sun</td>
      <td>9</td>
      <td>Autumn</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Potholes</td>
      <td>-26.074818</td>
      <td>28.066700</td>
      <td>2018-04-08 10:12:19</td>
      <td>Morning \n (06h00 - 12h00)</td>
      <td>April</td>
      <td>2</td>
      <td>2018</td>
      <td>Sun</td>
      <td>14</td>
      <td>Autumn</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Speedbump</td>
      <td>-26.040653</td>
      <td>28.107715</td>
      <td>2018-05-12 17:23:51</td>
      <td>Afternoon \n (12h00 - 18h00)</td>
      <td>May</td>
      <td>2</td>
      <td>2018</td>
      <td>Sat</td>
      <td>19</td>
      <td>Autumn</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Speedbump</td>
      <td>-26.194437</td>
      <td>28.037294</td>
      <td>2018-01-26 15:10:58</td>
      <td>Afternoon \n (12h00 - 18h00)</td>
      <td>January</td>
      <td>1</td>
      <td>2018</td>
      <td>Fri</td>
      <td>4</td>
      <td>Summer</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Speedbump</td>
      <td>-25.934343</td>
      <td>28.194816</td>
      <td>2018-06-16 06:07:04</td>
      <td>Early morning \n (00h00 - 06h00)</td>
      <td>June</td>
      <td>2</td>
      <td>2018</td>
      <td>Sat</td>
      <td>24</td>
      <td>Winter</td>
    </tr>
  </tbody>
</table>
</div>



    --------------------Test set-------------------



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
      <th>gps_lat</th>
      <th>gps_lon</th>
      <th>alert_timestamp</th>
      <th>Hour</th>
      <th>Month</th>
      <th>Quarter</th>
      <th>Year</th>
      <th>Day</th>
      <th>Week</th>
      <th>Season</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-26.105012</td>
      <td>28.154925</td>
      <td>2018-05-22 09:22:06</td>
      <td>Morning \n (06h00 - 12h00)</td>
      <td>May</td>
      <td>2</td>
      <td>2018</td>
      <td>Tue</td>
      <td>21</td>
      <td>Autumn</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-26.069402</td>
      <td>27.851743</td>
      <td>2018-02-13 08:06:28</td>
      <td>Morning \n (06h00 - 12h00)</td>
      <td>February</td>
      <td>1</td>
      <td>2018</td>
      <td>Tue</td>
      <td>7</td>
      <td>Summer</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-26.111047</td>
      <td>28.255036</td>
      <td>2018-05-03 12:59:05</td>
      <td>Morning \n (06h00 - 12h00)</td>
      <td>May</td>
      <td>2</td>
      <td>2018</td>
      <td>Thu</td>
      <td>18</td>
      <td>Autumn</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-26.167354</td>
      <td>28.080104</td>
      <td>2018-01-22 09:08:51</td>
      <td>Morning \n (06h00 - 12h00)</td>
      <td>January</td>
      <td>1</td>
      <td>2018</td>
      <td>Mon</td>
      <td>4</td>
      <td>Summer</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-25.974256</td>
      <td>27.965016</td>
      <td>2018-04-17 09:54:55</td>
      <td>Morning \n (06h00 - 12h00)</td>
      <td>April</td>
      <td>2</td>
      <td>2018</td>
      <td>Tue</td>
      <td>16</td>
      <td>Autumn</td>
    </tr>
  </tbody>
</table>
</div>


    -------------------------------------------------


Now, we shall focus on the training set for further exploratory data analysis.


```python

```


```python
display(train_data[["alert_timestamp","Year","Month","Day", "Hour","Week","Quarter","Season"]].head(10))
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
      <th>alert_timestamp</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
      <th>Hour</th>
      <th>Week</th>
      <th>Quarter</th>
      <th>Season</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-03-04 16:03:40</td>
      <td>2018</td>
      <td>March</td>
      <td>Sun</td>
      <td>Afternoon \n (12h00 - 18h00)</td>
      <td>9</td>
      <td>1</td>
      <td>Autumn</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-04-08 10:12:19</td>
      <td>2018</td>
      <td>April</td>
      <td>Sun</td>
      <td>Morning \n (06h00 - 12h00)</td>
      <td>14</td>
      <td>2</td>
      <td>Autumn</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-05-12 17:23:51</td>
      <td>2018</td>
      <td>May</td>
      <td>Sat</td>
      <td>Afternoon \n (12h00 - 18h00)</td>
      <td>19</td>
      <td>2</td>
      <td>Autumn</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-01-26 15:10:58</td>
      <td>2018</td>
      <td>January</td>
      <td>Fri</td>
      <td>Afternoon \n (12h00 - 18h00)</td>
      <td>4</td>
      <td>1</td>
      <td>Summer</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-06-16 06:07:04</td>
      <td>2018</td>
      <td>June</td>
      <td>Sat</td>
      <td>Early morning \n (00h00 - 06h00)</td>
      <td>24</td>
      <td>2</td>
      <td>Winter</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2018-01-20 20:43:40</td>
      <td>2018</td>
      <td>January</td>
      <td>Sat</td>
      <td>Evening \n (18h00 - 24h00)</td>
      <td>3</td>
      <td>1</td>
      <td>Summer</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2018-01-30 09:22:03</td>
      <td>2018</td>
      <td>January</td>
      <td>Tue</td>
      <td>Morning \n (06h00 - 12h00)</td>
      <td>5</td>
      <td>1</td>
      <td>Summer</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2018-03-27 15:11:48</td>
      <td>2018</td>
      <td>March</td>
      <td>Tue</td>
      <td>Afternoon \n (12h00 - 18h00)</td>
      <td>13</td>
      <td>1</td>
      <td>Autumn</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2018-05-10 07:53:08</td>
      <td>2018</td>
      <td>May</td>
      <td>Thu</td>
      <td>Morning \n (06h00 - 12h00)</td>
      <td>19</td>
      <td>2</td>
      <td>Autumn</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2018-02-23 17:03:00</td>
      <td>2018</td>
      <td>February</td>
      <td>Fri</td>
      <td>Afternoon \n (12h00 - 18h00)</td>
      <td>8</td>
      <td>1</td>
      <td>Summer</td>
    </tr>
  </tbody>
</table>
</div>


### Let's generate count plot for each date property:


```python
plt.figure(figsize = (15, 13))
plt.subplot(4,2,1)
sns.countplot(x = 'Season', hue = 'cause', data = train_data)
plt.xlabel('Season')
plt.ylabel('Number of accidents')
plt.tight_layout()


plt.subplot(4,2,2)
sns.countplot(x = 'Month', hue = 'cause', data = train_data)
plt.xlabel('Month')
plt.ylabel('Number of accidents')
plt.tight_layout()


plt.subplot(4,2,3)
sns.countplot(x = 'Day', hue = 'cause', data = train_data)
plt.xlabel('Day')
plt.ylabel('Number of accidents')
plt.tight_layout()


plt.subplot(4,2,4)
sns.countplot(x = 'Hour', hue = 'cause', data = train_data)
plt.xlabel('Hour')
plt.ylabel('Number of accidents')
plt.tight_layout()

plt.subplot(4,2,5)
sns.countplot(x = 'Week', hue = 'cause', data = train_data)
plt.xlabel('Week')
plt.ylabel('Number of accidents')
plt.tight_layout()


plt.subplot(4,2,6)
sns.countplot(x = 'Quarter', hue = 'cause', data = train_data)
plt.xlabel('Quarter')
plt.ylabel('Number of accidents')
plt.tight_layout()
```

![png](https://drive.google.com/uc?export=view&id=1fBXwOkUJdsez5Ssgzu0x9GJ1b6UZ3Hx7)


```python

```

### Geographical location:

We shall plot the locations of the impacts on a map.  folium will be used to view locations

`Steps to consider:`

- defining the center of the map from GPS coodinates.
- showing different impact causes with different colors on the map

**Warning**: color coding for different accident impacts are as follows:

- Red = Speedbump
- Green = Potholes
- Yellow = Car Wash
- Blue = Gravel road



```python
import folium as fl

#base map
gps_map = fl.Map(location=[(train_data['gps_lat']).mean(), (train_data['gps_lon']).mean()],
           tiles = 'Stamen Toner', zoom_start = 12)

#defining calors for different impacts
colors = {'Speedbump': 'red', 'Potholes': 'green', 'Car Wash': 'yellow', 'Gravel road': 'blue'}


for i, j, k in zip(train_data['gps_lat'], train_data['gps_lon'], train_data['cause']):

    fl.CircleMarker(radius = 3, location = [i, j],
                    color = 'g',
                    fill_color = colors[k], fill = True,
                    fill_opacity = 1
                   ).add_to(gps_map)
gps_map
```







Setting the alpha option to 0.3 makes it much easier to visualize the places where there is a high density of data points


```python
train_data.plot(kind="scatter", x="gps_lon", y="gps_lat",figsize=(10,7) ,alpha=0.3)
```

    'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.





    <matplotlib.axes._subplots.AxesSubplot at 0x7fc6fc20f518>




![png](https://drive.google.com/uc?export=view&id=1oMH8XvnX1Or7joiURGEt0RbFattCCJQe)

We see that different accident impacts are sparsely dispersed across Gauteng region

### Overview for `train dataset`:


```python
profile = pandas_profiling.ProfileReport(df1)

profile.to_file(output_file="output.html")
```

#### Let's go deeper:

- We will need a model to help classify different impact cause in the dataset.  We need to instintiate an algorithm which will learn patterns from dataset in order to create a reasonable model.  This model will be used to predict target quantities in new datasets (we shall assume that impact cause in the test dataset also occur in the training dataset).

- In this problem, our target variables are known (speedbumps, potholes, gravel road, and car wash - related accident impacts), so supervise machine learning type will be a good fit.

- Now, we have a test dataset which contains only feature matrix.  We need to train machine learning classifier and create a model based on the dataset previously explored dataset (called **train** dataset then) and use the model obtained therefore to predict target features ( which are **impact causes**) in the **test dataset**.  

# 2. Preprocessing

We are going to generate a feature matrix and target vector.  We shall remove redundant columns: columns which will not make the difference during training phase.

Note: Our target variable is a string, so we need to perform one-hot encouding technique to the target variable.



#### Feature matrix:


```python
#feature matrix
train_data2 = date_property(df1)[0]

#droping redundant columns
train_data2 = train_data2.drop(['cause', 'alert_timestamp', 'impact_num', 'Season', 'Year'], axis = 1)

x = train_data2

print('\n-----------------feature matrix----------------')
display(x.head())
print('\n----------------------------------------------')
```


    -----------------feature matrix----------------



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
      <th>gps_lat</th>
      <th>gps_lon</th>
      <th>Hour</th>
      <th>Month</th>
      <th>Quarter</th>
      <th>Day</th>
      <th>Week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-26.268468</td>
      <td>28.060555</td>
      <td>16</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-26.074818</td>
      <td>28.066700</td>
      <td>10</td>
      <td>4</td>
      <td>2</td>
      <td>8</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-26.040653</td>
      <td>28.107715</td>
      <td>17</td>
      <td>5</td>
      <td>2</td>
      <td>12</td>
      <td>19</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-26.194437</td>
      <td>28.037294</td>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>26</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-25.934343</td>
      <td>28.194816</td>
      <td>6</td>
      <td>6</td>
      <td>2</td>
      <td>16</td>
      <td>24</td>
    </tr>
  </tbody>
</table>
</div>



    ----------------------------------------------


#### Target variables:

Our target variables are categorical.  This means that categorical data must be converted to a numerical form using onehot encoding technique.


```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

#target variables
target = train_data['cause']

#change target from row vector to column vector
target = target[:, np.newaxis]

#apply onehot encoder to column vector
target_encoded = OneHotEncoder(handle_unknown='ignore', sparse=False).fit_transform(target)


order = ['Car wash', 'Gravel road', 'Potholes', 'Speedbump']

#create a target vector
y = pd.DataFrame(target_encoded)

y.columns = order

print('----------------Target vectors/variables----------')

display(y.head(10))
print('----------------------------------------------------')
```

    ----------------Target vectors/variables----------



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
      <th>Car wash</th>
      <th>Gravel road</th>
      <th>Potholes</th>
      <th>Speedbump</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


    ----------------------------------------------------


#### Data splicing - data partitioning:  

We are going to split the data into two sets -  80$\%$ train set and 20$\%$ validation set.  This means, 80% of the data will be set to train and optimize our machine learning model, and the remaining 25$\%$ will be used to validate the model.




```python
from sklearn.model_selection import train_test_split

#data splicing
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.20, random_state = 42, stratify = y)

print(x_train.shape)
print(x_val.shape)

```

    (2380, 7)
    (595, 7)



```python

```

Let's confirm again if data partition criteria was archieved!


```python
for dataset in [y_train, y_val]:


    print(round((len(dataset)/len(y))*100, 2), '%')

print('>>>>>>>>>>>> Done!!<<<<<<<<<<<<<<')

```

    80.0 %
    20.0 %
    >>>>>>>>>>>> Done!!<<<<<<<<<<<<<<


#### Getting data into shape:

Data are not usually presented to the machine learning algorithm in exactly the same raw form as it is found. Usually data are scaled to a specific range in a process called normalization.  However `Decision Tree` and `Random Forest` are immune to feature scaling.

We are going to make selected features to the same scale for optimal performance, which is often achieved by transforming the features in the range [0, 1]: standardize features by removing the mean and scaling to unit variance.  This will be done in the pipeline to run multiple processes in the order that they are listed. The purpose of the pipeline is to assemble several steps that can be cross-validated together while setting different parameters.


```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)

x_train_std = scaler.transform(x_train)
x_val_std = scaler.transform(x_val)
```

# 3. Machine Learning Models

Now, the input and output variables are ready for training. This is the part where different class will be instantiated and a model which perfom better in both training and testing set will be used for further predictions.


```python
from sklearn import preprocessing
from sklearn.pipeline import Pipeline


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier(random_state = 42)
DT.fit(x_train, y_train)

print('-----------------------Decision Tree Classifier---------------------\n')
print('Accuraccy in training set: {:.5f}' .format(DT.score(x_train, y_train)))
print('Accuraccy in val set: {:.5f}' .format(DT.score(x_val, y_val)))


#Random forest
from sklearn.ensemble import RandomForestClassifier

#instantiate random forest class
RF = RandomForestClassifier(n_estimators = 100, random_state = 42)

RF.fit(x_train, y_train)

print('\n-----------------------Random Forest Classifier---------------------\n')
print('Accuraccy in training set: {:.5f}' .format(RF.score(x_train, y_train)))
print('Accuraccy in val set: {:.5f}' .format(RF.score(x_val, y_val)))


from sklearn.neighbors import KNeighborsClassifier
KNN = Pipeline(steps=[('preprocessor', preprocessing.StandardScaler()),
                     ('model', KNeighborsClassifier(n_neighbors = 4))])
KNN.fit(x_train, y_train)

print('\n---------------------K-Nearest Neighbor Classifier------------------\n')
print('Accuraccy in training set: {:.5f}' .format(KNN.score(x_train, y_train)))
print('Accuraccy in val set: {:.5f}' .format(KNN.score(x_val, y_val)))


print('>>>>>>>>>>>>>>>>>>>>>>>>>>>Done!!!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

```

    -----------------------Decision Tree Classifier---------------------

    Accuraccy in training set: 1.00000
    Accuraccy in val set: 0.96303

    -----------------------Random Forest Classifier---------------------

    Accuraccy in training set: 1.00000
    Accuraccy in val set: 0.97815

    ---------------------K-Nearest Neighbor Classifier------------------

    Accuraccy in training set: 0.91176
    Accuraccy in val set: 0.81513
    >>>>>>>>>>>>>>>>>>>>>>>>>>>Done!!!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


`We know that accuracy works well on balanced data.  The data is imbalanced, so we cannot use accuracy to quantify model performance. So we need another perfomance measure for imbalanced data.  We shall consider using f1 score metric to quantify the perfomance.`

# 4. Model evaluation

We shall now show a confusion matrix showing the frequency of misclassifications by our
classifier.  


```python
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
```

### Cofussion Matrix


```python

```


```python
target_names = order

pred1 = DT.predict(x_val)

mat1 = confusion_matrix(y_val.values.argmax(axis=1), pred1.argmax(axis=1))


plt.figure(figsize = (15,12))
plt.subplot(2, 2, 1)
sns.set(font_scale = 1.2)
sns.heatmap(mat1, cbar = True, square = True, annot = True, yticklabels = target_names,
            annot_kws={'size': 15}, xticklabels = target_names, cmap = 'RdPu')
plt.title('Decision Tree Classifier')
plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.tight_layout()


pred2 = RF.predict(x_val)

mat2 = confusion_matrix(y_val.values.argmax(axis=1), pred2.argmax(axis=1))

plt.subplot(2, 2, 2)
sns.set(font_scale = 1.2)
sns.heatmap(mat2, cbar = True, square = True, annot = True, yticklabels = target_names,
            annot_kws={'size': 15}, xticklabels = target_names, cmap = 'RdPu')
plt.title('Random Forest Classifier')
plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.tight_layout()

pred3 = KNN.predict(x_val)

mat3 = confusion_matrix(y_val.values.argmax(axis=1), pred3.argmax(axis=1))

plt.subplot(2, 2, 3)
sns.set(font_scale = 1.2)
sns.heatmap(mat3, cbar = True, square = True, annot = True, yticklabels = target_names,
            annot_kws={'size': 15}, xticklabels = target_names, cmap = 'RdPu')
plt.title('K-Nearest Neighbor Classifier')
plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.tight_layout()



```


![png](https://drive.google.com/uc?export=view&id=1E5y4tVShoeAqP3ri_dWFu4dGPsUjZA1J)

### Classification Report (Precision, Recall & F1 Score)


```python

```


```python
summ_conf1 = classification_report(y_val.values.argmax(axis=1), pred1.argmax(axis=1), target_names = target_names)

print('\n----------------------Decision Tree Classifier-----------------------')
print('\n', summ_conf1, '\n')


summ_conf2 = classification_report(y_val.values.argmax(axis=1), pred2.argmax(axis=1), target_names = target_names)


print('\n----------------------Random Forest Classifier-----------------------')
print('\n', summ_conf2, '\n')


summ_conf3 = classification_report(y_val.values.argmax(axis=1), pred3.argmax(axis=1), target_names = target_names)


print('\n-------------------k-Nearest Neighbor Classifier-----------------------')
print('\n', summ_conf3, '\n')

```


    ----------------------Decision Tree Classifier-----------------------

                   precision    recall  f1-score   support

        Car wash       0.91      0.98      0.95       130
     Gravel road       0.91      0.91      0.91        11
        Potholes       0.98      0.97      0.97       175
       Speedbump       0.98      0.95      0.97       279

       micro avg       0.96      0.96      0.96       595
       macro avg       0.95      0.95      0.95       595
    weighted avg       0.96      0.96      0.96       595



    ----------------------Random Forest Classifier-----------------------

                   precision    recall  f1-score   support

        Car wash       0.94      1.00      0.97       130
     Gravel road       1.00      1.00      1.00        11
        Potholes       1.00      0.97      0.98       175
       Speedbump       0.99      0.99      0.99       279

       micro avg       0.98      0.98      0.98       595
       macro avg       0.98      0.99      0.99       595
    weighted avg       0.98      0.98      0.98       595



    -------------------k-Nearest Neighbor Classifier-----------------------

                   precision    recall  f1-score   support

        Car wash       0.68      0.86      0.76       130
     Gravel road       1.00      0.73      0.84        11
        Potholes       0.92      0.82      0.86       175
       Speedbump       0.92      0.88      0.90       279

       micro avg       0.85      0.85      0.85       595
       macro avg       0.88      0.82      0.84       595
    weighted avg       0.87      0.85      0.86       595





```python

```

## 5. Fine-Tune Models


```python
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score

```

#### Decision Tree Classifier


```python

params1 = {"max_depth": [3, 32, None], "min_samples_leaf": [np.random.randint(1,15)],
               "criterion": ["gini","entropy"], "max_features": [2, 4, 6] }


DT_search = RandomizedSearchCV(DT, param_distributions=params1, random_state=0,
                               verbose=1, cv = 3, return_train_score=True)
DT_search.fit(x_val, y_val)

print('\n-----------------------Decision Tree Classifier---------------------\n')


```

    Fitting 3 folds for each of 10 candidates, totalling 30 fits


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.



    -----------------------Decision Tree Classifier---------------------



    [Parallel(n_jobs=1)]: Done  30 out of  30 | elapsed:    0.9s finished
    /home/unarine/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_search.py:841: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)


#### Random Forest Classifier


```python

#parameter settings
params2 = {"max_depth": [3, 32, None],'n_estimators': [3, 10], 'max_features': [2, 4, 6, 8],
           'bootstrap': [True, False], 'n_estimators': [3, 10, 15], 'max_features': [2, 3, 4],
           "criterion": ["gini","entropy"]}


RF_search = RandomizedSearchCV(RF, param_distributions=params2, random_state=0,
                               cv = 3, verbose = 1, n_jobs = 1, return_train_score=True)

RF_search.fit(x_val, y_val)

print('\n-----------------------Random Forest Classifier---------------------\n')


accuracy_score(y_val, RF_search.predict(x_val))
```

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.


    Fitting 3 folds for each of 10 candidates, totalling 30 fits


    [Parallel(n_jobs=1)]: Done  30 out of  30 | elapsed:    1.4s finished
    /home/unarine/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_search.py:841: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)



    -----------------------Random Forest Classifier---------------------






    1.0



#### k-Nearest Neighbor Classifier


```python
params3 = {'weights':('uniform', 'distance'), 'n_neighbors':[3,4,5,6]}

KNN = KNeighborsClassifier(n_neighbors = 4)

KNN_search = RandomizedSearchCV(KNN, param_distributions=params3, random_state=42,
                                    cv = 5, verbose=1, return_train_score=True)

KNN_search.fit(x_val_std, y_val)


print('\n-----------------------K-Nearest Neighbor Classifier---------------------\n')

accuracy_score(y_val, KNN_search.predict(x_val_std))

```

    Fitting 5 folds for each of 8 candidates, totalling 40 fits


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.



    -----------------------K-Nearest Neighbor Classifier---------------------



    [Parallel(n_jobs=1)]: Done  40 out of  40 | elapsed:    4.6s finished





    1.0




```python

```


```python
target_names = order

pred11 = DT_search.predict(x_val)

mat11 = confusion_matrix(y_val.values.argmax(axis=1), pred11.argmax(axis=1))


plt.figure(figsize = (15,12))
plt.subplot(2, 2, 1)
sns.set(font_scale = 1.2)
sns.heatmap(mat11, cbar = True, square = True, annot = True, yticklabels = target_names,
            annot_kws={'size': 15}, xticklabels = target_names, cmap = 'RdPu')
plt.title('Decision Tree Classifier')
plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.tight_layout()


pred22 = RF_search.predict(x_val)

mat22 = confusion_matrix(y_val.values.argmax(axis=1), pred22.argmax(axis=1))

plt.subplot(2, 2, 2)
sns.set(font_scale = 1.2)
sns.heatmap(mat22, cbar = True, square = True, annot = True, yticklabels = target_names,
            annot_kws={'size': 15}, xticklabels = target_names, cmap = 'RdPu')
plt.title('Random Forest Classifier')
plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.tight_layout()

pred33 = KNN_search.predict(x_val)

mat33 = confusion_matrix(y_val.values.argmax(axis=1), pred33.argmax(axis=1))

plt.subplot(2, 2, 3)
sns.set(font_scale = 1.2)
sns.heatmap(mat33, cbar = True, square = True, annot = True, yticklabels = target_names,
            annot_kws={'size': 15}, xticklabels = target_names, cmap = 'RdPu')
plt.title('K-Nearest Neighbor Classifier')
plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.tight_layout()



```

![png](https://drive.google.com/uc?export=view&id=1mCI_S2zJGQq8mEaefqcd8ewHs7m17aFh)

#### Classification Report (After parameter tuning)


```python
summ_conf11 = classification_report(y_val.values.argmax(axis=1), pred11.argmax(axis=1), target_names = target_names)

print('\n----------------------Decision Tree Classifier-----------------------')
print('\n', summ_conf11, '\n')


summ_conf22 = classification_report(y_val.values.argmax(axis=1), pred22.argmax(axis=1), target_names = target_names)


print('\n----------------------Random Forest Classifier-----------------------')
print('\n', summ_conf22, '\n')


summ_conf33 = classification_report(y_val.values.argmax(axis=1), pred33.argmax(axis=1), target_names = target_names)


print('\n-------------------k-Nearest Neighbor Classifier-----------------------')
print('\n', summ_conf33, '\n')
```


    ----------------------Decision Tree Classifier-----------------------

                   precision    recall  f1-score   support

        Car wash       0.45      0.78      0.57       130
     Gravel road       0.00      0.00      0.00        11
        Potholes       0.77      0.58      0.66       175
       Speedbump       0.75      0.65      0.70       279

       micro avg       0.64      0.64      0.64       595
       macro avg       0.49      0.50      0.48       595
    weighted avg       0.68      0.64      0.65       595



    ----------------------Random Forest Classifier-----------------------

                   precision    recall  f1-score   support

        Car wash       1.00      1.00      1.00       130
     Gravel road       1.00      1.00      1.00        11
        Potholes       1.00      1.00      1.00       175
       Speedbump       1.00      1.00      1.00       279

       micro avg       1.00      1.00      1.00       595
       macro avg       1.00      1.00      1.00       595
    weighted avg       1.00      1.00      1.00       595



    -------------------k-Nearest Neighbor Classifier-----------------------

                   precision    recall  f1-score   support

        Car wash       0.20      0.39      0.26       130
     Gravel road       0.17      0.36      0.24        11
        Potholes       0.33      0.01      0.01       175
       Speedbump       0.46      0.51      0.48       279

       micro avg       0.33      0.33      0.33       595
       macro avg       0.29      0.32      0.25       595
    weighted avg       0.36      0.33      0.29       595




Clearly from Confusion Matrix and Classification Report, we see that `k-Nearest Neighbor` classifier performed poorly before and after parameter tuning.  Decision Tree also performed poorly towards `Gravel road` accident impact -- none was classified.  So we saw a major and noticeable improvements in `Random Forest` where all accident impacts were classified correctly after parameter tuning.



```python

```

### Conclusions from Machine Learning Models:

From the f1 score, the `Random Forest` performs the best! :)  

# 6. Prediction
### Now, we will use the model to predict targets in the new dataset (`test set`)


```python

```


```python
test_data2 = date_property(df2)[0]
test_data2 = test_data2.drop(['impact_num','alert_timestamp','Year', 'Season'], axis = 1)

#predictions in the new dataset
pred_test = RF.predict(test_data2)

#new data frame with column features label
New_df = pd.DataFrame(data = pred_test, columns = order)

#reverse onehot encoding to actual values
New_df['cause'] = (New_df.iloc[:, :] == 1).idxmax(1)


#different unique causes predicted
print('\n---------------------Predicted unique causes------------------')
display(New_df.cause.unique())
print('\n--------------------------------End---------------------------')

```


    ---------------------Predicted unique causes------------------



    array(['Potholes', 'Speedbump', 'Car wash', 'Gravel road'], dtype=object)



    --------------------------------End---------------------------



```python

```

Note:we see that the all types of impact cause in `train dataset` also occured in the `test dataset.` Let's visualise new predictions in the `test dataset`:


```python
New_df2 = pd.concat([test_data2, New_df.cause], axis = 1)

print('-------------------Impact cause predicted in the test dataset--------------')
display(New_df2.head(5))

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
      <th>gps_lat</th>
      <th>gps_lon</th>
      <th>Hour</th>
      <th>Month</th>
      <th>Quarter</th>
      <th>Day</th>
      <th>Week</th>
      <th>cause</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-26.105012</td>
      <td>28.154925</td>
      <td>9</td>
      <td>5</td>
      <td>2</td>
      <td>22</td>
      <td>21</td>
      <td>Potholes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-26.069402</td>
      <td>27.851743</td>
      <td>8</td>
      <td>2</td>
      <td>1</td>
      <td>13</td>
      <td>7</td>
      <td>Potholes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-26.111047</td>
      <td>28.255036</td>
      <td>12</td>
      <td>5</td>
      <td>2</td>
      <td>3</td>
      <td>18</td>
      <td>Potholes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-26.167354</td>
      <td>28.080104</td>
      <td>9</td>
      <td>1</td>
      <td>1</td>
      <td>22</td>
      <td>4</td>
      <td>Speedbump</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-25.974256</td>
      <td>27.965016</td>
      <td>9</td>
      <td>4</td>
      <td>2</td>
      <td>17</td>
      <td>16</td>
      <td>Speedbump</td>
    </tr>
  </tbody>
</table>
</div>


    ---------------------------------------End--------------------------------



```python

print('\n------------------Number of accident in training dataset--------------------')
display(df1.cause.value_counts())

print('\n---------------------Number of accident in test dataset---------------------')
display(New_df2.cause.value_counts())
print('--'*38)

```


    ------------------Number of accident in training dataset--------------------



    Speedbump      1394
    Potholes        878
    Car Wash        649
    Gravel road      54
    Name: cause, dtype: int64



    ---------------------Number of accident in test dataset---------------------



    Speedbump      266
    Potholes       173
    Car wash       146
    Gravel road     10
    Name: cause, dtype: int64


    ----------------------------------------------------------------------------


# Additional work - holdout dataset

Exploratory data analysys for a `test set` - named `train dataset` - with new labels predicted in this work.


```python
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
New_df2.cause.value_counts().plot(kind = 'bar', rot = 0)
plt.xlabel('Impact cause')
plt.ylabel('Number of accident')
plt.subplot(1,2,2)
New_df2.cause.value_counts().plot('pie', explode=(0,0,0,0.1), autopct='%1.2f%%',
                                  shadow=True, legend = 'best', labels = None)
plt.show()
```


![png](https://drive.google.com/uc?export=view&id=1B1myoj5yMQppVOoSDrf02G4csccM3Yqi)

We can also add new predicted targets to the test dataset, for exploratory data analysis purpose


```python
xx = pd.concat([test_data, New_df.cause], axis = 1)

display(xx.head())
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
      <th>gps_lat</th>
      <th>gps_lon</th>
      <th>alert_timestamp</th>
      <th>Hour</th>
      <th>Month</th>
      <th>Quarter</th>
      <th>Year</th>
      <th>Day</th>
      <th>Week</th>
      <th>Season</th>
      <th>cause</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-26.105012</td>
      <td>28.154925</td>
      <td>2018-05-22 09:22:06</td>
      <td>Morning \n (06h00 - 12h00)</td>
      <td>May</td>
      <td>2</td>
      <td>2018</td>
      <td>Tue</td>
      <td>21</td>
      <td>Autumn</td>
      <td>Potholes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-26.069402</td>
      <td>27.851743</td>
      <td>2018-02-13 08:06:28</td>
      <td>Morning \n (06h00 - 12h00)</td>
      <td>February</td>
      <td>1</td>
      <td>2018</td>
      <td>Tue</td>
      <td>7</td>
      <td>Summer</td>
      <td>Potholes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-26.111047</td>
      <td>28.255036</td>
      <td>2018-05-03 12:59:05</td>
      <td>Morning \n (06h00 - 12h00)</td>
      <td>May</td>
      <td>2</td>
      <td>2018</td>
      <td>Thu</td>
      <td>18</td>
      <td>Autumn</td>
      <td>Potholes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-26.167354</td>
      <td>28.080104</td>
      <td>2018-01-22 09:08:51</td>
      <td>Morning \n (06h00 - 12h00)</td>
      <td>January</td>
      <td>1</td>
      <td>2018</td>
      <td>Mon</td>
      <td>4</td>
      <td>Summer</td>
      <td>Speedbump</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-25.974256</td>
      <td>27.965016</td>
      <td>2018-04-17 09:54:55</td>
      <td>Morning \n (06h00 - 12h00)</td>
      <td>April</td>
      <td>2</td>
      <td>2018</td>
      <td>Tue</td>
      <td>16</td>
      <td>Autumn</td>
      <td>Speedbump</td>
    </tr>
  </tbody>
</table>
</div>


For a `test dataset:`


```python
plt.figure(figsize = (19, 15))
plt.subplot(4,2,1)
sns.countplot(x = 'Season', hue = 'cause', data = train_data)
plt.xlabel('Season')
plt.ylabel('Number of accidents')
plt.tight_layout()


plt.subplot(4,2,2)
sns.countplot(x = 'Month', hue = 'cause', data = train_data)
plt.xlabel('Month')
plt.ylabel('Number of accidents')
plt.tight_layout()


plt.subplot(4,2,3)
sns.countplot(x = 'Day', hue = 'cause', data = train_data)
plt.xlabel('Day')
plt.ylabel('Number of accidents')
plt.tight_layout()


plt.subplot(4,2,4)
sns.countplot(x = 'Hour', hue = 'cause', data = train_data)
plt.xlabel('Hour')
plt.ylabel('Number of accidents')
plt.tight_layout()

plt.subplot(4,2,5)
sns.countplot(x = 'Week', hue = 'cause', data = train_data)
plt.xlabel('Week')
plt.ylabel('Number of accidents')
plt.tight_layout()


plt.subplot(4,2,6)
sns.countplot(x = 'Quarter', hue = 'cause', data = train_data)
plt.xlabel('Quarter')
plt.ylabel('Number of accidents')
plt.tight_layout()
```


![png](https://drive.google.com/uc?export=view&id=1UYsud5emKtWlO28hXTrYfubvwmF9zzFl)

Let's now view locations where different accident impacts occurs in the `test dataset`:


```python
#base map
gps_map2 = fl.Map(location=[(train_data['gps_lat']).mean(), (train_data['gps_lon']).mean()],
           tiles = 'Stamen Toner', zoom_start = 12)

#defining calors for different impacts
colors = {'Speedbump': 'red', 'Potholes': 'green', 'Car wash': 'yellow', 'Gravel road': 'blue'}


for i, j, k in zip(xx['gps_lat'], xx['gps_lon'], xx['cause']):

    fl.CircleMarker(radius = 3, location = [i, j],
                    color = 'g',
                    fill_color = colors[k], fill = True,
                    fill_opacity = 1
                   ).add_to(gps_map2)
gps_map2
```








```python

```


```python

```