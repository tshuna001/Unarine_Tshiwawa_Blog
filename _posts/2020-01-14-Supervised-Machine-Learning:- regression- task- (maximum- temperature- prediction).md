---
layout: post
author: "Unarine Tshiwawa"
---

```python
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
%matplotlib inline
import os
from IPython.display import display
import seaborn as sns
sns.set(context='notebook')
import pandas_profiling
```


```python
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
```

# $$\mathrm{Regression\, for\, predicting\, continuous\, outcomes}$$

### Dataset:

Define the problem: We want to establish linear relationship between minimum temperature and maximum temperature in Barajas Airport in Madrid, between 1997 and 2015. The dataset in this work comes from [Kaggle](https://www.kaggle.com/juliansimon/weather_madrid_lemd_1997_2015.csv)

#### Main point: we are trying to build a linear model that will predict maximum temperature from minimum temperature.

Clearly, we see that this is a supervised machine leaning - regression - type of a problem.  An output to this problem is a continuous quantity.


### Read dataset


```python
file = 'weather_madrid_lemd_1997_2015.csv.zip'

df = pd.read_csv(file, compression = 'zip')

df1 = df.copy()
```

### Dataset information:

The info() method is useful to get a quick description of the data


```python
display(df1.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6812 entries, 0 to 6811
    Data columns (total 23 columns):
    CET                            6812 non-null object
    Max TemperatureC               6810 non-null float64
    Mean TemperatureC              6809 non-null float64
    Min TemperatureC               6810 non-null float64
    Dew PointC                     6810 non-null float64
    MeanDew PointC                 6810 non-null float64
    Min DewpointC                  6810 non-null float64
    Max Humidity                   6810 non-null float64
     Mean Humidity                 6810 non-null float64
     Min Humidity                  6810 non-null float64
     Max Sea Level PressurehPa     6812 non-null int64
     Mean Sea Level PressurehPa    6812 non-null int64
     Min Sea Level PressurehPa     6812 non-null int64
     Max VisibilityKm              5872 non-null float64
     Mean VisibilityKm             5872 non-null float64
     Min VisibilitykM              5872 non-null float64
     Max Wind SpeedKm/h            6812 non-null int64
     Mean Wind SpeedKm/h           6812 non-null int64
     Max Gust SpeedKm/h            3506 non-null float64
    Precipitationmm                6812 non-null float64
     CloudCover                    5440 non-null float64
     Events                        1798 non-null object
    WindDirDegrees                 6812 non-null int64
    dtypes: float64(15), int64(6), object(2)
    memory usage: 1.2+ MB



    None


- NB: There are 6812 instances in the dataset, which means that it is very small by Machine Learning standards, but it’s perfect to get started.

- All attributes are numerical, except the `CET` and `Events` field

### Shape of dataset:


```python
df1.shape
```




    (6812, 23)



### Features of each instance in the dataset:


```python
df1.columns
```




    Index(['CET', 'Max TemperatureC', 'Mean TemperatureC', 'Min TemperatureC',
           'Dew PointC', 'MeanDew PointC', 'Min DewpointC', 'Max Humidity',
           ' Mean Humidity', ' Min Humidity', ' Max Sea Level PressurehPa',
           ' Mean Sea Level PressurehPa', ' Min Sea Level PressurehPa',
           ' Max VisibilityKm', ' Mean VisibilityKm', ' Min VisibilitykM',
           ' Max Wind SpeedKm/h', ' Mean Wind SpeedKm/h', ' Max Gust SpeedKm/h',
           'Precipitationmm', ' CloudCover', ' Events', 'WindDirDegrees'],
          dtype='object')



### Take a Quick Look at the Data Structure

Let’s take a look at the top five rows using the DataFrame’s head() method


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
      <th>CET</th>
      <th>Max TemperatureC</th>
      <th>Mean TemperatureC</th>
      <th>Min TemperatureC</th>
      <th>Dew PointC</th>
      <th>MeanDew PointC</th>
      <th>Min DewpointC</th>
      <th>Max Humidity</th>
      <th>Mean Humidity</th>
      <th>Min Humidity</th>
      <th>Max Sea Level PressurehPa</th>
      <th>Mean Sea Level PressurehPa</th>
      <th>Min Sea Level PressurehPa</th>
      <th>Max VisibilityKm</th>
      <th>Mean VisibilityKm</th>
      <th>Min VisibilitykM</th>
      <th>Max Wind SpeedKm/h</th>
      <th>Mean Wind SpeedKm/h</th>
      <th>Max Gust SpeedKm/h</th>
      <th>Precipitationmm</th>
      <th>CloudCover</th>
      <th>Events</th>
      <th>WindDirDegrees</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1997-1-1</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>100.0</td>
      <td>95.0</td>
      <td>76.0</td>
      <td>1010</td>
      <td>1008</td>
      <td>1004</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>4.0</td>
      <td>13</td>
      <td>6</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>229</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1997-1-2</td>
      <td>7.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>92.0</td>
      <td>71.0</td>
      <td>1007</td>
      <td>1003</td>
      <td>997</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>4.0</td>
      <td>26</td>
      <td>8</td>
      <td>47.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>Rain</td>
      <td>143</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1997-1-3</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>100.0</td>
      <td>85.0</td>
      <td>70.0</td>
      <td>1005</td>
      <td>999</td>
      <td>996</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>7.0</td>
      <td>27</td>
      <td>19</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>Rain-Snow</td>
      <td>256</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1997-1-4</td>
      <td>7.0</td>
      <td>3.0</td>
      <td>-1.0</td>
      <td>-2.0</td>
      <td>-3.0</td>
      <td>-4.0</td>
      <td>86.0</td>
      <td>63.0</td>
      <td>49.0</td>
      <td>1012</td>
      <td>1010</td>
      <td>1005</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>27</td>
      <td>19</td>
      <td>40.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>284</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1997-1-5</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>-3.0</td>
      <td>100.0</td>
      <td>95.0</td>
      <td>86.0</td>
      <td>1012</td>
      <td>1008</td>
      <td>1005</td>
      <td>10.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>14</td>
      <td>6</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>Snow</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1997-1-6</td>
      <td>7.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>-1.0</td>
      <td>-3.0</td>
      <td>100.0</td>
      <td>82.0</td>
      <td>57.0</td>
      <td>1014</td>
      <td>1010</td>
      <td>1008</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>11</td>
      <td>5</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>64</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1997-1-7</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>-2.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>-3.0</td>
      <td>100.0</td>
      <td>93.0</td>
      <td>75.0</td>
      <td>1016</td>
      <td>1014</td>
      <td>1009</td>
      <td>10.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>6</td>
      <td>2</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>Snow</td>
      <td>43</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1997-1-8</td>
      <td>8.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>100.0</td>
      <td>96.0</td>
      <td>87.0</td>
      <td>1015</td>
      <td>1005</td>
      <td>1003</td>
      <td>10.0</td>
      <td>8.0</td>
      <td>4.0</td>
      <td>26</td>
      <td>8</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>Rain</td>
      <td>273</td>
    </tr>
  </tbody>
</table>
</div>


When looking at the top five rows, you probably noticed that the inputs in the `Events` column are repetitive,
which means that it is probably a categorical attribute. So we can find out what categories exist and how many weather events belong to each category by using the value_counts() method:


```python
df1[' Events'].value_counts()
```




    Rain                      1140
    Rain-Thunderstorm          247
    Fog                        233
    Fog-Rain                    69
    Thunderstorm                45
    Rain-Snow                   33
    Snow                        14
    Rain-Hail-Thunderstorm       7
    Fog-Snow                     4
    Fog-Rain-Thunderstorm        1
    Tornado                      1
    Rain-Hail                    1
    Fog-Thunderstorm             1
    Rain-Snow-Thunderstorm       1
    Fog-Rain-Snow                1
    Name:  Events, dtype: int64



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
      <th>Max TemperatureC</th>
      <th>Mean TemperatureC</th>
      <th>Min TemperatureC</th>
      <th>Dew PointC</th>
      <th>MeanDew PointC</th>
      <th>Min DewpointC</th>
      <th>Max Humidity</th>
      <th>Mean Humidity</th>
      <th>Min Humidity</th>
      <th>Max Sea Level PressurehPa</th>
      <th>Mean Sea Level PressurehPa</th>
      <th>Min Sea Level PressurehPa</th>
      <th>Max VisibilityKm</th>
      <th>Mean VisibilityKm</th>
      <th>Min VisibilitykM</th>
      <th>Max Wind SpeedKm/h</th>
      <th>Mean Wind SpeedKm/h</th>
      <th>Max Gust SpeedKm/h</th>
      <th>Precipitationmm</th>
      <th>CloudCover</th>
      <th>WindDirDegrees</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6810.000000</td>
      <td>6809.000000</td>
      <td>6810.000000</td>
      <td>6810.000000</td>
      <td>6810.000000</td>
      <td>6810.000000</td>
      <td>6810.000000</td>
      <td>6810.000000</td>
      <td>6810.000000</td>
      <td>6812.000000</td>
      <td>6812.000000</td>
      <td>6812.000000</td>
      <td>5872.000000</td>
      <td>5872.000000</td>
      <td>5872.000000</td>
      <td>6812.000000</td>
      <td>6812.000000</td>
      <td>3506.000000</td>
      <td>6812.000000</td>
      <td>5440.000000</td>
      <td>6812.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>21.039648</td>
      <td>14.658687</td>
      <td>8.640529</td>
      <td>8.120705</td>
      <td>4.976211</td>
      <td>1.451248</td>
      <td>81.139354</td>
      <td>57.971366</td>
      <td>34.729369</td>
      <td>1020.529360</td>
      <td>1017.973136</td>
      <td>1015.217410</td>
      <td>14.644074</td>
      <td>11.719857</td>
      <td>9.134877</td>
      <td>21.953171</td>
      <td>9.170728</td>
      <td>43.988306</td>
      <td>0.111182</td>
      <td>3.206066</td>
      <td>197.234586</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.867187</td>
      <td>7.580461</td>
      <td>6.837626</td>
      <td>4.741067</td>
      <td>4.654270</td>
      <td>4.909705</td>
      <td>17.531839</td>
      <td>19.675744</td>
      <td>19.320359</td>
      <td>6.235941</td>
      <td>6.480085</td>
      <td>6.944745</td>
      <td>8.770024</td>
      <td>5.592324</td>
      <td>5.075065</td>
      <td>9.903914</td>
      <td>5.110013</td>
      <td>12.252462</td>
      <td>0.967174</td>
      <td>1.808948</td>
      <td>119.872777</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>-3.000000</td>
      <td>-10.000000</td>
      <td>-12.000000</td>
      <td>-15.000000</td>
      <td>-22.000000</td>
      <td>16.000000</td>
      <td>15.000000</td>
      <td>4.000000</td>
      <td>994.000000</td>
      <td>986.000000</td>
      <td>965.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>19.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>13.000000</td>
      <td>8.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>2.000000</td>
      <td>-2.000000</td>
      <td>68.000000</td>
      <td>41.000000</td>
      <td>19.000000</td>
      <td>1017.000000</td>
      <td>1014.000000</td>
      <td>1011.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>7.000000</td>
      <td>14.000000</td>
      <td>6.000000</td>
      <td>35.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>66.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>20.000000</td>
      <td>14.000000</td>
      <td>9.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>87.000000</td>
      <td>59.000000</td>
      <td>32.000000</td>
      <td>1020.000000</td>
      <td>1018.000000</td>
      <td>1015.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>21.000000</td>
      <td>8.000000</td>
      <td>42.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>223.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>29.000000</td>
      <td>21.000000</td>
      <td>14.000000</td>
      <td>12.000000</td>
      <td>8.000000</td>
      <td>5.000000</td>
      <td>94.000000</td>
      <td>74.000000</td>
      <td>47.750000</td>
      <td>1024.000000</td>
      <td>1022.000000</td>
      <td>1019.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>27.000000</td>
      <td>11.000000</td>
      <td>52.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>299.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>41.000000</td>
      <td>32.000000</td>
      <td>28.000000</td>
      <td>20.000000</td>
      <td>16.000000</td>
      <td>14.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>1047.000000</td>
      <td>1043.000000</td>
      <td>1041.000000</td>
      <td>31.000000</td>
      <td>31.000000</td>
      <td>31.000000</td>
      <td>182.000000</td>
      <td>39.000000</td>
      <td>103.000000</td>
      <td>32.000000</td>
      <td>8.000000</td>
      <td>360.000000</td>
    </tr>
  </tbody>
</table>
</div>


The count , mean , min , and max rows are self-explanatory.  Note that the null values are ignored (so, for example, count of `Max TemperatureC`is 6810, not 6812). The std row shows the standard deviation, which measures how dispersed the values are. The 25%, 50%, and 75% rows show the corresponding percentiles: a percentile indicates the value below which a given percentage of observations in a group of observations falls

### Looking for Correlations:

Since the dataset is not too large, you can easily compute the standard correlation coefficient (also called Pearson’s r) between every pair of attributes using the corr() method:


```python
corr_matrix = df1.corr()

corr_matrix["Max TemperatureC"].sort_values(ascending=False)
```




    Max TemperatureC               1.000000
    Mean TemperatureC              0.970983
    Min TemperatureC               0.856143
    Dew PointC                     0.583509
    MeanDew PointC                 0.495857
     Min VisibilitykM              0.388285
    Min DewpointC                  0.329547
     Mean VisibilityKm             0.299083
     Max VisibilityKm              0.126435
     Max Wind SpeedKm/h            0.025244
    WindDirDegrees                 0.000365
     Mean Sea Level PressurehPa   -0.011203
     Min Sea Level PressurehPa    -0.042676
    Precipitationmm               -0.068196
     Mean Wind SpeedKm/h          -0.073119
     Max Sea Level PressurehPa    -0.081856
     Max Gust SpeedKm/h           -0.135634
     CloudCover                   -0.449610
    Max Humidity                  -0.718184
     Min Humidity                 -0.761490
     Mean Humidity                -0.805961
    Name: Max TemperatureC, dtype: float64



The correlation coefficient ranges from –1 to 1. When it is close to 1, it means that there is a strong positive correlation.  When the coefficient is close to –1, it means that there is a strong negative correlation.  Finally, coefficients close to zero mean that there is no linear correlation


```python

```

## Exploratory Data Analysis


```python
df1[' Events'].unique()
```




    array([nan, 'Rain', 'Rain-Snow', 'Snow', 'Fog', 'Fog-Rain',
           'Rain-Thunderstorm', 'Thunderstorm', 'Rain-Hail-Thunderstorm',
           'Fog-Thunderstorm', 'Tornado', 'Fog-Rain-Thunderstorm',
           'Fog-Rain-Snow', 'Fog-Snow', 'Rain-Snow-Thunderstorm', 'Rain-Hail'],
          dtype=object)




```python

```

### Time Series & Feature engineering:

Let's convert date (`CET`) to datetime object so that we can easily extract some useful information from it, such as `time`, `day` of the week, and `month`.



Date:


```python
display('Beginning date:', df1['CET'][0], 'End date:', df1['CET'][-1:])
```


    'Beginning date:'



    '1997-1-1'



    'End date:'



    6811    2015-12-31
    Name: CET, dtype: object


Let's convert date (`CET`) to datetime object so that we can easily extract some useful information from it, such as day of the week, month and year.


```python
df1['CET'] = df1['CET'].apply(pd.to_datetime)

df1['year'] = [i.year for i in df1['CET']]
df1['month'] = [i.month_name()[0:3] for i in df1['CET']]
df1['day'] = [i.day_name()[0:3] for i in df1['CET']]
```


```python

```


```python
def date_property(df):

    "extract all date properties from a datetime datatype by month, quarter, year, day and week"

    df['CET'] = pd.to_datetime(df['CET'], infer_datetime_format=True)


    df["Month"] = df["CET"].dt.month
    df["Quarter"] = df["CET"].dt.quarter
    df["Year"] = df["CET"].dt.year
    df["Day"] = df["CET"].dt.day
    df["Week"] = df["CET"].dt.week


    df["Season"] = np.where(df["Month"].isin([3,4,5]),"Autumn",
                np.where(df["Month"].isin([6,7,8]), "Winter",
                np.where(df["Month"].isin([9,10,11]),"Spring",
                np.where(df["Month"].isin([12,1,2]), "Summer","None"))))

    df_1 = df.copy()


    '''mapping months from jan -> Aug'''

    df_1.Month = df_1.Month.map({1:'January', 2:'February', 3:'March', 4:'April', 5:'May', 6:'June',
                                 7:'July', 8:'August', 9:'September', 10:'October', 11:'November', 12:'December'})
    df_1.Month = pd.Categorical(df_1.Month, ['January', 'February', 'March', 'April', 'May', 'June',
                                             'July', 'August', 'September', 'October', 'November', 'December'])

    '''extracting days of the week'''

    df_1.Day = [j.day_name()[0:3] for j in df_1['CET']]
    df_1.Day = pd.Categorical(df_1.Day, ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])



    return df, df_1





```


```python

```


```python
train_data = date_property(df1)[1]


print('\n----------------------Train set------------------')
display(train_data[["CET","Year","Month","Day","Week","Quarter","Season"]].sample(15))

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
      <th>CET</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
      <th>Week</th>
      <th>Quarter</th>
      <th>Season</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3622</th>
      <td>2007-04-08</td>
      <td>2007</td>
      <td>April</td>
      <td>Sun</td>
      <td>14</td>
      <td>2</td>
      <td>Autumn</td>
    </tr>
    <tr>
      <th>3401</th>
      <td>2006-08-30</td>
      <td>2006</td>
      <td>August</td>
      <td>Wed</td>
      <td>35</td>
      <td>3</td>
      <td>Winter</td>
    </tr>
    <tr>
      <th>588</th>
      <td>1998-08-12</td>
      <td>1998</td>
      <td>August</td>
      <td>Wed</td>
      <td>33</td>
      <td>3</td>
      <td>Winter</td>
    </tr>
    <tr>
      <th>3281</th>
      <td>2006-05-02</td>
      <td>2006</td>
      <td>May</td>
      <td>Tue</td>
      <td>18</td>
      <td>2</td>
      <td>Autumn</td>
    </tr>
    <tr>
      <th>6051</th>
      <td>2013-12-01</td>
      <td>2013</td>
      <td>December</td>
      <td>Sun</td>
      <td>48</td>
      <td>4</td>
      <td>Summer</td>
    </tr>
    <tr>
      <th>4607</th>
      <td>2009-12-18</td>
      <td>2009</td>
      <td>December</td>
      <td>Fri</td>
      <td>51</td>
      <td>4</td>
      <td>Summer</td>
    </tr>
    <tr>
      <th>2183</th>
      <td>2003-03-31</td>
      <td>2003</td>
      <td>March</td>
      <td>Mon</td>
      <td>14</td>
      <td>1</td>
      <td>Autumn</td>
    </tr>
    <tr>
      <th>3789</th>
      <td>2007-09-22</td>
      <td>2007</td>
      <td>September</td>
      <td>Sat</td>
      <td>38</td>
      <td>3</td>
      <td>Spring</td>
    </tr>
    <tr>
      <th>4752</th>
      <td>2010-05-12</td>
      <td>2010</td>
      <td>May</td>
      <td>Wed</td>
      <td>19</td>
      <td>2</td>
      <td>Autumn</td>
    </tr>
    <tr>
      <th>5689</th>
      <td>2012-12-04</td>
      <td>2012</td>
      <td>December</td>
      <td>Tue</td>
      <td>49</td>
      <td>4</td>
      <td>Summer</td>
    </tr>
    <tr>
      <th>4388</th>
      <td>2009-05-13</td>
      <td>2009</td>
      <td>May</td>
      <td>Wed</td>
      <td>20</td>
      <td>2</td>
      <td>Autumn</td>
    </tr>
    <tr>
      <th>6248</th>
      <td>2014-06-16</td>
      <td>2014</td>
      <td>June</td>
      <td>Mon</td>
      <td>25</td>
      <td>2</td>
      <td>Winter</td>
    </tr>
    <tr>
      <th>1423</th>
      <td>2001-03-01</td>
      <td>2001</td>
      <td>March</td>
      <td>Thu</td>
      <td>9</td>
      <td>1</td>
      <td>Autumn</td>
    </tr>
    <tr>
      <th>4501</th>
      <td>2009-09-03</td>
      <td>2009</td>
      <td>September</td>
      <td>Thu</td>
      <td>36</td>
      <td>3</td>
      <td>Spring</td>
    </tr>
    <tr>
      <th>3865</th>
      <td>2007-12-07</td>
      <td>2007</td>
      <td>December</td>
      <td>Fri</td>
      <td>49</td>
      <td>4</td>
      <td>Summer</td>
    </tr>
  </tbody>
</table>
</div>



```python

```


```python
plt.figure(figsize = (14, 6))
sns.countplot(x = 'Season', hue = ' Events', data = train_data)
plt.legend(loc=1)
plt.xlabel('Season', size = 15)
plt.ylabel('Count', size=15)
plt.tight_layout()
```

![png](https://drive.google.com/uc?export=view&id=1kF9x4SE2_JZIwSmlA4h8uRvxhMFDHkfb)


```python

```


```python
plt.figure(figsize = (15, 6))
sns.countplot(x = 'Month', hue = ' Events', data = train_data)
plt.xlabel('Month', size=15)
plt.ylabel('Count', size=15)
#plt.tick_params(axis='x', labelsize=12, width=2, length = 4)
#plt.tick_params(axis='y', labelsize=12, width=2, length = 4)
plt.tight_layout()
```


![png](https://drive.google.com/uc?export=view&id=1V8gHuPBrm1Al9Q6eAiCCntOdG7TDcEpM)


```python

```


```python
plt.figure(figsize = (12, 6))
sns.countplot(x = 'Day', hue = ' Events', data = train_data)
plt.xlabel('Day', size = 15)
plt.ylabel('Count', size = 15)
plt.tight_layout()

```

![png](https://drive.google.com/uc?export=view&id=1AenjnIdIXuABNDi84vtmguUf6B4mBSsO)


```python

```


```python
plt.figure(figsize=(15,7))
sns.countplot(x = 'Quarter', hue = ' Events', data = train_data)
plt.xlabel('Quarter', size=15)
plt.ylabel('Event', size=15)
```




    Text(0, 0.5, 'Event')




![png](https://drive.google.com/uc?export=view&id=1waxYFhEjbvh1PtGFQFGPcrh-H1Oe_j_J)



```python

```


```python
plt.figure(figsize=(15,6))
sns.countplot(data = df1, x = 'year', hue = ' Events')
plt.xlabel('Year', size = 15)
plt.ylabel('Count', size = 15)
```




    Text(0, 0.5, 'Count')




![png](https://drive.google.com/uc?export=view&id=1mSAUz6TgbHU04e-Eas15PiMN6wY0uTiC)


```python

```

A histogram for each numerical attribute:


```python
df1.hist(bins=50, figsize=(20,15))
plt.show()
```

![png](https://drive.google.com/uc?export=view&id=1ftsiIGBBDQf5GeJFoc1OjH9EiaR0xKck)


Bar plot for categorical attribute:


```python
plt.figure(figsize = (14,7))

plt.subplot(1, 2, 1)
df1[' Events'].value_counts().plot(kind = 'bar', title = 'Weather event', rot = 90)

plt.subplot(1, 2, 2)
df1[' Events'].value_counts().plot(kind = 'pie',
                                 autopct='%1.2f%%',
                                 shadow=True,
                                 title = 'Weather event', labels = None, legend = 'bbox_to_anchor=(1.5, 2)')

plt.tight_layout()
```


![png](https://drive.google.com/uc?export=view&id=1sxu3-MKwHALbMibj8YRzE6e7xdFWcwPF)

- Madrid has on average only ~ 63.4 % precipitation from 1997 - 2015

Imputing nan values:


```python
display(df1.isnull().sum())
```


    CET                               0
    Max TemperatureC                  2
    Mean TemperatureC                 3
    Min TemperatureC                  2
    Dew PointC                        2
    MeanDew PointC                    2
    Min DewpointC                     2
    Max Humidity                      2
     Mean Humidity                    2
     Min Humidity                     2
     Max Sea Level PressurehPa        0
     Mean Sea Level PressurehPa       0
     Min Sea Level PressurehPa        0
     Max VisibilityKm               940
     Mean VisibilityKm              940
     Min VisibilitykM               940
     Max Wind SpeedKm/h               0
     Mean Wind SpeedKm/h              0
     Max Gust SpeedKm/h            3306
    Precipitationmm                   0
     CloudCover                    1372
     Events                        5014
    WindDirDegrees                    0
    year                              0
    month                             0
    day                               0
    Month                             0
    Quarter                           0
    Year                              0
    Day                               0
    Week                              0
    Season                            0
    dtype: int64


#### Recall, the prime goal is to use minimum temperature to predict maximum temperature in Barajas Airport - Madrid

We shall replace nan values with the median of the column for both features:


```python
display(df1[["Min TemperatureC","Max TemperatureC"]].head())
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
      <th>Min TemperatureC</th>
      <th>Max TemperatureC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
display(df1[["Min TemperatureC","Max TemperatureC"]].isnull().sum())
```


    Min TemperatureC    2
    Max TemperatureC    2
    dtype: int64


We see that `Min TemperatureC` and `Max TemperatureC` in the dataset have 2 null values, respectively.  So we can relace null values by the median of the column


```python
x = (df1["Min TemperatureC"].fillna(df1["Min TemperatureC"].median()))

y = (df1["Max TemperatureC"].fillna(df1["Max TemperatureC"].median()))

print(x.shape, y.shape)
```

    (6812,) (6812,)


Let's make some plots:


```python
plt.figure(figsize=(10,6))
plt.plot(x, y, '.k' )
plt.title("MaxTemp vs MinTemp")
plt.xlabel("MinTemp")
plt.ylabel("MaxTemp")


```




    Text(0, 0.5, 'MaxTemp')



![png](https://drive.google.com/uc?export=view&id=1cgPBlIdSoS7ogsf_lHOl1nUCl8D3GD2W)


Note: Looking at the figure above, it appears that there is some sort of linear relationship between minimum and maximum temperature.

Let's try to inspect the average maximum temperature:


```python
plt.figure(figsize=(10,6))
sns.distplot(y)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fb43f253b70>




![png](https://drive.google.com/uc?export=view&id=1qQYP3liYP3i147ywRqkM4OVTmz5xt4GB)


## Data Splicing

We are going to split the data into two sets -  80$\%$ train set and 20$\%$ test set.



```python
from sklearn.model_selection import train_test_split

#transform from row vector to column vector
xx = x[:, np.newaxis]
yy = y[:, np.newaxis]

#splicing
x_train, x_test, y_train, y_test = train_test_split(xx, yy, test_size = 0.2, random_state = 42)

```

Let's check the dimensions of all sets:


```python
y_train.shape, y_test.shape
```




    ((5449, 1), (1363, 1))



## Machine Learning model

This is the part where the classifier will be instantiate and trained.


```python
from sklearn import metrics
from sklearn.linear_model import LinearRegression

#training the algorithm
LR = LinearRegression(normalize = True).fit(x_train, y_train)

#To retrieve the intercept
print("intercept :", LR.intercept_)
print("coefficient :", LR.coef_)

```

    intercept : [11.53244273]
    coefficient : [[1.1037861]]


# Prediction:

Let's make prediction to see how accurately the algorithm predict the percentage score


```python
y_pred = LR.predict(x_test)
```

## Model evaluation

Select a Performance Measure:

A typical performance measure for regression problems is the `Mean Absolute Error`, `Mean Squared Error`, `Root Mean Square Error (RMSE)`. It gives an idea of how much error the system typically makes in its predictions.


```python
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Square Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
```

    Mean Absolute Error: 3.962475426159121
    Mean Squared Error: 21.49057920091619
    Root Mean Square Error: 4.635793265549726


creating new dataframe consist of actual and predicted values:


```python
df2 = pd.DataFrame({"Actual": y_test.flatten(), "Predicted": y_pred.flatten()})

print('\n --------------------Let us view few rows in new dataframe------------------')
display(df2.head())
```


     --------------------Let us view few rows in new dataframe------------------



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
      <th>Actual</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32.0</td>
      <td>30.296806</td>
    </tr>
    <tr>
      <th>1</th>
      <td>32.0</td>
      <td>28.089234</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.0</td>
      <td>14.843801</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>20.362732</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15.0</td>
      <td>14.843801</td>
    </tr>
  </tbody>
</table>
</div>


Let's view few actual and predicted values:


```python
df2 = df2.head(15)
df2.plot(kind = 'bar', figsize=(10,6))
plt.grid(which = 'major', linestyle = '-', linewidth = '0.5', color = 'green')
plt.title('Comparison of Predicted and Actual values')
plt.xlabel('Variables')
plt.ylabel('Values')
```




    Text(0, 0.5, 'Values')




![png](https://drive.google.com/uc?export=view&id=1f4YiIZS5hlJpyrZNeZa0cJaS_5R4RpLI)


Let's view our linear model


```python
plt.figure(figsize=(10,6))
plt.scatter(x_test, y_test, color = 'gray')
plt.plot(x_test, y_pred, color = 'red', linewidth = 2)
plt.xlabel("MinTemp")
plt.ylabel("PredTemp")
plt.tight_layout()
plt.show()
```

![png](https://drive.google.com/uc?export=view&id=1SN0jqMT11REItWqx6vZOYTChaJTLfrDU)



```python

```
