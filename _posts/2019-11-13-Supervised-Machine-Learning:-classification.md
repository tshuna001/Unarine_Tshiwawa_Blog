---
layout: post
author: Unarine Tshiwawa
---
# 1. We shall start by defining the problem:

We are given a dataset containing number of false accidents impacts that are labelled (by time, season, and location) and a test dataset that contain impacts that are not labelled.  In this exercise, we will use a model trained on the dataset containing false impacts to classify different kinds of impacts cause in the test dataset.

#### Main point: we are trying to build a model that will predict labels for a new data.

Clearly, we see that this is a machine leaning - classification - type of a problem.  An output to this problem is a categorical quantity.


Consider that the only classification of impacts alert will be:

* Speedbumps
* Potholes
* Car Wash and
* Gravel Road



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

Read data:


```python
#false impacts (we shall call it a training dataset)
dataset1 = pd.read_csv('impacts_dataset_ch.csv')

#testset
dataset2 = pd.read_csv('impacts_test_dataset_ch.csv')

df1 = dataset1.copy()
df2 = dataset2.copy()
```

# Data Exploration Analysis:

Impute for missing values:


```python
print('\n ----------------------Train dataset------------------------')

display(df1.isna().sum())

print('\n ----------------------Test dataset------------------------')
display(df2.isna().sum())
```


     ----------------------Train dataset------------------------



    cause              0
    gps_lat            0
    gps_lon            0
    alert_timestamp    0
    impact_num         0
    dtype: int64



     ----------------------Test dataset------------------------



    impact_num         0
    gps_lat            0
    gps_lon            0
    alert_timestamp    0
    dtype: int64


Both datasets doen't contain missing values.

`Let's view some few rows in both datasets:`


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



Now, let's make some plots for `train dataset`:


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


![](https://raw.githubusercontent.com/tshuna001/images/master/Impact_alert_11_0.png?token=AE5UC7ZRJBTYXLITN6XZNZS54GDEE)


For each alert impact, we want to know how many accident happened per month in our dataset within stipulated time (2018-01-01 to 2018-08-13).  There is higher accident contribution from speedbump, potholes, car wash, and gravel road impact, respectively, to the total number of impacts.

### Working with Time Series:

For each impact cause, we would like to see/know the number of impacts per month.  This will be demonstrated via a count plot.




```python
df1['alert_timestamp'].describe()
```




    count                    2975
    unique                   2975
    top       2018-06-15 14:19:06
    freq                        1
    Name: alert_timestamp, dtype: object



Clearly, the sample doesn't cover the entire year.  

- We would like to know how many accidents happens per hour and month for each impact cause in dataset provided.


$\mathrm{\textbf{Approach}}$: We shall define a function that will divide the column feature "alert_timestamp" into new feature columns comprising of "Month" and "Hour" in both training and test set.


```python

def time_month_extraction(df):

    '''This function convert alert timestamp to hour and month '''

    df.alert_timestamp = pd.to_datetime(df.alert_timestamp)

    M = [] #months
    H = [] #Hours

    for i, j in enumerate(df.alert_timestamp):

        M.append(j.month)

        k = j.hour #extracting hours
        if k == 0:
            H.append(24) #24 hours
        else:
            H.append(k)

    '''Adding new columns consisting of hour and month'''    

    df['Hour'] = H
    df['Month'] = M

    df.Hour = df.Hour.astype(float)
    df.Month = df.Month.astype(float)

    df_1 = df.copy()

    '''mapping months from jan -> Aug'''

    df_1.Month = df_1.Month.map({1:'January', 2:'February', 3:'March', 4:'April', 5:'May', 6:'June',
                                 7:'July', 8:'August'})
    df_1.Month = pd.Categorical(df_1.Month, ['January', 'February', 'March', 'April', 'May', 'June',
                                             'July', 'August'])

    '''dividing a day into four parts.'''

    df_1.Hour = pd.cut(df_1.Hour, bins = [0, 6, 12, 18, 24], labels = ['Early morning (00h00-06h00)',
                                                                 'Morning (06h00 - 12h00)',
                                                                'Afternoon (12h00 - 18h00)',
                                                                'Evening (18h00 - 24h00)'])

    return df, df_1
```

Let's view few columns in both new training and test set with new feature columns comprises of "Month" and "Hour":


```python
train_data = time_month_extraction(dataset1)[1]
train_data = train_data.drop('impact_num', axis = 1)

test_data = time_month_extraction(dataset2)[1]
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Speedbump</td>
      <td>-26.268468</td>
      <td>28.060555</td>
      <td>2018-03-04 16:03:40</td>
      <td>Afternoon (12h00 - 18h00)</td>
      <td>March</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Potholes</td>
      <td>-26.074818</td>
      <td>28.066700</td>
      <td>2018-04-08 10:12:19</td>
      <td>Morning (06h00 - 12h00)</td>
      <td>April</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Speedbump</td>
      <td>-26.040653</td>
      <td>28.107715</td>
      <td>2018-05-12 17:23:51</td>
      <td>Afternoon (12h00 - 18h00)</td>
      <td>May</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Speedbump</td>
      <td>-26.194437</td>
      <td>28.037294</td>
      <td>2018-01-26 15:10:58</td>
      <td>Afternoon (12h00 - 18h00)</td>
      <td>January</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Speedbump</td>
      <td>-25.934343</td>
      <td>28.194816</td>
      <td>2018-06-16 06:07:04</td>
      <td>Early morning (00h00-06h00)</td>
      <td>June</td>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-26.105012</td>
      <td>28.154925</td>
      <td>2018-05-22 09:22:06</td>
      <td>Morning (06h00 - 12h00)</td>
      <td>May</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-26.069402</td>
      <td>27.851743</td>
      <td>2018-02-13 08:06:28</td>
      <td>Morning (06h00 - 12h00)</td>
      <td>February</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-26.111047</td>
      <td>28.255036</td>
      <td>2018-05-03 12:59:05</td>
      <td>Morning (06h00 - 12h00)</td>
      <td>May</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-26.167354</td>
      <td>28.080104</td>
      <td>2018-01-22 09:08:51</td>
      <td>Morning (06h00 - 12h00)</td>
      <td>January</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-25.974256</td>
      <td>27.965016</td>
      <td>2018-04-17 09:54:55</td>
      <td>Morning (06h00 - 12h00)</td>
      <td>April</td>
    </tr>
  </tbody>
</table>
</div>


    -------------------------------------------------


Now, we shall focus on the training set for further exploratory data analysis.


```python
train_data.alert_timestamp.describe()
```




    count                    2975
    unique                   2975
    top       2018-03-25 19:19:20
    freq                        1
    first     2018-01-01 19:44:54
    last      2018-08-13 11:49:50
    Name: alert_timestamp, dtype: object



Clearly, train dataset samples does not cover the entire year!


```python
plt.figure(figsize = (20, 7))
plt.subplot(1,2,1)
sns.countplot(x = 'Month', hue = 'cause', data = train_data)
plt.xlabel('Month')
plt.ylabel('Number of accidents')
plt.tight_layout()
plt.title('Impact cause per month')

plt.subplot(1,2,2)
sns.countplot(x = 'Hour', hue = 'cause', data = train_data)
plt.xlabel('Hour')
plt.ylabel('Number of accidents')
plt.tight_layout()
plt.title('Impact cause per hours')

plt.show()
```


![](https://raw.githubusercontent.com/tshuna001/images/master/Impact_alert_22_0.png?token=AE5UC76FVWDQO7VPCVN6WI254GC7W)


We shall plot the locations of the impacts on a map.  folium will be used to view locations

`Steps to consider:`

- defining the center of the map from GPS coodinates.
- showing different impact causes with different colors on the map



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





```python

```

#### Let's go deeper:

- We will need a model to help classify different impact cause in the dataset.  We need to instintiate an algorithm which will learn patterns from dataset in order to create a reasonable model.  This model will be used to predict target quantities in new datasets (we shall assume that impact cause in the test dataset also occur in the training dataset).

- In this problem, our target variables are known (speedbumps, potholes, gravel road, and car wash - related accident impacts), so supervise machine learning type will be a good fit.

- Now, we have a test dataset which contains only feature matrix.  We need to train machine learning classifier and create a model based on the dataset previously explored dataset (called **train** dataset then) and use the model obtained therefore to predict target features ( which are **impact causes**) in the **test dataset**.  

## 2. Preprocessing

We are going to generate a feature matrix and target vector.

Note: Our target variable is a string, so we need to perform one-hot encouding technique to the target variable.



#### Feature matrix:


```python
#feature matrix
train_data2 = time_month_extraction(df1)[0]

#droping redundant columns
train_data2 = train_data2.drop('cause', axis = 1)
train_data2 = train_data2.drop('alert_timestamp', axis = 1)
train_data2 = train_data2.drop('impact_num', axis = 1)

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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-26.268468</td>
      <td>28.060555</td>
      <td>16.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-26.074818</td>
      <td>28.066700</td>
      <td>10.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-26.040653</td>
      <td>28.107715</td>
      <td>17.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-26.194437</td>
      <td>28.037294</td>
      <td>15.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-25.934343</td>
      <td>28.194816</td>
      <td>6.0</td>
      <td>6.0</td>
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

We are going to split the data into two sets -  75$\%$ train set and 25$\%$ test set.  This means, 75$\%$ of the data will be set to train and optimize our machine learning model, and the remaining 25$\%$ will be used to test the model.




```python
from sklearn.model_selection import train_test_split

#data splicing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 10, stratify = y)

print(x_train.shape)
print(x_test.shape)

```

    (2231, 4)
    (744, 4)


#### Getting data into shape:

Data are not usually presented to the machine learning algorithm in exactly the same raw form as it is found. Usually data are scaled to a specific range in a process called normalization.

We are going to make selected features to the same scale for optimal performance, which is often achieved by transforming the features in the range [0, 1]: standardize features by removing the mean and scaling to unit variance.  This will be done in the pipeline to run multiple processes in the order that they are listed. The purpose of the pipeline is to assemble several steps that can be cross-validated together while setting different parameters.

# 3. Machine Learning Models

Now, the input and output variables are ready for training. This is the part where different class will be instintiated and a model which score high accuracy in both training and testing set will be used for further predictions.


```python
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

DT = Pipeline(steps=[('preprocessor', preprocessing.StandardScaler()),
                     ('model', DecisionTreeClassifier(random_state = 30))])
DT.fit(x_train, y_train)

print('-----------------------Decision Tree Classifier---------------------')
print('Accuraccy in training set: {:.2f}' .format(DT.score(x_train, y_train)))
print('Accuraccy in test set: {:.2f}' .format(DT.score(x_test, y_test)))


#Random forest
from sklearn.ensemble import RandomForestClassifier

#instantiate random forest class
RF = Pipeline(steps=[('preprocessor', preprocessing.StandardScaler()),
                     ('model', RandomForestClassifier(n_estimators = 1000, random_state = 30, n_jobs = -1))])

RF.fit(x_train, y_train)

print('\n-----------------------Random Forest Classifier---------------------')
print('Accuraccy in training set: {:.2f}' .format(RF.score(x_train, y_train)))
print('Accuraccy in test set: {:.2f}' .format(RF.score(x_test, y_test)))



from sklearn.neighbors import KNeighborsClassifier
KNN = Pipeline(steps=[('preprocessor', preprocessing.StandardScaler()),
                     ('model', KNeighborsClassifier())])
KNN.fit(x_train, y_train)

print('\n---------------------K-Nearest Neighbor Classifier------------------')
print('Accuraccy in training set: {:.2f}' .format(KNN.score(x_train, y_train)))
print('Accuraccy in test set: {:.2f}' .format(KNN.score(x_test, y_test)))

```

    -----------------------Decision Tree Classifier---------------------
    Accuraccy in training set: 1.00
    Accuraccy in test set: 0.91

    -----------------------Random Forest Classifier---------------------
    Accuraccy in training set: 1.00
    Accuraccy in test set: 0.97

    ---------------------K-Nearest Neighbor Classifier------------------
    Accuraccy in training set: 0.88
    Accuraccy in test set: 0.79


We see that Random Forest classifier perfoms better in the test and training set.  However, the other classifiers also perform better in the training set than in the testing set.  So we will use `Random Forest which gives an accuracy of 91 % in test set`.  This method will be employed for further predictions.

### 4. Model evaluation

We shall now show a confusion matrix showing the frequency of misclassifications by our
classifier.


```python
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, roc_auc_score, roc_curve


#predictions
pred = RF.predict(x_test)

target_names = order


mat = confusion_matrix(y_test.values.argmax(axis=1), pred.argmax(axis=1))

plt.figure(figsize = (12,7))
sns.set(font_scale=1.2)
sns.heatmap(mat, cbar = True, square=True, annot=True, yticklabels = target_names,
            annot_kws={'size': 15}, xticklabels=target_names, cmap='RdPu')

plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.tight_layout()
plt.show()

```


![](https://raw.githubusercontent.com/tshuna001/images/master/Impact_alert_40_0.png?token=AE5UC73KB63ICSN5M53HCEC54GC2K)


Let's look at summary of confusion matrix:


```python

target_names = order

summ_conf = classification_report(y_test.values.argmax(axis=1), pred.argmax(axis=1), target_names = target_names)


print('\n----------------------Classification Report-----------------------')
print('\n', summ_conf)
print('--'*33)
```


    ----------------------Classification Report-----------------------

                   precision    recall  f1-score   support

        Car wash       0.94      0.99      0.96       162
     Gravel road       1.00      1.00      1.00        13
        Potholes       1.00      0.94      0.96       220
       Speedbump       0.98      0.99      0.98       349

       micro avg       0.97      0.97      0.97       744
       macro avg       0.98      0.98      0.98       744
    weighted avg       0.97      0.97      0.97       744

    ------------------------------------------------------------------



```python

```

### We have trained our model and it's performing well.

Now, we will use the model to predict targets in the new dataset ('test set')


```python
test_data2 = time_month_extraction(df2)[0]
test_data2 = test_data2.drop('alert_timestamp', axis = 1)
test_data2 = test_data2.drop('impact_num', axis = 1)


#predictions in the new dataset
pred_test = RF.predict(test_data2)

#new data frame with column features label
New_df = pd.DataFrame(data = pred_test, columns = order)

#reverse onehot encoding to actual values
New_df['cause'] = (New_df.iloc[:, :] == 1).idxmax(1)

New_df.cause.unique()


```




    array(['Potholes', 'Speedbump', 'Car wash', 'Gravel road'], dtype=object)



Note:we see that the all types of impact cause in `train dataset` also occured in the `test dataset.`

#### Now, we are going to add the target variable (predicted by the model) to the `test dataset.`


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
      <th>cause</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-26.105012</td>
      <td>28.154925</td>
      <td>9.0</td>
      <td>5.0</td>
      <td>Potholes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-26.069402</td>
      <td>27.851743</td>
      <td>8.0</td>
      <td>2.0</td>
      <td>Potholes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-26.111047</td>
      <td>28.255036</td>
      <td>12.0</td>
      <td>5.0</td>
      <td>Potholes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-26.167354</td>
      <td>28.080104</td>
      <td>9.0</td>
      <td>1.0</td>
      <td>Speedbump</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-25.974256</td>
      <td>27.965016</td>
      <td>9.0</td>
      <td>4.0</td>
      <td>Speedbump</td>
    </tr>
  </tbody>
</table>
</div>


    ---------------------------------------End--------------------------------



```python

print('\n------------------Number of accident in training dataset--------------------')
display(df1.cause.value_counts())
print('--'*37)
```


    ------------------Number of accident in training dataset--------------------



    Speedbump      1394
    Potholes        878
    Car Wash        649
    Gravel road      54
    Name: cause, dtype: int64


    --------------------------------------------------------------------------



```python

print('\n---------------------Number of accident in test dataset--------------------')
display(New_df2.cause.value_counts())
print('--'*37)
```


    ---------------------Number of accident in test dataset--------------------



    Speedbump      265
    Potholes       169
    Car wash       152
    Gravel road      9
    Name: cause, dtype: int64


    --------------------------------------------------------------------------


For a `train dataset`:


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


![](https://raw.githubusercontent.com/tshuna001/images/master/Impact_alert_52_0.png?token=AE5UC7Y7M42MZM5Q3JOI7FK54GCXM)


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
      <th>cause</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-26.105012</td>
      <td>28.154925</td>
      <td>2018-05-22 09:22:06</td>
      <td>Morning (06h00 - 12h00)</td>
      <td>May</td>
      <td>Potholes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-26.069402</td>
      <td>27.851743</td>
      <td>2018-02-13 08:06:28</td>
      <td>Morning (06h00 - 12h00)</td>
      <td>February</td>
      <td>Potholes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-26.111047</td>
      <td>28.255036</td>
      <td>2018-05-03 12:59:05</td>
      <td>Morning (06h00 - 12h00)</td>
      <td>May</td>
      <td>Potholes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-26.167354</td>
      <td>28.080104</td>
      <td>2018-01-22 09:08:51</td>
      <td>Morning (06h00 - 12h00)</td>
      <td>January</td>
      <td>Speedbump</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-25.974256</td>
      <td>27.965016</td>
      <td>2018-04-17 09:54:55</td>
      <td>Morning (06h00 - 12h00)</td>
      <td>April</td>
      <td>Speedbump</td>
    </tr>
  </tbody>
</table>
</div>


For a `test dataset:`


```python
plt.figure(figsize = (20, 7))
plt.subplot(1,2,1)
sns.countplot(x = 'Month', hue = 'cause', data = xx)
plt.xlabel('Month')
plt.ylabel('Number of accidents')
plt.tight_layout()
plt.title('Impact cause per month')

plt.subplot(1,2,2)
sns.countplot(x = 'Hour', hue = 'cause', data = xx)
plt.xlabel('Hour')
plt.ylabel('Number of accidents')
plt.tight_layout()
plt.title('Impact cause per hours')
plt.show()
```


![](https://raw.githubusercontent.com/tshuna001/images/master/Impact_alert_56_0.png?token=AE5UC74K3JATWE4QK3RE67254GCS2)


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