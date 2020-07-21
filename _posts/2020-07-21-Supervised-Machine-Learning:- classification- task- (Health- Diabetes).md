---
layout: post
author: Unarine Tshiwawa
---
$$\mathrm{\textbf{Diabetes Task}}$$

The objective in this work is to use information from the patient to predict whether or not the patient has diabetes.  There are 8 features (predictor variables) and 1 label (response variable). The data is collected from acutal patients and represents a task which might commonly be undertaken by a human doctor interested in identifying the patients most at risk for diabetes in order to recommend preventative measures. The data was originally collected by the National Institute of Diabetes and Digestive and Kidney Diseases from a set of females at least 21 years old and of Pima Indian Heritage.

Clearly, we see that this is a machine leaning - binary classification task - type of a problem where the machine learning algorithm learns a set of rules in order to distinguish between two possible classes: non-diabetic and diabetic conditions in different petients. The main goal in supervised learning is to learn a model (based on past observations) from labeled training data that allows us to make predictions about unseen or future data of new instances.


```python
# Import libraries
'''Main'''
import numpy as np
import pandas as pd
import gzip

'''Data Viz'''
import matplotlib.pyplot as plt
import seaborn as sns
sns.set('notebook')
color = sns.color_palette()
import matplotlib.pylab as plt
from IPython.display import display, Javascript

%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

```


```python
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
```


```python

```

## Data

data source: [diabetes](https://www.kaggle.com/uciml/pima-indians-diabetes-database/data)


```python
name = 'datasets_228_482_diabetes.csv'

df = pd.read_csv(name)

df1 = df.copy()
```

Features of each instance in the dataset


```python
display(df1.columns)
```


    Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
           'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
          dtype='object')


Dimension of the dataset


```python
display(df1.shape)
```


    (768, 9)


We are dealing with small dataset according to machine learning standard, however we shall deal with this problem at a later stage.


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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


Dataset information:

The info() method is useful to get a quick description of the data


```python
display(df1.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 768 entries, 0 to 767
    Data columns (total 9 columns):
    Pregnancies                 768 non-null int64
    Glucose                     768 non-null int64
    BloodPressure               768 non-null int64
    SkinThickness               768 non-null int64
    Insulin                     768 non-null int64
    BMI                         768 non-null float64
    DiabetesPedigreeFunction    768 non-null float64
    Age                         768 non-null int64
    Outcome                     768 non-null int64
    dtypes: float64(2), int64(7)
    memory usage: 54.1 KB



    None


### Basic statistics:

Summary of each numerical attribute in dataset.


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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.845052</td>
      <td>120.894531</td>
      <td>69.105469</td>
      <td>20.536458</td>
      <td>79.799479</td>
      <td>31.992578</td>
      <td>0.471876</td>
      <td>33.240885</td>
      <td>0.348958</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.369578</td>
      <td>31.972618</td>
      <td>19.355807</td>
      <td>15.952218</td>
      <td>115.244002</td>
      <td>7.884160</td>
      <td>0.331329</td>
      <td>11.760232</td>
      <td>0.476951</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.078000</td>
      <td>21.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>99.000000</td>
      <td>62.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>27.300000</td>
      <td>0.243750</td>
      <td>24.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>117.000000</td>
      <td>72.000000</td>
      <td>23.000000</td>
      <td>30.500000</td>
      <td>32.000000</td>
      <td>0.372500</td>
      <td>29.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.000000</td>
      <td>140.250000</td>
      <td>80.000000</td>
      <td>32.000000</td>
      <td>127.250000</td>
      <td>36.600000</td>
      <td>0.626250</td>
      <td>41.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>17.000000</td>
      <td>199.000000</td>
      <td>122.000000</td>
      <td>99.000000</td>
      <td>846.000000</td>
      <td>67.100000</td>
      <td>2.420000</td>
      <td>81.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>


The count, mean, min, and max rows are self-explanatory.  Note that the null values are ignored, but we do not have to worry about that in this case -- both datasets have no null values. The std row shows the standard deviation, which measures how dispersed the values are. The 25%, 50%, and 75% rows show the corresponding percentiles: a percentile indicates the value below which a given percentage of observations in a group of observations falls

So let's look at the following attribute/features indicated above, minimum `glucose`, `blood pressure`, `skin thickness`, `insulin`, and `BMI` are all 0. This appears suspect because these are physical quantities that cannot be 0 for a live person! So we shall assume that all 0 in the above-mentioned features are as a result of lack of data.  To fill in these missing values, we will replace them with the median value in the column. There are other, more complicated methods for filling in missing values, but in practice, median imputation generally performs well.

## Imputing missing values


```python
df1['Glucose'] = df1['Glucose'].replace({0: df1['Glucose'].median()})
df1['BloodPressure'] = df1['BloodPressure'].replace({0: df1['BloodPressure'].median()})
df1['SkinThickness'] = df1['SkinThickness'].replace({0: df1['SkinThickness'].median()})
df1['Insulin'] = df1['Insulin'].replace({0: df1['Insulin'].median()})
df1['BMI'] = df1['BMI'].replace({0: df1['BMI'].median()})
```

Let's verify:


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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.845052</td>
      <td>121.656250</td>
      <td>72.386719</td>
      <td>27.334635</td>
      <td>94.652344</td>
      <td>32.450911</td>
      <td>0.471876</td>
      <td>33.240885</td>
      <td>0.348958</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.369578</td>
      <td>30.438286</td>
      <td>12.096642</td>
      <td>9.229014</td>
      <td>105.547598</td>
      <td>6.875366</td>
      <td>0.331329</td>
      <td>11.760232</td>
      <td>0.476951</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>44.000000</td>
      <td>24.000000</td>
      <td>7.000000</td>
      <td>14.000000</td>
      <td>18.200000</td>
      <td>0.078000</td>
      <td>21.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>99.750000</td>
      <td>64.000000</td>
      <td>23.000000</td>
      <td>30.500000</td>
      <td>27.500000</td>
      <td>0.243750</td>
      <td>24.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>117.000000</td>
      <td>72.000000</td>
      <td>23.000000</td>
      <td>31.250000</td>
      <td>32.000000</td>
      <td>0.372500</td>
      <td>29.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.000000</td>
      <td>140.250000</td>
      <td>80.000000</td>
      <td>32.000000</td>
      <td>127.250000</td>
      <td>36.600000</td>
      <td>0.626250</td>
      <td>41.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>17.000000</td>
      <td>199.000000</td>
      <td>122.000000</td>
      <td>99.000000</td>
      <td>846.000000</td>
      <td>67.100000</td>
      <td>2.420000</td>
      <td>81.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>


Great! Now we are good to continue :)

### Impute missing values


```python
display(df1.isnull().sum())
```


    Pregnancies                 0
    Glucose                     0
    BloodPressure               0
    SkinThickness               0
    Insulin                     0
    BMI                         0
    DiabetesPedigreeFunction    0
    Age                         0
    Outcome                     0
    dtype: int64


### Looking for Correlations:

Since the dataset is not too large, you can easily compute the standard correlation coefficient (also called Pearson’s r) between every pair of attributes using the corr() method:


```python
corr_matrix = df1.corr()

corr_matrix["Outcome"].sort_values(ascending=False)
```




    Outcome                     1.000000
    Glucose                     0.492782
    BMI                         0.312249
    Age                         0.238356
    Pregnancies                 0.221898
    SkinThickness               0.189065
    DiabetesPedigreeFunction    0.173844
    BloodPressure               0.165723
    Insulin                     0.148457
    Name: Outcome, dtype: float64



Note: the correlation coefficient ranges from –1 to 1. When it is close to 1, it means that there is a strong positive correlation.  When the coefficient is close to –1, it means that there is a strong negative correlation.  Finally, coefficients close to zero mean that there is no linear correlation

Glucose has the highest correlation value with the outcome, however none of the features are strongly correlated with the outcome and there are no negative correlations.

# 1. Exploratory Data Analasis

Moving on, we have few more numeric columns to explore, so we can use the `hist function` provided within pandas to save time.

A histogram for each numerical attribute


```python
df1.hist(figsize=(12,8))
plt.show()
```

![png](https://drive.google.com/uc?export=view&id=1-hjihmait1Vgf3fCEGSYQ3yBSd6OR6HU)


Let's analyse the result:

We can see that the feature `Outcome` is actually binary categorical features: It represent two possible values similar to gender: Male or Female. Therefore, this is actually categorical feature but already encoded as numeric column.

Mapping numerical feature (`Outcome`) to categorical feature.  O and 1 indicates patients that are non-diabetic and diabetic, respectively.  We are doing this bacause we want to clearly visualise numerical features.


```python
df1['Class'] = df1['Outcome'].map({1:'Diabetic', 0:'Non-diabetic'})
df1['Class'].unique()
```




    array(['Diabetic', 'Non-diabetic'], dtype=object)




```python

```

## data viz: pie plot


```python
df1['Class'].value_counts().plot(kind = 'pie', explode=(0,0.1), autopct='%1.2f%%',
                                         shadow = True, figsize = (8,6), legend = True, labels = None)

df1['Class'].value_counts()
```




    Non-diabetic    500
    Diabetic        268
    Name: Class, dtype: int64




![png](https://drive.google.com/uc?export=view&id=1BRE3ErNBom2TftY423wLRTbVivPb0P4o)

Note: `We see that we have huge disparity between diabetic and non-diabetic patience.`


```python

```

### Impute Missing Values


```python
display(df1.isnull().sum())
```


    Pregnancies                 0
    Glucose                     0
    BloodPressure               0
    SkinThickness               0
    Insulin                     0
    BMI                         0
    DiabetesPedigreeFunction    0
    Age                         0
    Outcome                     0
    Class                       0
    dtype: int64


There are no missing values in the dataset.

## Check correlation of features

The correlation coefficient only measures linear correlations.  Let's check for linearity amoung features.  So we can easily compute the standard correlation coefficient (also called Pearson’s r) between every pair of attributes using the corr() method:


```python
corr_matrix = df1.corr()

corr_matrix['Outcome'].sort_values(ascending=False)

plt.figure(figsize=(12,7))
corr_mat = sns.heatmap(corr_matrix, square=True, annot = True)
```


![png](https://drive.google.com/uc?export=view&id=1rh94IkHFRkRsYRJO0_cQnO--zY9AV072)


The correlation coefficient ranges from –1 to 1. When it is close to 1, it means that there is a strong positive correlation.  When the coefficient is close to –1, it means that there is a strong negative correlation.  Finally, coefficients close to zero mean that there is no linear correlation.

Info: We see noticeable correlation between, `Glucose` and `Outcome`.


```python

```

# 2. Preprocessing - getting data into shape

To determine whether our machine learning algorithm not only performs well on the training set but also generalizes well to new data, we also want to randomly divide the dataset into a separate `training` and `validation set`. `We use the training set to train and optimize our machine learning model`, while we keep the `tvalidation set until the very end to evaluate the final model`.


```python

```


```python
#feature matrix
x = df1.drop('Class', axis = 1)

#Target vector/labels
y = df1['Outcome']

print(x.shape)
print(y.shape)
```

    (768, 9)
    (768,)



```python
print('---------------------------------Feature Matrix--------------------------\n')
display(x.head(10))
print('----------------------------------Target Vector--------------------------\n')
display(y.head(10))
```

    ---------------------------------Feature Matrix--------------------------




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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148.0</td>
      <td>72.0</td>
      <td>35.0</td>
      <td>30.5</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85.0</td>
      <td>66.0</td>
      <td>29.0</td>
      <td>30.5</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183.0</td>
      <td>64.0</td>
      <td>23.0</td>
      <td>30.5</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89.0</td>
      <td>66.0</td>
      <td>23.0</td>
      <td>94.0</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137.0</td>
      <td>40.0</td>
      <td>35.0</td>
      <td>168.0</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>116.0</td>
      <td>74.0</td>
      <td>23.0</td>
      <td>30.5</td>
      <td>25.6</td>
      <td>0.201</td>
      <td>30</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3</td>
      <td>78.0</td>
      <td>50.0</td>
      <td>32.0</td>
      <td>88.0</td>
      <td>31.0</td>
      <td>0.248</td>
      <td>26</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10</td>
      <td>115.0</td>
      <td>72.0</td>
      <td>23.0</td>
      <td>30.5</td>
      <td>35.3</td>
      <td>0.134</td>
      <td>29</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2</td>
      <td>197.0</td>
      <td>70.0</td>
      <td>45.0</td>
      <td>543.0</td>
      <td>30.5</td>
      <td>0.158</td>
      <td>53</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>8</td>
      <td>125.0</td>
      <td>96.0</td>
      <td>23.0</td>
      <td>30.5</td>
      <td>32.0</td>
      <td>0.232</td>
      <td>54</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


    ----------------------------------Target Vector--------------------------




    0    1
    1    0
    2    1
    3    0
    4    1
    5    0
    6    1
    7    0
    8    1
    9    1
    Name: Outcome, dtype: int64


### Data Splicing

We are going to split the data into three sets -  80%, 20% for training, validation, respectively.  This means, 80% of the data will be set to train and optimize our machine learning model and 20% will be used to validate the model.


```python
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify = y)
```

Let's verify data partitioning:


```python
for dataset in [Y_train, Y_val]:

    print(round((len(dataset)/len(y))*100), '%')

print('>>>>>>>>>>>> Done!!<<<<<<<<<<<<<<')
```

    80 %
    20 %
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


```python

```

# 3. Machine Learning model

Now, the feature matrix and labels are ready for training. This is the part where different classifier will be instantiated (for training) and a model which performs better in both training and test set will be used for further predictions.

It is important to note that the parameters for the previously mentioned procedures, such as feature scaling are solely obtained from the training dataset, and the same parameters are later reapplied to transform the test dataset, as well as any new data samples—the performance measured on the test data may be overly optimistic otherwise.


```python
from sklearn import preprocessing
from sklearn.pipeline import Pipeline


#Logistic regression
print('-----------------------Logistic Regression Classifier---------------------\n')

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(random_state = 42)
LR.fit(X_train, Y_train)

print('Accuraccy in training set: {:.5f}' .format(LR.score(X_train, Y_train)))
print('Accuraccy in validation set: {:.5f}' .format(LR.score(X_val, Y_val)))


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier(random_state = 42)
DT.fit(X_train, Y_train)

print('\n-----------------------Decision Tree Classifier---------------------\n')
print('Accuraccy in training set: {:.5f}' .format(DT.score(X_train, Y_train)))
print('Accuraccy in validation set: {:.5f}' .format(DT.score(X_val, Y_val)))


#Random forest
from sklearn.ensemble import RandomForestClassifier

#instantiate random forest class
RF = RandomForestClassifier(random_state = 42)

RF.fit(X_train, Y_train)

print('\n-----------------------Random Forest Classifier---------------------\n')
print('Accuraccy in training set: {:.5f}' .format(RF.score(X_train, Y_train)))
print('Accuraccy in validation set: {:.5f}' .format(RF.score(X_val, Y_val)))



from sklearn.neighbors import KNeighborsClassifier
#KNN = Pipeline(steps=[('preprocessor', preprocessing.StandardScaler()),
#                     ('model', KNeighborsClassifier())])
KNN = KNeighborsClassifier(n_neighbors = 2)
KNN.fit(X_train_std, Y_train)

print('\n---------------------k-Nearest Neighbor Classifier------------------\n')
print('Accuraccy in training set: {:.5f}' .format(KNN.score(X_train, Y_train)))
print('Accuraccy in validation set: {:.5f}' .format(KNN.score(X_val, Y_val)))


print('\n---------------------C-Support Vector Classifier------------------\n')


from sklearn.svm import SVC #C-Support Vector Classification

svc_clf = SVC(kernel='linear') #instantiate C-Support Vector class

svc_clf.fit(X_train_std, Y_train) #training the model

print('Accuraccy in training set: {:.5f}' .format(svc_clf.score(X_train_std, Y_train)))
print('Accuraccy in validation set: {:.5f}' .format(svc_clf.score(X_val_std, Y_val)))


print('\n-------------------Gaussian Naive Bayes classifier------------------\n')

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train_std, Y_train)

print('Accuraccy in training set: {:.5f}' .format(gnb.score(X_train_std, Y_train)))
print('Accuraccy in validation set: {:.5f}' .format(gnb.score(X_val_std, Y_val)))


```

    -----------------------Logistic Regression Classifier---------------------

    Accuraccy in training set: 1.00000
    Accuraccy in validation set: 1.00000

    -----------------------Decision Tree Classifier---------------------

    Accuraccy in training set: 1.00000
    Accuraccy in validation set: 1.00000

    -----------------------Random Forest Classifier---------------------

    Accuraccy in training set: 1.00000
    Accuraccy in validation set: 1.00000

    ---------------------K-Nearest Neighbor Classifier------------------

    Accuraccy in training set: 0.51140
    Accuraccy in validation set: 0.50000

    ---------------------C-Support Vector Classifier------------------

    Accuraccy in training set: 1.00000
    Accuraccy in validation set: 1.00000

    -------------------Gaussian Naive Bayes classifier------------------

    Accuraccy in training set: 1.00000
    Accuraccy in validation set: 1.00000


We know that accuracy works well on balanced data.  The data is imbalanced, so we cannot use accuracy to quantify model performance. So we need another perfomance measure for imbalanced data.  We shall consider using `Confusion matrix`, `F1 score metric`, and `Receiver Operating Characteristics (ROC) Curve` to quantify the perfomance.

# 4. Model Evaluation - selecting optimal predictive model

We shall now show a `confusion matrix`, showing the frequency of misclassifications by our
classifier, to measure performance, `classification accuracy` which is defined as the proportion of correctly classified instances will be used, to see the performance of each classifier.  In this part, the `test set` will be used to evaluate each model performance


```python

```


```python
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

names = ['Non-diabetic', 'Diabetic']

pred1 = LR.predict(X_val)

mat1 = confusion_matrix(Y_val, pred1)

plt.figure(figsize = (15,8))
plt.subplot(2, 3, 1)
sns.set(font_scale = 1.2)
sns.heatmap(mat1, cbar = True, square = True, annot = True, yticklabels = names,
            annot_kws={'size': 15}, xticklabels = names, cmap = 'RdPu')
plt.title('Logistic Regression Classifier')
plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.tight_layout()


pred2 = DT.predict(X_val)

mat2 = confusion_matrix(Y_val, pred2)

plt.subplot(2, 3, 2)
sns.set(font_scale = 1.2)
sns.heatmap(mat2, cbar = True, square = True, annot = True, yticklabels = names,
            annot_kws={'size': 15}, xticklabels = names, cmap = 'RdPu')
plt.title('Decision Tree Classifier')
plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.tight_layout()


pred3 = RF.predict(X_val)

mat3 = confusion_matrix(Y_val, pred3)

plt.subplot(2, 3, 3)
sns.set(font_scale = 1.2)
sns.heatmap(mat3, cbar = True, square = True, annot = True, yticklabels = names,
            annot_kws={'size': 15}, xticklabels = names, cmap = 'RdPu')
plt.title('Random Forest Classifier')
plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.tight_layout()

pred4 = KNN.predict(X_val)
mat4 = confusion_matrix(Y_val, pred4)

plt.subplot(2, 3, 4)
sns.set(font_scale = 1.2)
sns.heatmap(mat4, cbar = True, square = True, annot = True, yticklabels = names,
            annot_kws={'size': 15}, xticklabels = names, cmap = 'RdPu')
plt.title('K-Nearest Neighbor Classifier')
plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.tight_layout()

pred5 = svc_clf.predict(X_val)
mat5 = confusion_matrix(Y_val, pred5)

plt.subplot(2, 3, 5)
sns.set(font_scale = 1.2)
sns.heatmap(mat5, cbar = True, square = True, annot = True, yticklabels = names,
            annot_kws={'size': 15}, xticklabels = names, cmap = 'RdPu')
plt.title('C-Support Vector Classifier')
plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.tight_layout()


pred6 = gnb.predict(X_val)
mat6 = confusion_matrix(Y_val, pred6)

plt.subplot(2, 3, 6)
sns.set(font_scale = 1.2)
sns.heatmap(mat6, cbar = True, square = True, annot = True, yticklabels = names,
            annot_kws={'size': 15}, xticklabels = names, cmap = 'RdPu')
plt.title('Gaussian Naive Bayes Classifier')
plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.tight_layout()


plt.show()
```

![png](https://drive.google.com/uc?export=view&id=1RJsm9twgko-gfkzExOSSgmj4g3yQ8Fc6)



```python

```

### Classification report


```python
print('\n\n ---------------------Logistic Regression model--------------------\n')
print(classification_report(Y_val, pred1, target_names = names))

print('\n\n --------------------Decision Tree model-------------------\n')
print(classification_report(Y_val, pred2, target_names = names))

print('\n\n -----------------------Random Forest model-------------------\n')
print(classification_report(Y_val, pred3, target_names = names))

print('\n\n --------------------K-Nearest Neighbor model-------------------\n')
print(classification_report(Y_val, pred4, target_names = names))

print('\n\n -----------------------C-Support Vector model-------------------\n')
print(classification_report(Y_val, pred5, target_names = names))

print('\n\n -----------------------Gaussian Naive Bayes model-------------------\n')
print(classification_report(Y_val, pred6, target_names = names))
```



     ---------------------Logistic Regression model--------------------

                  precision    recall  f1-score   support

    Non-diabetic       1.00      1.00      1.00       100
        Diabetic       1.00      1.00      1.00        54

       micro avg       1.00      1.00      1.00       154
       macro avg       1.00      1.00      1.00       154
    weighted avg       1.00      1.00      1.00       154



     --------------------Decision Tree model-------------------

                  precision    recall  f1-score   support

    Non-diabetic       1.00      1.00      1.00       100
        Diabetic       1.00      1.00      1.00        54

       micro avg       1.00      1.00      1.00       154
       macro avg       1.00      1.00      1.00       154
    weighted avg       1.00      1.00      1.00       154



     -----------------------Random Forest model-------------------

                  precision    recall  f1-score   support

    Non-diabetic       1.00      1.00      1.00       100
        Diabetic       1.00      1.00      1.00        54

       micro avg       1.00      1.00      1.00       154
       macro avg       1.00      1.00      1.00       154
    weighted avg       1.00      1.00      1.00       154



     --------------------K-Nearest Neighbor model-------------------

                  precision    recall  f1-score   support

    Non-diabetic       0.66      0.47      0.55       100
        Diabetic       0.36      0.56      0.44        54

       micro avg       0.50      0.50      0.50       154
       macro avg       0.51      0.51      0.49       154
    weighted avg       0.56      0.50      0.51       154



     -----------------------C-Support Vector model-------------------

                  precision    recall  f1-score   support

    Non-diabetic       1.00      1.00      1.00       100
        Diabetic       1.00      1.00      1.00        54

       micro avg       1.00      1.00      1.00       154
       macro avg       1.00      1.00      1.00       154
    weighted avg       1.00      1.00      1.00       154



     -----------------------Gaussian Naive Bayes model-------------------

                  precision    recall  f1-score   support

    Non-diabetic       1.00      1.00      1.00       100
        Diabetic       1.00      1.00      1.00        54

       micro avg       1.00      1.00      1.00       154
       macro avg       1.00      1.00      1.00       154
    weighted avg       1.00      1.00      1.00       154



## Receiver Operating Characteriscics (ROC) Curve


```python
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0,1], linestyle = '--', color = 'r')                                    
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) # Not shown
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    # Not shown
    plt.grid(True)   
    plt.tight_layout()
```


```python
from sklearn.metrics import f1_score, roc_auc_score, roc_curve


#predicting class probabilities for input x_test
probs1 = LR.predict_proba(X_val)[:, 1]
#Calculate the area under the roc curve
auc1 = roc_auc_score(Y_val, probs1)

plt.style.use('bmh')
plt.figure(figsize = (12,8))
plt.subplot(2, 2, 1)

# Plot the roc curve
fpr1, tpr1, thresholds1 = roc_curve(Y_val, probs1)

plot_roc_curve(fpr1, tpr1)
plt.title('Logistic Regression - ROC Curve, \n AUC = %0.4f' % auc1, size = 18)

#predicting class probabilities for input x_test
probs2 = DT.predict_proba(X_val)[:, 1]
#Calculate the area under the roc curve
auc2 = roc_auc_score(Y_val, probs2)
fpr2, tpr2, thresholds2 = roc_curve(Y_val, probs2)


plt.subplot(2, 2, 2)
# Plot the roc curve
plot_roc_curve(fpr2, tpr2)
plt.title('Decision Tree - ROC Curve, \n AUC = %0.4f' % auc2, size = 18)


#predicting class probabilities for input x_test
probs3 = RF.predict_proba(X_val)[:, 1]
#Calculate the area under the roc curve
auc3 = roc_auc_score(Y_val, probs3)

fpr3, tpr3, thresholds3 = roc_curve(Y_val, probs3)


plt.subplot(2, 2, 3)
plot_roc_curve(fpr3, tpr3)
plt.title('Random Forest - ROC Curve, \n AUC = %0.4f' % auc3, size = 18)

#predicting class probabilities for input x_test
probs4 = KNN.predict_proba(X_val)[:, 1]
#Calculate the area under the roc curve
auc4 = roc_auc_score(Y_val, probs4)

fpr4, tpr4, thresholds4 = roc_curve(Y_val, probs4)

# Plot the roc curve
plt.subplot(2, 2, 4)

plot_roc_curve(fpr4, tpr4)
plt.title('K-Nearest Neighbor - ROC Curve, \n AUC = %0.4f' % auc4, size = 18)


```




    Text(0.5, 1.0, 'K-Nearest Neighbor - ROC Curve, \n AUC = 0.5128')




![png](https://drive.google.com/uc?export=view&id=1wQl5F4-tY_lqCZjQpAMkwwGUWf5Q5gx7)


Point to note: K-Nearest Neighbor (KNN) Classiffier performed pooly.  So we shall subject all models to parameter tuning to further evaluate them.  We want to ensure that we get a model which doesn't overfit or underfit.


```python

```

### Fine-Tune Models



```python
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score

```

#### Randomized search

We shall use `Randomized Search`.  `RandomizedSearchCV` class can be used in much the same way as the `GridSearchCV` class, but instead of trying out all possible combinations, it evaluates a given number of random combinations by selecting a random value for each hyperparameter at every iteration.

Note:  We shall use 5-fold cross-validation mechanisms to resample the data to produce multiple datasets, that is, for a given hyperparameter setting, each of the 5 folds takes turns being the hold-out validation set; a model is trained on the rest of the 5 – 1 folds and measured on the held-out fold.  The purpose performing cross-validation is because we have a `samll dataset` by machine learning standard


```python
print("\033[1m"+'Decision Tree Classifier params:'+"\033[10m")
display(DT.get_params)


params11 = {"max_depth": [3, 32, None], "min_samples_leaf": [np.random.randint(1,9)],
               "criterion": ["gini","entropy"], "max_features": [2, 4, 6] }


DT_search = RandomizedSearchCV(DT, param_distributions=params11, random_state=42,
                               n_iter = 200, cv = 5, verbose = 1, n_jobs = -1, return_train_score=True)
DT_search.fit(X_train, Y_train)

print('\n-----------------------Decision Tree Classifier---------------------\n')

print('Accuracy in training set: {:.5f}' .format(accuracy_score(Y_train, DT.predict(X_train))))
print('Accuracy in validation set: {:.5f}' .format(accuracy_score(Y_val, DT.predict(X_val))))
print('Accuracy in training set(hyperparameter tunning): {:.5f}' .format(accuracy_score(Y_train, DT_search.predict(X_train))))
print('Accuracy in validation set(hyperparameter tunning): {:.5f}' .format(accuracy_score(Y_val, DT_search.predict(X_val))))
```

    [1mDecision Tree Classifier params:[10m



    <bound method BaseEstimator.get_params of DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=42,
                splitter='best')>


    Fitting 5 folds for each of 18 candidates, totalling 90 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:    8.9s



    -----------------------Decision Tree Classifier---------------------

    Accuracy in training set: 1.00000
    Accuracy in validation set: 1.00000
    Accuracy in training set(hyperparameter tunning): 1.00000
    Accuracy in validation set(hyperparameter tunning): 1.00000


    [Parallel(n_jobs=-1)]: Done  90 out of  90 | elapsed:    9.6s finished



```python

```


```python
#parameter settings
print("\033[1m"+'Random Forest Classifier params:'+"\033[10m")
display(RF.get_params)


params22 = {"max_depth": [3, 32, None],'n_estimators': [3, 10, 12, 15, 20, 30], 'max_features': [2, 4, 6, 8],
           'bootstrap': [True, False], 'n_estimators': [3, 10, 15, 20, 30], 'max_features': [2, 3, 4, 8],
           "criterion": ["gini","entropy"]}


RF_search = RandomizedSearchCV(RF, param_distributions=params22, random_state=42,
                               n_iter = 100, cv = 5, verbose = 1, n_jobs = -1, return_train_score=True)

RF_search.fit(X_train, Y_train)


print('\n-----------------------Random Forest Classifier---------------------\n')

print('Accuracy in training set: {:.5f}' .format(accuracy_score(Y_train, RF.predict(X_train))))
print('Accuracy in validation set: {:.5f}' .format(accuracy_score(Y_val, RF.predict(X_val))))
print('Accuracy in training set(hyperparameter tunning): {:.5f}' .format(accuracy_score(Y_train, RF_search.predict(X_train))))
print('Accuracy in validation set(hyperparameter tunning): {:.5f}' .format(accuracy_score(Y_val, RF_search.predict(X_val))))

accuracy_score(Y_val, RF_search.predict(X_val))
```

    [1mRandom Forest Classifier params:[10m



    <bound method BaseEstimator.get_params of RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,
                oob_score=False, random_state=42, verbose=0, warm_start=False)>


    Fitting 5 folds for each of 100 candidates, totalling 500 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:    3.6s
    [Parallel(n_jobs=-1)]: Done 192 tasks      | elapsed:   10.7s
    [Parallel(n_jobs=-1)]: Done 442 tasks      | elapsed:   23.9s



    -----------------------Random Forest Classifier---------------------

    Accuracy in training set: 1.00000
    Accuracy in validation set: 1.00000
    Accuracy in training set(hyperparameter tunning): 1.00000
    Accuracy in validation set(hyperparameter tunning): 1.00000


    [Parallel(n_jobs=-1)]: Done 500 out of 500 | elapsed:   26.6s finished





    1.0



**Warning**: the next cell may take time to run, depending on your hardware.


```python
#parameter settings
params33 = {'weights':('uniform', 'distance'), 'n_neighbors':[1,2]}


KNN_search = RandomizedSearchCV(KNN, param_distributions=params33, random_state=42,
                                    cv = 5, return_train_score=True)

KNN_search.fit(X_train_std, Y_train)


print('\n-----------------------K-Nearest Neighbor Classifier---------------------\n')

print('Accuracy in training set: {:.5f}' .format(accuracy_score(Y_train, KNN.predict(X_train_std))))
print('Accuracy in validation set: {:.5f}' .format(accuracy_score(Y_val, KNN.predict(X_val_std))))
print('Accuracy in training set(hyperparameter tunning): {:.5f}' .format(accuracy_score(Y_train, KNN_search.predict(X_train_std))))
print('Accuracy in validation set(hyperparameter tunning): {:.5f}' .format(accuracy_score(Y_val, KNN_search.predict(X_val_std))))

accuracy_score(Y_val, KNN_search.predict(X_val_std))

```


    -----------------------k-Nearest Neighbor Classifier---------------------

    Accuracy in training set: 0.99837
    Accuracy in validation set: 1.00000
    Accuracy in training set(hyperparameter tunning): 1.00000
    Accuracy in validation set(hyperparameter tunning): 1.00000





    1.0



Note: there seemed to be a major improvement in kNN classifier after parameter tuning.

**Warning**: the next cell may also take time to run, depending on your hardware.


```python
#parameter settings
params44 = {'kernel':('linear', 'rbf', 'poly'), 'C':[0.1, 1, 10, 100, 1000, 1E4, 1E10], 'gamma':[0.1, 1, 10, 100, 1000]}

svc_clf_search = RandomizedSearchCV(svc_clf, param_distributions=params44, random_state=42,
                                    cv = 5, verbose=1, scoring='neg_mean_squared_error', return_train_score=True)

svc_clf_search.fit(X_train_std, Y_train)


print('\n-----------------------Support Vector Machine Classifier---------------------')

print('Accuracy in training set: {:.5f}' .format(accuracy_score(Y_train, svc_clf.predict(X_train_std))))
print('Accuracy in validation set: {:.5f}' .format(accuracy_score(Y_val, svc_clf.predict(X_val_std))))
print('Accuracy in training set(hyperparameter tunning): {:.5f}' .format(accuracy_score(Y_train, svc_clf_search.predict(X_train_std))))
print('Accuracy in validation set(hyperparameter tunning): {:.5f}' .format(accuracy_score(Y_val, svc_clf_search.predict(X_val_std))))

accuracy_score(Y_val, svc_clf_search.predict(X_val_std))
```

    Fitting 5 folds for each of 10 candidates, totalling 50 fits


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.



    -----------------------Support Vector Machine Classifier---------------------
    Accuracy in training set: 1.00000
    Accuracy in validation set: 1.00000
    Accuracy in training set(hyperparameter tunning): 1.00000
    Accuracy in validation set(hyperparameter tunning): 1.00000


    [Parallel(n_jobs=1)]: Done  50 out of  50 | elapsed:    0.7s finished





    1.0



### Confusion matrix (after parameter tuning)


```python

pred2 = DT_search.predict(X_val)

mat22 = confusion_matrix(Y_val, pred2)

plt.figure(figsize = (12,8))
plt.subplot(2, 2, 1)
sns.set(font_scale = 1.2)
sns.heatmap(mat22, cbar = True, square = True, annot = True, yticklabels = names,
            annot_kws={'size': 15}, xticklabels = names, cmap = 'RdPu')
plt.title('Decision Tree Classifier')
plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.tight_layout()


pred33 = RF_search.predict(X_val)

mat33 = confusion_matrix(Y_val, pred33)

plt.subplot(2, 2, 2)
sns.set(font_scale = 1.2)
sns.heatmap(mat33, cbar = True, square = True, annot = True, yticklabels = names,
            annot_kws={'size': 15}, xticklabels = names, cmap = 'RdPu')
plt.title('Random Forest Classifier')
plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.tight_layout()

pred44 = KNN_search.predict(X_val)
mat44 = confusion_matrix(Y_val, pred44)

plt.subplot(2, 2, 3)
sns.set(font_scale = 1.2)
sns.heatmap(mat44, cbar = True, square = True, annot = True, yticklabels = names,
            annot_kws={'size': 15}, xticklabels = names, cmap = 'RdPu')
plt.title('K-Nearest Neighbor Classifier')
plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.tight_layout()

pred55 = svc_clf_search.predict(X_val)
mat55 = confusion_matrix(Y_val, pred55)

plt.subplot(2, 2, 4)
sns.set(font_scale = 1.2)
sns.heatmap(mat55, cbar = True, square = True, annot = True, yticklabels = names,
            annot_kws={'size': 15}, xticklabels = names, cmap = 'RdPu')
plt.title('C-Support Vector Classifier')
plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.tight_layout()



```

![png](https://drive.google.com/uc?export=view&id=1ax_62EKGTrQ23IETZ5ldIluT9oi25ffn)


### Classification report (after parameter tuning)


```python
print('\n\n --------------------Decision Tree model-------------------\n')
print(classification_report(Y_val, pred22, target_names = names))

print('\n\n -----------------------Random Forest model-------------------\n')
print(classification_report(Y_val, pred33, target_names = names))

print('\n\n --------------------k-Nearest Neighbor model-------------------\n')
print(classification_report(Y_val, pred44, target_names = names))

print('\n\n -----------------------C-Support Vector model-------------------\n')
print(classification_report(Y_val, pred55, target_names = names))


```



     --------------------Decision Tree model-------------------

                  precision    recall  f1-score   support

    Non-diabetic       1.00      1.00      1.00       100
        Diabetic       1.00      1.00      1.00        54

       micro avg       1.00      1.00      1.00       154
       macro avg       1.00      1.00      1.00       154
    weighted avg       1.00      1.00      1.00       154



     -----------------------Random Forest model-------------------

                  precision    recall  f1-score   support

    Non-diabetic       1.00      1.00      1.00       100
        Diabetic       1.00      1.00      1.00        54

       micro avg       1.00      1.00      1.00       154
       macro avg       1.00      1.00      1.00       154
    weighted avg       1.00      1.00      1.00       154



     --------------------K-Nearest Neighbor model-------------------

                  precision    recall  f1-score   support

    Non-diabetic       0.70      0.46      0.55       100
        Diabetic       0.39      0.63      0.48        54

       micro avg       0.52      0.52      0.52       154
       macro avg       0.54      0.54      0.52       154
    weighted avg       0.59      0.52      0.53       154



     -----------------------C-Support Vector model-------------------

                  precision    recall  f1-score   support

    Non-diabetic       1.00      1.00      1.00       100
        Diabetic       1.00      1.00      1.00        54

       micro avg       1.00      1.00      1.00       154
       macro avg       1.00      1.00      1.00       154
    weighted avg       1.00      1.00      1.00       154




```python

```

### ROC Curve (after parameter tunning)


```python
#predicting class probabilities for input x_test


plt.style.use('bmh')
plt.figure(figsize = (12,8))

#predicting class probabilities for input x_test
probs22 = DT_search.predict_proba(X_val)[:, 1]
#Calculate the area under the roc curve
auc22 = roc_auc_score(Y_val, probs22)
fpr22, tpr22, thresholds22 = roc_curve(Y_val, probs22)


plt.subplot(2, 2, 1)
# Plot the roc curve
plot_roc_curve(fpr2, tpr2)
plt.title('Decision Tree - ROC Curve, \n AUC = %0.4f' % auc22, size = 18)


#predicting class probabilities for input x_test
probs33 = RF.predict_proba(X_val)[:, 1]
#Calculate the area under the roc curve
auc33 = roc_auc_score(Y_val, probs33)

fpr33, tpr33, thresholds33 = roc_curve(Y_val, probs33)


plt.subplot(2, 2, 2)
plot_roc_curve(fpr33, tpr33)
plt.title('Random Forest - ROC Curve, \n AUC = %0.4f' % auc33, size = 18)

#predicting class probabilities for input x_test
probs44 = KNN_search.predict_proba(X_val)[:, 1]
#Calculate the area under the roc curve
auc44 = roc_auc_score(Y_val, probs44)

fpr44, tpr44, thresholds44 = roc_curve(Y_val, probs44)

# Plot the roc curve
plt.subplot(2, 2, 3)

plot_roc_curve(fpr44, tpr44)
plt.title('K-Nearest Neighbor - ROC Curve, \n AUC = %0.4f' % auc44, size = 18)
plt.tight_layout()


```

![png](https://drive.google.com/uc?export=view&id=1N_gWh0hLBIMg4TKoxvbzE5BnGG_jS0-6)


```python

```

### Summary \& Conclusion



In exploratory data analysis, the dataset unveil two aspects.  Firstly, imputing missing values in several columns was necessary because they were physically impossible.  Then the median imputation was employed as an effective method for filling impossible values (0s for BloodPressure).  Secondly, slightly positive correlation between features and response were established, despite the fact that they were not strong.  There were no feature to engineer was not necessary either since the number of observations (768) outnumbers the number of features (8), which dramatically reduces the chances of overfitting.

A model is a simplified version of the observations. The simplification are meant to discard the superfluous details that are unlikely to generalise to new instances. For example, a linear model makes assumption that the data is fundamentally linear and that the distance between instances and the straight line is just noise, which can safely be ignored.

Suprisingly all models performed better (~ 100%), except k-Nearest Neighbor (kNN), both in training and validation stage as indicated by diffent evaluation metric (Confusion matrix, F1 Score, and ROC curve).  However, we have performed parameter tunning in some models (Decision Tree, Random Forest, kNN, Support Vector Machine) and the results didn't change significantly.  kNN performed very poorly as compared to other models and it should not be considered for making predictions.  However, we acknowledge the fact that our dataset was small and caution was taken to account for that (i.e. k-fold cross validation was performed to resample the dataset).

So to choose the best model in this case, one has to consider number of factors:

Parametric machine learning algorithms used in this work: `Logistic Regression` and `Naive Bayes` models are very fast to learn from data.  They do not require as much training data and can work well even if the fit to the data is not perfect. Nonparametric machine learning algorithms used in this work, `k-Nearest Neighbors`, `Decision Trees`, `Random Forest` and `Support Vector Machines` requires a lot more training data to estimate the mapping function and are susceptible to overfit the training data and it is harder to explain why specific predictions are made.

So we can see that all models have their prons and cons.  However, in this work ensemble methods (Decision Tree & Random Forest) will be prefered.  At this point, any model (except kNN) the reader may chose will perform just fine!




```python

```
