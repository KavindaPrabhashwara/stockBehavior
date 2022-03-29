# Behavior of Asian Stock Markets in COVID-19 wrt GFC

## I. LOAD THE DATA
In this section you will:

- Import the libraries
- Creating the dataset
- Load the dataset

### A) Import the Libraries


```python
# Import libraries 

# Data Manipulation
import numpy as np 
import pandas as pd
from   pandas import DataFrame

# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# Directoy handling
import os, sys 

# Statistics
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from scipy.interpolate import interp1d

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from dateutil.parser import parse 
import pickle
```


```python
# Set the options for plotting
pd.set_option('display.max_rows', 800)
pd.set_option('display.max_columns', 500)
%matplotlib inline
plt.rcParams.update({'figure.figsize': (16, 5), 'figure.dpi': 300})
plt.style.use('seaborn-whitegrid')
```

### B) Creating the Dataset


```python
def symbolToPath(symbol, baseDir = 'Data'):
    """Return XLSX file path given ticker symbol"""
    ## to get the CSV file path change xlsx to csv
    return os.path.join(baseDir, "{}.xlsx".format(str(symbol)))

def getData(symbols, dates):
    """Read stock data (Price) for given symbols from CSV files """
    df = pd.DataFrame(index = dates)
    if 'Sri Lanka FX' not in symbols:
        # add Sri Lanka for reference, if absent
        symbols.insert(0, 'Sri Lanka FX')
        
    for symbol in symbols:
        dfTemp = pd.read_excel(symbolToPath(symbol),
                                index_col = 'Date',
                                parse_dates = True,
                                usecols = ['Date', 'Price'],
                                na_values =['nan']
                            )
        dfTemp = dfTemp.rename(columns = {'Price': symbol})
        df  = df.join(dfTemp)
        #if symbol == 'SriLanka':
            # drop dates SL CSE did not trade
            #df = df.dropna(subset = ["SriLanka"])
            
    return df

def plotData(df):
    """Plot stock prices"""
    ax = df.plot(title = "Stock Prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show()

def test_run():
    # Define a date range
    dates = pd.date_range('2001-07-01', '2021-12-31')
    
    # Choose symbols to read
    # Sri Lanka FX will be added in getData
    indices = ["Sri Lanka FX","Sri Lanka",
               "China FX","China", 
               "India FX","India", 
               "Indonesia FX", "Indonesia",
               "Israel FX", "Israel",
               "Japan FX", "Japan",
               "Jordhan FX", "Jordhan",
               "Lebanon FX", "Lebanon",
               "Malaysia FX", "Malaysia",
               "Nepal FX", "Nepal",
               "Pakistan FX", "Pakistan",
               "Phillipines FX", "Phillipines",
               "Saudi Arabia FX", "Saudi Arabia",
               "Thailand FX", "Thailand",
               "Turkey FX", "Turkey",
               "UAE FX","UAE",
               "Vietnam FX", "Vietnam",
              "WTI_USD", "XAU_USD"] 
    
    # get stock data
    df = getData(indices, dates)
    
    # drop rows that contains missing values in all columns
    df =df.dropna(how="all")
    
    # first doing the forward filling then the backward filling to remove NA
    df = df.fillna(method="ffill")
    df = df.fillna(method="bfill")
       
    # divide all the data by 1st row to standardize
    #df.div(df.iloc[0])
    
    #save the file as a CSV
    df.to_csv('/Data_01072001_01012022.csv')
    
    # get the head details
    #print (df.head(10))
    #print(df.describe())
    
    print (df.tail(10))
    
    
    # Slice and plot
    # plotData(df)  
```

### C) Loading the Dataset


```python
# Loading the dataset
# Here the dataset is present in the same level as notebook
df = pd.read_csv('./Data_01072001_01012022.csv' ,index_col="Date",parse_dates=True)
df.head()
```


```python
# Getting only the stock indices
stockIndices = ["Sri Lanka",
             "China", 
             "India", 
             "Indonesia", 
             "Israel",
             "Japan", 
             "Jordhan", 
             "Lebanon", 
             "Malaysia", 
             "Nepal", 
             "Pakistan", 
             "Phillipines",
             "Saudi Arabia", 
             "Thailand",
             "Turkey",
             "UAE",
              "Vietnam"]

# Getting all the indices without WTI and Gold (XAU)
indices = ["Sri Lanka FX","Sri Lanka",
               "China FX","China", 
               "India FX","India", 
               "Indonesia FX", "Indonesia",
               "Israel FX", "Israel",
               "Japan FX", "Japan",
               "Jordhan FX", "Jordhan",
               "Lebanon FX", "Lebanon",
               "Malaysia FX", "Malaysia",
               "Nepal FX", "Nepal",
               "Pakistan FX", "Pakistan",
               "Phillipines FX", "Phillipines",
               "Saudi Arabia FX", "Saudi Arabia",
               "Thailand FX", "Thailand",
               "Turkey FX", "Turkey",
               "UAE FX","UAE",
               "Vietnam FX", "Vietnam"] 
```

## II. OVERVIEW OF THE DATA
- Get the descriptive statistics of the data
- Get the information about missing values in the data

### A) Descriptive Statistics

As the name says descriptive statistics describes the data. It gives you information about
- Mean, median, mode 
- Min, max
- Skewness, Kurtosis, Jarque-berra
- Count etc


```python
# Dimension of the data
df.shape
```


```python
# Summary of the dataset
df.describe()
```

#### a) Jarque-Bera Test


```python
# Importing Jarque-bera test from Scipy
from scipy.stats import jarque_bera

# function to check Jarque-Bera
def jarque_bera_test(country,start_date,end_date):
    """Setting a function to get the Jarque-Bera Statistics for a index in a given period of time"""
    
    # creating a temporary dataframe to store data of country, start and end dates
    dfTemp = df.loc[start_date:end_date,country]
    
    # Assigning the value of Jarque Bera test to a variable
    result = (jarque_bera(dfTemp))
    
    # Getting the test results
    print (country)
    print(f"JB statistic: {result[0]}")
    print(f"p-value: {result[1]}\n")
```


```python
# edit the parameters (country, start date, end date) to get the Jarque-Bera statistics
for index in indices:
    jarque_bera_test(country,start_date,end_date)
```

#### b) Skewness


```python
# function to get Skewness values
def skewnessTest(index, start_date,end_date):
    """Getting the skewness test score for a given date range and the index"""
    dfTemp = df.loc[start_date:end_date, index]
    print (dfTemp.skew(axis=0))
```


```python
# change the parameters countries, start date and end date to get skewness
skewnessTest(countries, start_date,end_date)
```

#### c) Kurtosis


```python
# function to get Kurtosis values
def kurtosisTest(index, start_date,end_date):
    """Getting the kurtosis for a given date range and the index"""
    dfTemp = df.loc[start_date:end_date, index]
    print (dfTemp.kurt(axis=0))
```


```python
# change the parameters countries, start date and end date to get Kurtosis
kurtosisTest(countries, start_date,end_date)
```

### B) Missing Values


```python
# Missing values for every column
df.isna().sum()
```

## III. EXPLORATORY DATA ANALYSIS
Exploratory data analysis is an approach to analyze or investigate data sets to find out patterns in the data. Visual methods are often used to summarise the data. Primarily EDA is for seeing what the data can tell us beyond the formal modeling or hypothesis testing tasks.

In this section:
- Visualize the time series using line plot
- Check distribution of the time series
- Check monthly seasonality using multiple lines. Seasonality may be differ for each series.
- Check monthly seasonality and yearly trend using box plot


```python
# Creating a new dataframe without an index
dfTemp = df.reset_index()
dfTemp.head()
```


```python
# Setting date in datetime format
dfTemp['Date'] = pd.to_datetime(dfTemp['Date'])

# Target class name
input_target_variable = "Sri Lanka"

# Date column name
input_date_variable = 'Date'

# Exogenous variable
input_exogenous_variable = 'month_no'

# Seasonality
input_seasonality = 12
input_order = (0, 1 , 2)
input_seasonal_order = (2, 1, 0, input_seasonality)
```


```python
# Prepare data
dfTemp['year'] = [d.year for d in dfTemp[input_date_variable]]
dfTemp['month'] = [d.strftime('%b') for d in dfTemp[input_date_variable]]
years = dfTemp['year'].unique()
```


```python
# Prep Colors
np.random.seed(100)
mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(years), replace=False)
```

### A) Line Plots


```python
# Draw line Plot
def plot_df(x, y, title="", xlabel='Date', dpi=100):
    """Function to get line plots of the data"""
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(xlabel=xlabel)
```


```python
for index in indices:
    plot_df(x=dfTemp[input_date_variable], y=dfTemp[symbol])
    plt.title('Proportional Change of {}'.format(index))
    plt.ylabel('Proportional Change')
    # plt.savefig('Proportional Change of Closing Price of {}.png'.format(index),dpi=300)
    plt.show()
```


```python
# list of colors
color = ['#FA8406','#0696FA','#FA0606','#0ABC01']
```


```python
for index in range(0, len(indices), 2):
    # Plotting Stock, FOREX, WTI and GOLD indices 
    s = df[indices[index:index+2]+['WTI_USD','XAU_USD']].plot(color=color)
    plt.title('Proportional Change of Markets from 2001-07-01 to 2021-12-31')
    plt.ylabel('Proportional Change')
    # plt.savefig('Proportional Change of Closing Price {}.png'.format(indices[index]),dpi=300)
    plt.show()
```

### B) Plot Distribution


```python
# function to plot the distribution
def plot():
    '''Function to plot the Distribution of the data'''
    for index in indices:
        sns.distplot(df[index], kde = False, color ='blue', bins = 20)
        plt.title('Distribution of the {}'.format(index), fontsize=16)
        plt.ylabel('Count')
        plt.show()
        
if __name__ == "__main__":
    plot()
```

### C) Seasonality and Trend


```python
def plotTrendandSeasonality(index):
    '''Funtion to plot the Trend and Seasonality'''
    fig, axes = plt.subplots(1, 2, figsize=(20,7), dpi= 200)
    sns.boxplot(x='year', y=index, data=df, ax=axes[0])
    sns.boxplot(x='month', y=index, data=df.loc[~df.year.isin([2000, 2021]), :])

    # Set Title
    axes[0].set_title('Yearly Trend of {}'.format(index), fontsize=18)
    axes[0].tick_params(axis='x', rotation=45)
    axes[1].set_title('Monthly Seasonality of {}'.format(index), fontsize=18)
    plt.xticks(rotation=45)
    plt.show()
```


```python
# change the parameter index to plot the Seasonality and trend
plotTrendandSeasonality(index)
```

## D) Decomposition


```python
def decompositions(index):
    """Function to plot the Decomposition"""
    # Multiplicative Decomposition 
    result_mul = seasonal_decompose(df[index], model='multiplicative', period=input_seasonality)

    # Additive Decomposition
    result_add = seasonal_decompose(df[index], model='additive', period=input_seasonality)

    # Plot
    plt.rcParams.update({'figure.figsize': (10,8)})
    result_mul.plot().suptitle('Multiplicative Decomposition of {}'.format(index), fontsize=16,y=1.02)
    result_add.plot().suptitle('Additive Decomposition of {}'.format(index), fontsize=16,y=1.02)
    plt.show()
```


```python
# change the parameter index to plot the Decompositions
decompositions(index)
```

# IV. CHANGE POINT DETECTION
Change points in time series data are abrupt shifts in the data. Transitions between states may be represented by such sudden changes. Changepoint detection is useful in time series modelling and prediction, and it's used in things like medical condition monitoring, climatic change detection, voice and image analysis, and human activity analysis 


```python
# function to plot data
def plot_data(dfTemp, country,start_date, end_date, breaks_rpt,model_name='', legend=True):
    """Plotting the data of change points"""
    plt.plot(dfTemp, label='Data')
    plt.title('{} Change Point Detection of {} from {} to {}'.format(model_name, country,start_date, end_date))
    plt.xlabel('Dates')
    plt.ylabel('Proportional Change')

    for i in breaks_rpt:
        if legend:
            plt.axvline(i, color='red',linestyle='dashed', label='breaks')
            legend = False
        else:
            plt.axvline(i, color='red',linestyle='dashed')
    plt.grid()
    plt.legend()
    plt.savefig('{} Change Point Detection of {} from {} to {}.png'.format(model_name, country,start_date, end_date),dpi=300)
    plt.show()
```


```python
#function to get the model and do change point analysis
def algo(country, start_date, end_date,num_pens, model_name='Binary Segementation'):
    """Getting the change points of the necessary date range and of the country. If a model is explicitly not given it will use Binary Segementation"""
    # Slice the dataframe to get the required dates
    dfTemp = df.loc[start_date: end_date, country]
    y = np.array(dfTemp.tolist())

    # Selecting the required model
    # If number of change points are known beforehand go for Dynp, Window or Binary
    # If number of change points are unknown beforehand go for Pelt
    if model_name=='Pelt':
        model = "rbf"
        model = rpt.Pelt(model= model) 
    elif model_name=='Window Search':
        model = "rbf" 
        model = rpt.Window(width=40, model=model)
    elif model_name=='Dynamic Programming_L1':
        model = "l1" 
        model = rpt.Dynp(model=model, min_size=3, jump=5)
    elif model_name=='Dynamic Programming_L2':
        model = "l2" 
        model = rpt.Dynp(model=model, min_size=3, jump=5)
    else:
        model = "l2" 
        model = rpt.Binseg(model=model)
        
    # fitting the model    
    model.fit(y)
    
    # Keeping pen low for Pelt increase accuracy
    # Keeping n_bkps high for other search algorithms increases accuracy
    pens = num_pens
    for pen in pens:
        if model_name=='Pelt':
            breaks = model.predict(pen=pen)
            breaks_rpt = pd.to_datetime(dfTemp.index[np.array(breaks)-1])
        else:
            breaks = model.predict(n_bkps=pen)
            breaks_rpt = pd.to_datetime(dfTemp.index[np.array(breaks)-1])
        breaks_rpt = pd.to_datetime(dfTemp.index[np.array(breaks)-1])
        plot_data(dfTemp, country,start_date, end_date, breaks_rpt, model_name=model_name, legend=True)
        print (breaks_rpt)
```


```python
# change the parameter index, Start date, End date, Model name to do the change point analysis
algo(index, startDate, endDate, modelName)
```

# V. PREDICTION MODEL


```python
# importing necessary libraries for prediction modellin
import math
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

## A) Creating the Model


```python
def stockPrediction(index):
  #splitting train and test set
  training_set = df.loc[:6000, index].values.reshape(-1,1)
  test_set = df.loc[6000:, index].values.reshape(-1,1)

  # using min max scaler
  sc = MinMaxScaler(feature_range = (0, 1))
  training_set_scaled = sc.fit_transform(training_set)

  # Creating a data structure with 60 time-steps and 1 output
  X_train = []
  y_train = []
  for i in range(36, 6000):
      X_train.append(training_set_scaled[i-36:i, 0])
      y_train.append(training_set_scaled[i, 0])
  X_train, y_train = np.array(X_train), np.array(y_train)
  X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

  # Define Sequential model with 9 layers
  model = keras.Sequential(
      [
      layers.GRU(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)),
      layers.Dropout(0.2),
      layers.GRU(units = 50, return_sequences = True),
      layers.Dropout(0.2),
      layers.GRU(units = 50, return_sequences = True),
      layers.Dropout(0.2),
      layers.GRU(units = 50),
      layers.Dropout(0.2),
      layers.Dense(units = 1)
      ]
  )

  #compiling the model
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss=tf.keras.losses.MeanSquaredError())
  
  # saving the best model
  # following can be done only in a Colab environment
  checkpoint = ModelCheckpoint('/content/drive/My Drive/Colab Notebooks/{}/best.tf'.format(index), monitor='loss', verbose=1, save_best_only=True, mode='min') 

  # fitting the model
  model.fit(X_train, y_train, epochs = 100, batch_size = 32, callbacks=[checkpoint])

  dataset_train = df.loc[:6000, index]
  dataset_test = df.loc[6000:, index]
  dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
  inputs = dataset_total[len(dataset_total) - len(dataset_test) - 36:].values
  inputs = inputs.reshape(-1,1)
  inputs = sc.transform(inputs)
  X_test = []
  for i in range(36, inputs.shape[0]):
      X_test.append(inputs[i-36:i, 0])
  X_test = np.array(X_test)
  X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
  print(X_test.shape)

  # loading previously built model
  # following can be done only in a Colab environment
  model.load_weights('/content/drive/My Drive/Colab Notebooks/{}/best.tf'.format(index))
  predicted_stock_price = model3.predict(X_test)
  predicted_stock_price = sc.inverse_transform(predicted_stock_price)

  #plotting the stock prices
  plt.plot(df.loc[6000:, 'Date'],dataset_test.values, color = 'red', label = 'Real {} Stock Price'.format(index))
  plt.plot(df.loc[6000:, 'Date'],predicted_stock_price, color = 'blue', label = 'Predicted {} Stock Price'.format(index))
  plt.xticks(np.arange(0,inputs.shape[0],120),rotation=45)
  plt.title('{} Stock Price Prediction'.format(index))
  plt.xlabel('Time')
  plt.ylabel('{} Stock Price'.format(index))
  plt.legend()
  plt.show()
```

## B) Checking Model Accuracy


```python
def model_accuracy_check(index):
  #splitting train and test set
  training_set = df.loc[:6000, index].values.reshape(-1,1)
  test_set = df.loc[6000:, index].values.reshape(-1,1)

  # using min max scaler
  sc = MinMaxScaler(feature_range = (0, 1))
  training_set_scaled = sc.fit_transform(training_set)

  # Creating a data structure with 60 time-steps and 1 output
  X_train = []
  y_train = []
  for i in range(36, 6000):
      X_train.append(training_set_scaled[i-36:i, 0])
      y_train.append(training_set_scaled[i, 0])
  X_train, y_train = np.array(X_train), np.array(y_train)
  X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

  # Define Sequential model2 with 9 layers
  model = keras.Sequential(
      [
      layers.GRU(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)),
      layers.Dropout(0.2),
      layers.GRU(units = 50, return_sequences = True),
      layers.Dropout(0.2),
      layers.GRU(units = 50, return_sequences = True),
      layers.Dropout(0.2),
      layers.GRU(units = 50),
      layers.Dropout(0.2),
      layers.Dense(units = 1)
      ]
  )

  #compiling the model
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss=tf.keras.losses.MeanSquaredError())
  


  dataset_train = df.loc[:6000, index]
  dataset_test = df.loc[6000:, index]
  dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
  inputs = dataset_total[len(dataset_total) - len(dataset_test) - 36:].values
  inputs = inputs.reshape(-1,1)
  inputs = sc.transform(inputs)
  X_test = []
  for i in range(36, inputs.shape[0]):
      X_test.append(inputs[i-36:i, 0])
  X_test = np.array(X_test)
  X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
  print(X_test.shape)

  # loading previously built model
  model3.load_weights('/content/drive/My Drive/Colab Notebooks/{}/best.tf'.format(index))
  predicted_stock_price = model.predict(X_test)
  predicted_stock_price = sc.inverse_transform(predicted_stock_price)

  # evaluate the model accuracy
  print('R Square of {}'.format(index),'\n',r2_score(dataset_test, predicted_stock_price))
  print('Mean absolute error of {}'.format(index),'\n',mean_absolute_error(dataset_test, predicted_stock_price))
  print('Mean squared error of {}'.format(index),'\n',mean_squared_error(dataset_test, predicted_stock_price),'\n','\n')
```
