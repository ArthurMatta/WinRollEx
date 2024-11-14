# WinRollEx 

## Introduction

WinRollEx is a simple rolling and expanding windows for machine learning applications. It supports both temporal and 
spatial window types.

## Installation

_WinRollEx_ can be installed using the following command:

```bash
pip install winrollex
```

## Usage

```python
import winrollex as wrx
import pandas as pd
import numpy as np

# Create a DataFrame
df = pd.DataFrame(np.random.randn(1000, 4))
print(df.head(10))

# Create a window generator
window = wrx.Window(df)

# Rolling Window
# using training_window_size, test_window_size and window_increment
for train, test in window.rolling(training_window_size=200, test_window_size=100, window_increment=50):
    print(train)
    print(test)

# using training_window_size and test_window_size.
# Note: if window_increment is not set, it will assume the value of test_window_size.
for train, test in window.rolling(training_window_size=200, test_window_size=100):
    print(train)
    print(test)

# using training_window_size and iterations
# Note: the test_window_size and window_increment are ignored when iterations is set and will assume the value of
# (dataset_size - training_window_size) / iterations.
for train, test in window.rolling(training_window_size=200, iterations=10):
    print(train)
    print(test)

# using training_window_size, test_window_size and window_increment as a % of the dataset size
for train, test in window.rolling(training_window_size=0.3, test_window_size=0.1, window_increment=0.1):
    print(train)
    print(test)

# using training_window_size as a % of the dataset size and iterations
for train, test in window.rolling(training_window_size=0.3, iterations=10):
    print(train)
    print(test)

# Expanding Window
# using training_window_size, test_window_size and window_increment
for train, test in window.expanding(training_window_size=200, test_window_size=100, window_increment=50):
    print(train)
    print(test)

# using training_window_size and test_window_size.
# Note: if window_increment is not set, it will assume the value of test_window_size.
for train, test in window.expanding(training_window_size=200, test_window_size=100):
    print(train)
    print(test)

# using training_window_size and iterations
# Note: the test_window_size and window_increment are ignored when iterations is set and will assume the value of
# (dataset_size - training_window_size) / iterations.
for train, test in window.expanding(training_window_size=200, iterations=10):
    print(train)
    print(test)

# using training_window_size, test_window_size and window_increment as a % of the dataset size
for train, test in window.expanding(training_window_size=0.3, test_window_size=0.1, window_increment=0.1):
    print(train)
    print(test)

# using training_window_size as a % of the dataset size and iterations
for train, test in window.expanding(training_window_size=0.3, iterations=10):
    print(train)
    print(test)

# Temporal Rolling Window
# Note: for temporal windows, it is assumed that the dataframe index is a datetime
df = pd.DataFrame(np.random.randn(1000, 4), index=pd.date_range('1/1/2000', periods=1000))

# The windows are either pandas.Timedelta, datetime.timedelta or dateutils.relativedelta objects.
training_delta = pd.Timedelta(days=200)
test_delta = pd.Timedelta(days=100)
increment_delta = pd.Timedelta(days=50)

# using training_window_size, test_window_size and timedelta
for train, test in window.temporal_rolling(training_window_size=training_delta, test_window_size=test_delta, timedelta=increment_delta):
    print(train)
    print(test)

# using training_window_size and test_window_size.
# Note: if timedelta is not set, it will assume the value of test_window_size.
for train, test in window.temporal_rolling(training_window_size=training_delta, test_window_size=test_delta):
    print(train)
    print(test)

# using training_window_size and iterations
# Note: the test_window_size and timedelta are ignored when iterations is set and will assume the value of
# ((end_date - start_date) - training_window_size) / iterations.
for train, test in window.temporal_rolling(training_window_size=training_delta, iterations=10):
    print(train)
    print(test)

# using training_window_size, test_window_size and timedelta as a % of the dataset size
for train, test in window.temporal_rolling(training_window_size=0.3, test_window_size=0.1, timedelta=increment_delta):
    print(train)
    print(test)

# using training_window_size as a % of the dataset size and iterations
for train, test in window.temporal_rolling(training_window_size=0.3, iterations=10):
    print(train)
    print(test)

# Temporal Expanding Window
# using training_window_size, test_window_size and timedelta
for train, test in window.temporal_expanding(training_window_size=training_delta, test_window_size=test_delta, timedelta=increment_delta):
    print(train)
    print(test)

# using training_window_size and test_window_size.
# Note: if timedelta is not set, it will assume the value of test_window_size.
for train, test in window.temporal_expanding(training_window_size=training_delta, test_window_size=test_delta):
    print(train)
    print(test)

# using training_window_size and iterations
# Note: the test_window_size and timedelta are ignored when iterations is set and will assume the value of
# ((end_date - start_date) - training_window_size) / iterations.
for train, test in window.temporal_expanding(training_window_size=training_delta, iterations=10):
    print(train)
    print(test)

# using training_window_size and test_window_size as a % of the dataset size
for train, test in window.temporal_expanding(training_window_size=0.3, test_window_size=0.1, timedelta=increment_delta):
    print(train)
    print(test)

# using training_window_size as a % of the dataset size and iterations
for train, test in window.temporal_expanding(training_window_size=0.3, iterations=10):
    print(train)
    print(test)
```
