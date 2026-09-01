# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
 



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# === Load Tomato dataset ===
data = pd.read_csv("/content/Month_Value_1.csv")
data['Date'] = pd.to_datetime(data['Period'], format='%d.%m.%Y') # Use 'Period' column and specify format
data = data.sort_values('Date')

# Filter out rows where 'Revenue' is NaN before creating X and using for plotting
data_filtered = data.dropna(subset=['Revenue']).reset_index(drop=True)
X = data_filtered['Revenue'] # X now corresponds to the filtered data

# === Basic settings ===
N = 1000
plt.rcParams['figure.figsize'] = [12, 6]

# === Plot original series ===
plt.plot(data_filtered['Date'], X)
plt.title('Tomato Revenue') # Updated title to reflect 'Revenue'
plt.xlabel('Date')
plt.ylabel('Revenue') # Updated label to reflect 'Revenue'
plt.show()

# === ACF & PACF ===
fig, axes = plt.subplots(2, 1, figsize=(12,6))
plot_acf(X, lags=len(X)//4, ax=axes[0])
axes[0].set_title('ACF - Tomato Revenue') # Updated title
plot_pacf(X, lags=len(X)//4, ax=axes[1])
axes[1].set_title('PACF - Tomato Revenue') # Updated title
plt.tight_layout()
plt.show()

# === Fit ARMA(1,1) ===
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
print("\nARMA(1,1) parameters:")
print(arma11_model.params)

phi1 = arma11_model.params.get('ar.L1',0)
theta1 = arma11_model.params.get('ma.L1',0)

# Simulate ARMA(1,1)
ar1 = np.array([1, -phi1])
ma1 = np.array([1, theta1])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_1)
plt.title("ACF Simulated ARMA(1,1)")
plt.show()
plot_pacf(ARMA_1)
plt.title("PACF Simulated ARMA(1,1)")
plt.show()

# === Fit ARMA(2,2) ===
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
print("\nARMA(2,2) parameters:")
print(arma22_model.params)

phi1 = arma22_model.params.get('ar.L1',0)
phi2 = arma22_model.params.get('ar.L2',0)
theta1 = arma22_model.params.get('ma.L1',0)
theta2 = arma22_model.params.get('ma.L2',0)

# Simulate ARMA(2,2)
ar2 = np.array([1, -phi1, -phi2])
ma2 = np.array([1, theta1, theta2])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_2)
plt.title("ACF Simulated ARMA(2,2)")
plt.show()
plot_pacf(ARMA_2)
plt.title("PACF Simulated ARMA(2,2)")
plt.show()
```

OUTPUT:
<img width="750" height="858" alt="image" src="https://github.com/user-attachments/assets/ceb01dbf-dd18-4d34-93af-5bdf83d7b93a" />

<img width="763" height="887" alt="image" src="https://github.com/user-attachments/assets/f064edb4-76f4-4410-bcef-40c198b827e6" />
<img width="702" height="581" alt="image" src="https://github.com/user-attachments/assets/eed4bed9-ce10-47cf-ae9c-07fed3b74408" />
<img width="722" height="778" alt="image" src="https://github.com/user-attachments/assets/dba95dd8-916d-4c84-afc9-582f8d44a54a" />
<img width="732" height="406" alt="image" src="https://github.com/user-attachments/assets/5f2057a6-dc8c-41e8-b87c-167d36c0df12" />

RESULT:
Thus, a python program is created to fir ARMA Model successfully.
