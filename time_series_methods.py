import pandas as pd
import matplotlib.pyplot as plt

# Ftiaxnoume paradeigma dataset me 30 meres
data = {
    "date": pd.date_range(start="2023-01-01", periods=30, freq="D"),
    "temperature": [10, 12, 13, 15, 14, 16, 18, 17, 19, 20, 21, 22, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 8, 9, 10, 11]
}
df = pd.DataFrame(data)

#Metatropi se datetime kai index
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

print("Olo to DataFrame:")
print(df)

#Slicing time series data
print("\nSlicing apo 2023-01-05 mexri 2023-01-10:")
print(df["2023-01-05":"2023-01-10"])

#Extracting statistics
print("\nMean temperature:", df["temperature"].mean())
print("Median temperature:", df["temperature"].median())
print("\nDescriptive statistics:")
print(df["temperature"].describe())

#Resampling (apo daily se weekly)
print("\nResampling weekly mean:")
print(df["temperature"].resample("W").mean())

print("\nResampling weekly median:")
print(df["temperature"].resample("W").median())

#Rolling mean 
df["rolling_mean"] = df["temperature"].rolling(window=7).mean()
print("\nDataFrame me rolling mean:")
print(df)

#Plot gia na doume tin xroniki seira kai rolling mean
df[["temperature", "rolling_mean"]].plot(title="Temperature with 7-day Rolling Mean")
plt.show()
