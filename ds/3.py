import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\\HP\\dataScience lab\\iris.csv')

print("First few rows of the dataset:")
print(df.head())

print("\nDescriptive Statistics:")
print(df.describe())

print("\nMean of each Column:")
print(df.select_dtypes(include='number').mean())

print("\nMedian of each Column:")
print(df.select_dtypes(include='number').median())

print("\nStandard Deviation for each Column:")
print(df.select_dtypes(include='number').std())

print("\nCount of unique values for each column:")
print(df.nunique())

print("\nClass distribution:")
print(df['class'].value_counts())

print("\nCorrelation Matrix:")
print