import pandas as pd
import statistics

uci = pd.read_csv('C:\\Users\\HP\\Desktop\\uni.csv')

uci.columns = uci.columns.str.strip()

if 'num_lab_procedures' in uci.columns:
    mean = statistics.mean(uci['num_lab_procedures'])
    mode = statistics.mode(uci['num_lab_procedures'])
    median = statistics.median(uci['num_lab_procedures'])
    variance = statistics.variance(uci['num_lab_procedures'])
    standard_deviation = statistics.stdev(uci['num_lab_procedures'])
    fre_count = uci['num_lab_procedures'].value_counts()
    skew = uci['num_lab_procedures'].skew()
    kurt = uci['num_lab_procedures'].kurtosis()

    print("\nStatistics for 'num_lab_procedures' column:")
    print("Mean:", mean)
    print("Mode:", mode)
    print("Median:", median)
    print("Variance:", variance)
    print("Standard Deviation:", standard_deviation)
    print("Frequency Count:\n", fre_count)
    print("Skewness:", skew)
    print("Kurtosis:", kurt)
else:
    print("'num_lab_procedures' column not found in the dataset.")
