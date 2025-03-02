import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('C:\\Users\\HP\\dataScience lab\\tips.csv')
print(data.columns)

X = data['Day']
Y = data['Total_bill']

plt.bar(X, Y)
plt.title("tips dataset")
plt.xlabel('Day')
plt.ylabel('Total_bill')
plt.show()
