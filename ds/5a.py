import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('C:\\Users\\HP\\dataScience lab\\tips.csv')


print(data.columns)


grouped_data = data.groupby('day')['total_bill'].sum().reset_index()


X = grouped_data['day']
Y = grouped_data['total_bill']


plt.bar(X, Y)
plt.title("Total Bill by Day")
plt.xlabel('Day')
plt.ylabel('Total Bill')
plt.show()