
# 2a
import numpy as np
arr = np.array( [[ 1, 2, 3], [ 4, 2, 5]] )
print("Array is of type: ", type(arr))
print("No. of dimensions: ", arr.ndim)
print("Shape of array: ", arr.shape)
print("Size of array: ", arr.size)
print("Array stores elements of type: ", arr.dtype)


#2b
import numpy as np
list_data=[1,2,3,4,5]
array_from_list=np.array(list_data)
tuple_data=tuple(list_data)
array_from_tuple=np.array(tuple_data)
print("Original list:")
print(list_data)
print("Original tuple")
print(tuple_data)
print("Numpy array from list:")
print(array_from_list)
print("Numpy array from tuple")
print(array_from_tuple)

#2c
import pandas as pd
df=pd.DataFrame()
print(df)
print("Dataframe Creation using list")
list=['Geeks','For','Geeks','is','portal','for','Geeks']
df=pd.DataFrame(list)
print(df)
Data={'name':['tom','nick','krish','jack'],'Age':[20,21,19,18]}
df=pd.DataFrame(Data)
print(df)
print("Create dataframe from dictionary of lists")
dict={"name":["aparna","pankaj","sudhir","geeku"],'Degree':["MBA","BCA","M.Tech","MBA"],'Score':[90,40,80,98]}
df=pd.DataFrame(dict)
print(df)
for i,j in df.iterrows():
print(i,j)
print()





