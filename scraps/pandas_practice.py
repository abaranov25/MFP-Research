import numpy as np
import pandas as pd
'''
my_list = [1,2,3]
pd.Series(data=my_list)

labels = ['a','b','c']

print(pd.Series(my_list, labels))

my_dict = {'a': 1, 'b': 2, 'c': 3}
print(pd.Series(my_dict)[2])
'''
np.random.seed(100)
df = pd.DataFrame(data = np.random.randn(4,4), index=["A","B","C","D"], columns=["W","X","Y","Z"])
#print(df)
#print(df.describe().T)
#print(list(df['W'])) # THIS IS FOR COLUMNS
#print(df[['W', 'Y']])
df['new'] = df['X'] + df['Y']
#print(df)

df.drop('A', axis = 0, inplace=True)
df.drop('new', axis = 1, inplace=True)


print(df)
print(df.loc['C']) # THIS IS FOR ROWS
print(df.iloc[1])
print(df.iloc[np.arange(3)])
print(df.loc[['B','D'],['W','Z']])


print(df[df['Y']>0]['X'])
