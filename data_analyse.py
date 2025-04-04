
import pandas as pd
import numpy as np
import seaborn as sns                       #visualisation
import matplotlib.pyplot as plt             #visualisation
pd.set_option('display.max_columns', None)
df=pd.read_csv('APA-DDoS-Dataset.csv')
#
# print(f"first 5 lines of data \n {df.head(5)}")
# print(f"last 5 lines of data \n {df.tail(5)}")
# print(f"data shape \n {df.shape}")
# print(f"data types \n {df.dtypes}")
# column_name_df = df.columns
# print(f"column names \n {column_name_df}")

column_need = df[['ip.src','frame.time','Label']]
column_need = column_need.rename(columns={'ip.src':'ip','frame.time':'time','Label':'label'})
print(column_need.head(5))
print(f"dublicate check\nnumber of duplicates = {column_need.duplicated().sum()}")
print(f"check of empty values\nnumber of empty rows : \n{column_need.isnull().sum()}")
sns.boxplot(x='time',y='ip',data=column_need)
plt.show()
