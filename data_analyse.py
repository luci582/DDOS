
import pandas as pd
import numpy as np
import seaborn as sns                       #visualisation
import matplotlib.pyplot as plt             #visualisation
pd.set_option('display.max_columns', None)
df=pd.read_csv('APA-DDoS-Dataset.csv')
import copy
#
# print(f"first 5 lines of data \n {df.head(5)}")
# print(f"last 5 lines of data \n {df.tail(5)}")
# print(f"data shape \n {df.shape}")
# print(f"data types \n {df.dtypes}")
# column_name_df = df.columns
# print(f"column names \n {column_name_df}")

# column_need = df[['ip.src','frame.time','Label']]
# column_need = column_need.rename(columns={'ip.src':'ip','frame.time':'time','Label':'label'})
# print(column_need.head(5))
# print(f"dublicate check\nnumber of duplicates = {column_need.duplicated().sum()}")
# print(f"check of empty values\nnumber of empty rows : \n{column_need.isnull().sum()}")
# sns.boxplot(x='time',y='ip',data=column_need)
# plt.show()
# print(df.describe(include= 'all'))
# for i in df.columns.tolist():
#   print("No. of unique values in",i,"is",df[i].nunique())
# print(df.columns)
columns_to_keep = [
    'ip.src',
    'tcp.srcport',
    'frame.len',
    'tcp.flags.push',
    'ip.flags.df',
    'Packets',
    'Bytes',
    'Tx Packets',
    'Tx Bytes',
    'Rx Packets',
    'Rx Bytes',
    'frame.time',
    'Label'
]
data = df[columns_to_keep]
time_split = data["frame.time"].str.split(" ", expand=True)
data["timez"] = copy.deepcopy(time_split[3])
data=data.drop(columns=['frame.time'])
columns_to_keep= data.columns.tolist()
print(data.head(5))
# for i in columns_to_keep:
#     plt.hist(data[i])
#     plt.title(i)
#     plt.show()
label_list = data["Label"].unique()
colors = ['red', 'yellow', 'green']
plot_helper = [
    ('Packets', 'Bytes'),
    ('Rx Packets', 'Tx Packets'),
    ('frame.len', 'Packets'),
    ('timez', 'Packets'),
    ('timez', 'Bytes')
]
for x_feature, y_feature in plot_helper:
    plt.figure(figsize=(10, 6))
    for i in range(3):
         x=data[data["Label"]==label_list[i]]
         plt.scatter(x[x_feature], x[y_feature], c=colors[i], label=label_list[i], alpha=0.6, s=10)
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.legend()
    plt.show()

numerical_data = data.select_dtypes(include=np.number)
corr_matrix = numerical_data.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='Reds_r')
plt.title('Correlation Matrix heatmap')
plt.show()

