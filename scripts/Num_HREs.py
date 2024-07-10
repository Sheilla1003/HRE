import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.family"] = 'Helvetica'
#plt.rcParams['savefig.transparent'] = True
plt.rcParams['font.size'] = 18
plt.rcParams['pdf.fonttype'] = 42

dir = 'D:/00_Final_npj/data/'


datasource = 'imerg'

data = pd.read_csv(dir + 'pca_kmeans_optimum_3clusters_' + datasource + '.csv')

#data = cluster_1
#columns_to_drop = data.iloc[:, 56:97].columns
#data = data.drop(columns=columns_to_drop, axis=1)
#data.set_index('Date', inplace=True)


#max_RR = data.values.max()
#max_column = data.max().idxmax()
#max_row_index = data[max_column].idxmax()
#print(max_RR,max_column,max_row_index)


#### ---
#df = pd.DataFrame(data)

# Bin days into three groups
#cluster = ['1','2','3']
cluster =['1']
for c in cluster:

    cluster_number = c

    fname = 'NumHRE_cluster' + cluster_number + '_' + datasource + '.png'

    num = int(c) - 1

    cluster = data[data['Kmeans_PCA'] == num]

    data = cluster
    date = pd.DataFrame(data.Date)
    date['Date'] = pd.to_datetime(date.Date)
    bins = [0, 10, 20, 32]  # Use 32 to ensure days 31 are included
    labels = ['Early (1-10)', 'Mid (11-20)', 'Late (21-end)']
    date['Day Bin'] = pd.cut(date.Date.dt.day, bins=bins, labels=labels, right=False)

    result = date.groupby([date['Date'].dt.month_name(), 'Day Bin']).size().unstack(fill_value=0)
    desired_order = ['November', 'December', 'January', 'February']
    label = ['Nov', 'Dec', 'Jan', 'Feb']

    result = result.reindex(desired_order)
    result.fillna(0, inplace=True)

    colors = ['blue', 'orange', 'red', 'green']
    edge_color = 'black'
    line_width = 1

    fig, ax = plt.subplots(figsize=(5, 6))
    positions = np.arange(len(result))
    width = 0.25
    for i, column in enumerate(result.columns):
        ax.bar(positions + i * width, result[column], width=width, color=colors,
               edgecolor=edge_color, linewidth=line_width, label=column)

    ax.set_xticks(positions + width)  # Adjust this to center the labels under the group of bars
    ax.set_xticklabels(label)
    ax.set_ylabel('Number of HREs')
    ax.set_ylim(0,18)

    outdir = 'D:/00_Final_npj/figures/'

    plt.savefig(outdir + fname,bbox_inches='tight', pad_inches=0.2, dpi=600)
    plt.close()