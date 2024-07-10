import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ----------------------------------------------------------------------------------------------------------------------
#DATA USED: PAGASA 24H RAINFALL

# This script does the following:
# 1. Uses in-situ data
# 2. heavy rainfall days is determined when more than 6 stations (approx. 10% of 55) has 24h rr more than 50 mm
# 2. Determine the number of PCs
# 3. PCA
# 4. KMEANS
# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# Determine the number of pc components using scree plot
def num_PC(scaled_data, dir):

    len_data = len(scaled_data)
    pca= PCA()
    pca.fit(scaled_data)
    pca_data = pca.transform(scaled_data)
    t = pca.explained_variance_ratio_

    num_components = np.argmax(pca.explained_variance_ratio_.cumsum() >= 0.8) + 1

    plt.figure(figsize = (10,8))
    plt.plot(range(1,t.shape[0] + 1), pca.explained_variance_ratio_.cumsum(),marker='o', linestyle='--')
    plt.title('Explained Variance by Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.xticks(np.arange(1, t.shape[0], 10))
    plt.margins(x=0.01)
    plt.savefig(dir + 'scree_plot.png')
    print('Number of PC components:', num_components)

    return num_components
# ---------------------------------------------------------------------------------------------------------------------
def num_clusters(scores_pca,dir):
    wcss = []

    for i in range(1,10):
        kmeans_pca = KMeans(n_clusters=i, init = 'k-means++', random_state=42)
        kmeans_pca.fit(scores_pca)
        wcss.append(kmeans_pca.inertia_)

    plt.figure(figsize = (10,8))
    plt.plot(range(1,10), wcss ,marker='o',linestyle='--')
    plt.title('Number of Clusters')
    plt.xlabel('WCSS')
    plt.ylabel('K-means with PCA Clustering')
    plt.margins(x=0.01)
    plt.xticks(np.arange(1, 10, 1))
    plt.savefig(dir + 'elbow_method.png')
# ---------------------------------------------------------------------------------------------------------------------

dir = 'D:/00_Final_npj/data/'


data = pd.read_csv(dir + 'reconstructed_RR.csv')

df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])
NDJF_filter =(df['Date'] >= '2003-11-01') & (df['Date'] <= '2022-02-28')

df_NDJF = df.loc[NDJF_filter]

HRE = df_NDJF[df_NDJF.loc[:, df_NDJF.columns != 'Date'].gt(50).sum(axis=1) > 6]
HRE.set_index('Date', inplace=True)

# Determine the heavy rainfall days
# Set 'Date' as the index (optional)
#df.set_index('Date', inplace=True)

# Calculate the mean across each row
#df['Average'] = df.mean(axis=1)

#threshold = np.percentile(df['Average'], 90)
#HRE = df[df['Average'] > threshold]
#HRE = HRE.drop('Average', axis=1)

# Perform PCA

# 1. Standardized the data
scaler = StandardScaler()
scaler.fit(HRE)
scaled_data = scaler.transform(HRE)

# Determine the number of PC components
PC_comps = num_PC(scaled_data,dir)
print(PC_comps)
pca = PCA(n_components=PC_comps)
pca.fit(scaled_data)
scores_pca = pca.transform(scaled_data)

# Kmeans clustering

kmeans_pca = KMeans(n_clusters=3,init = 'k-means++', random_state = 42)
kmeans_pca.fit(scores_pca)

# Saving the results as the csv
result = HRE.rename_axis('Date').reset_index()

data_pca_kmeans = pd.concat([result.reset_index(drop=True),
                             pd.DataFrame(scores_pca)], axis=1)

data_pca_kmeans['Kmeans_PCA'] = kmeans_pca.labels_
data_pca_kmeans.to_csv(dir + 'pca_kmeans_optimum_3clusters_obs.csv', index=False)

