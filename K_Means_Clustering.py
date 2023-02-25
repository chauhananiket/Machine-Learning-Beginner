import random 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs 
import pandas as pd
cust_df = pd.read_csv(r'F:\Python Programs(Machine Learning)\Dataset\Unzipped\CSV Files\Cust_Segmentation.csv')
print(cust_df.head())

'''Pre-processing
As you can see, Address in this dataset is a categorical variable. k-means algorithm isn't directly applicable to categorical variables because Euclidean distance function isn't really meaningful for discrete variables. So, lets drop this feature and run clustering.
'''
df = cust_df.drop('Address', axis=1)
df.head()
   
'''Normalizing over the standard deviation
Now let's normalize the dataset. But why do we need normalization in the first place? Normalization is a statistical method that helps mathematical-based algorithms to interpret features with different magnitudes and distributions equally. We use StandardScaler() to normalize our dataset.
'''
from sklearn.preprocessing import StandardScaler
X = df.values[:,1:]
print(X)
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet

clusterNum = 2
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)


'''We assign the labels to each row in dataframe.'''
df["Clus_km"] = labels
print(df.head(5))

#We can easily check the centroid values by averaging the features in each cluster.

print(df.groupby('Clus_km').mean())
