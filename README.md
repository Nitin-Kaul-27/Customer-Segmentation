In this project, I have used K-Means Clustering algorithm to make the clusters based on their income and spending. K-Means algorithm is a unsupervised Machine Learning algorithm which helps in categorizing.

The steps involved in K-Means Clustering are as follows:
1) Initialize the number of Clusters to be considered. (For that optimum number of clusters selection must be used first, a relation between no. of clusters and inertia), but in this case it is considered by default which was = 5.
2) Randomly initialize the centroid points based on the data in 2 columns.
3) Calculate the minimum distance between the data points and the centroids by using Euclidean distance method.
4) Based on minimum distance, assign the clusters to data points.
5) Re-Calculated the centroids, and applied steps 3 to 4 again on data points.
6) Match the new clusters with previous clusters. If matched code breaks, if not Repeat Steps 3 to 5 until the clusters do not change.

In Python, there is a library sklearn.cluster from where you can directly import K_means algo and use it in a single line. But in my code, I have created the algorithm without using inbuilt Python Machine Learning library and written all the lines on my own
