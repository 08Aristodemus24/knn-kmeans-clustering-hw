import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class KMeans:
    def __init__(self, X, K, epochs=10, initializer="permutation"):
        self.X = X
        self.K = K
        self.num_instances = X.shape[0]
        self.num_features = X.shape[1]
        self.epochs = epochs
        self.initializer = initializer


    def init_centroids(self, X):
        # num features and clusters
        num_instances, K = self.num_instances, self.K

        if self.initializer == "random_sampling":
            # sample K samples from the list of indeces
            rand_idxs = np.random.choice(list(range(num_instances)), size=K, replace=False)
            print(rand_idxs)

        elif self.initializer == "permutation":
            # Randomly reorder the indices of examples
            rand_idxs = np.random.permutation(X.shape[0])[:K]


        # get the data points with these random indeces
        init_centroids = X[rand_idxs, :]
        return init_centroids

    def _assign_centroids_to_xs(self, X, centroids):
        """
        assigns the appropriate centroid for every example
        
        Args:
            X (ndarray): (m, n) Input values      
            centroids (ndarray): (K, n) centroids
        
        Returns:
            xs_centroids (array_like): (m,) values that indicate the
            respective centroids of each training example
        
        """

        # the vector which contains all the 
        # centroids assigned to a particular index
        num_instances = self.num_instances
        xs_centroids = np.zeros(shape=(num_instances))

        for index in range(num_instances):
            # index denotes what training example is to be assigned to the closest 
            # centroid there are k centroids so np.argmin will be values from 0 to k-1
            xs_centroids[index] = np.argmin(np.sum((X[index] - centroids) ** 2, axis=1))

        return xs_centroids

    def _calc_centroids(self, X, xs_centroids, K):
        """
        Returns the new centroids by computing the means of the 
        data points assigned to each centroid.
        
        Args:
            X (ndarray):   (m, n) Data points
            xs_centroids (ndarray): (m,) Array containing index of closest centroid for each 
                        example in X. Concretely, idx[i] contains the index of 
                        the centroid closest to example i
            K (int):       number of centroids
        
        Returns:
            new_centroids (ndarray): (K, n) New centroids computed
        """

        # Useful variables
        n = self.num_features
        
        # You need to return the following variables correctly
        new_centroids = np.zeros((K, n))
        
        
        # loops from 0 to k-1
        for k in range(K):
            # extract all the points closest/assigned to centroid k
            points_of_ck = X[np.where(xs_centroids == k)]

            # calculate the mean of these data points
            new_centroids[k] = np.sum(points_of_ck, axis=0) / len(points_of_ck)
        
        return new_centroids


    def train(self):
        X, K = self.X, self.K
        init_centroids = self.init_centroids(X)

        print(init_centroids)
        self.visualize(X)

        prev_centroids = []
        

        # assign centroids to initial centroids since this will change overtime
        centroids = init_centroids
        for epoch in range(self.epochs):
            # print(f'epoch {epoch}/{self.epochs - 1}')

            # assign appropriate centroids to every data point
            xs_centroids = self._assign_centroids_to_xs(X, centroids)

            # save previous centroids before assining new centroids
            prev_centroids.append(centroids)
            print(f'centroid at epoch {epoch}/{self.epochs - 1}: \n{centroids}\n')

            # calculate new centroids after assigning
            # appropriate centroids to every data point
            centroids = self._calc_centroids(X, xs_centroids, K)
        
        # add last centroid to the prev_centroids
        prev_centroids.append(centroids)
        print(f'centroid at epoch {epoch}/{self.epochs - 1}: \n{centroids}\n')

        # plot centroids across iterations
        self.plot_evolution(X, np.array(prev_centroids))

            
    def plot_evolution(self, X, centroids):
        """
        centroids - is a list of 2D numpy arrays
        """
        K = self.K

        fig = plt.figure(figsize=(15, 15))
        axis = fig.add_subplot()
        axis.scatter(X[:, 0], X[:, 1], c=np.random.randn(self.num_instances), marker='p',alpha=0.75, cmap='magma')
        axis.set_xlabel('age')
        axis.set_ylabel('monthly_mileage')

        for k in range(K):
            cs_of_k = centroids[:, k, :]
        
            print(f'centroids of cluster {k}: {cs_of_k}\n')
            axis.plot(cs_of_k[:, 0], cs_of_k[:, 1], 'x--', alpha=0.25, color='black')

        plt.legend()
        plt.show()

    def visualize(self, X):
        fig = plt.figure(figsize=(15, 15))
        axis = fig.add_subplot()
        axis.scatter(X[:, 0], X[:, 1], c=np.random.randn(self.num_instances), marker='p',alpha=0.75, cmap='magma')
        axis.set_xlabel('age')
        axis.set_ylabel('monthly_mileage')

        plt.legend()
        plt.show()


def load_data():
    X = np.load("data/ex7_X.npy")
    return X

if __name__ == "__main__":
    # data = load_data()
    data = pd.DataFrame({'age': [35, 45, 22, 55, 30, 40, 50, 27, 48, 33], 'monthly_mileage': [500, 800, 300, 200, 400, 700, 250, 350, 600, 450]})

    model = KMeans(data, 3)
    model.train()