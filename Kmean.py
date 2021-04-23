import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


#Importing data and replace Null with 0
D_Train = pd.read_csv('salju_train.csv')
D_Test  = pd.read_csv('salju_test.csv')
D_TrainCopy = D_Train
D_Train = D_Train.fillna(0)
D_Test  = D_Test.fillna(0)

#changing categorical data with numerical data
cleanup_nums_Train = {
    "KodeLokasi":{"C1":1,"C2":2,"C3":3,"C4":4,"C5":5,"C6":6,"C7":7,
    "C8":8,"C9":9,"C10":10,"C11":11,"C12":12,"C13":13,"C14":14,"C15":15,
    "C16":16,"C17":17,"C18":18,"C19":19,"C20":20,"C21":21,
    "C22":22,"C23":23,"C24":24,"C25":25,"C26":26,"C27":27,"C28":28,"C29":29,"C30":30,
    "C31":31,"C32":32,"C33":33,"C34":34,"C35":35,"C36":36,"C37":37,"C38":38,"C39":39,
    "C40":40,"C41":41,"C42":42,"C43":43,"C44":44,"C45":45,"C46":46,"C47":47,"C48":48,
    "C49":49},
    "ArahAnginTerkencang":{"N":1,"NNE":2,"NE":3,"ENE":4,"E":5,"ESE":6,"SE":7,"SSE":8,
    "S":9,"SSW":10,"SW":11,"WSW":12,"W":13,"WNW":14,"NW":15,"NNW":16},
    "ArahAngin9am":{"N":1,"NNE":2,"NE":3,"ENE":4,"E":5,"ESE":6,"SE":7,"SSE":8,
    "S":9,"SSW":10,"SW":11,"WSW":12,"W":13,"WNW":14,"NW":15,"NNW":16},
    "ArahAngin3pm":{"N":1,"NNE":2,"NE":3,"ENE":4,"E":5,"ESE":6,"SE":7,"SSE":8,
    "S":9,"SSW":10,"SW":11,"WSW":12,"W":13,"WNW":14,"NW":15,"NNW":16},
    "BersaljuHariIni":{"Ya":1,"Tidak":0},
    "BersaljuBesok":{"Ya":1,"Tidak":0}
}
cleanup_nums_Test = {
    "KodeLokasi":{"C1":1,"C2":2,"C3":3,"C4":4,"C5":5,
    "C6":6,"C7":7,"C8":8,"C9":9,"C10":10,"C11":11,"C12":12,"C13":13,
    "C14":14,"C15":15,"C16":16,"C17":17,"C18":18,"C19":19,"C20":20,"C21":21,
    "C22":22,"C23":23,"C24":24,"C25":25,"C26":26,"C27":27,"C28":28,"C29":29,"C30":30,
    "C31":31,"C32":32,"C33":33,"C34":34,"C35":35,"C36":36,"C37":37,"C38":38,"C39":39,
    "C40":40,"C41":41,"C42":42,"C43":43,"C44":44,"C45":45,"C46":46,"C47":47,
    "C48":48,"C49":49},
    "ArahAnginTerkencang":{"N":1,"NNE":2,"NE":3,"ENE":4,"E":5,"ESE":6,"SE":7,"SSE":8,
    "S":9,"SSW":10,"SW":11,"WSW":12,"W":13,"WNW":14,"NW":15,"NNW":16},
    "ArahAngin9am":{"N":1,"NNE":2,"NE":3,"ENE":4,"E":5,"ESE":6,"SE":7,"SSE":8,
    "S":9,"SSW":10,"SW":11,"WSW":12,"W":13,"WNW":14,"NW":15,"NNW":16},
    "ArahAngin3pm":{"N":1,"NNE":2,"NE":3,"ENE":4,"E":5,"ESE":6,"SE":7,"SSE":8,
    "S":9,"SSW":10,"SW":11,"WSW":12,"W":13,"WNW":14,"NW":15,"NNW":16},
    "BersaljuHariIni":{"Yes":1,"No":0},
    "BersaljuBesok":{"Yes":1,"No":0}
}
D_Train = D_Train.replace(cleanup_nums_Train)
D_Test = D_Test.replace(cleanup_nums_Test)

#Function and class

def EuclideanDistance(x1,x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def standardScaler(Dataframe):
    new = Dataframe - np.mean(Dataframe, axis=0)
    return new / np.std(new, axis=0)

def Silhouette(Cluster):
    Inner_Cluster = []
    Outer_Cluster = []
    for idx in range(len(Cluster)):
        rand_data_from_clus = Cluster[idx][np.random.choice(len(Cluster[idx]))]
        dist_inner = 0
        dist_outer = 0
        for point in range(1, len(Cluster[idx])):
            dist_inner += EuclideanDistance(rand_data_from_clus, Cluster[idx][point])
        Inner_Cluster.append(dist_inner / len(Cluster[idx]))
        # Menghitung inter-cluster distance
        next_idx = idx + 1
        while next_idx < len(Cluster):
            for point in range(len(Cluster[next_idx])):
                dist_outer += EuclideanDistance(rand_data_from_clus, Cluster[next_idx][point])
            Outer_Cluster.append(dist_outer / len(Cluster[next_idx]))
            next_idx += 1

    Inner_Cluster = np.mean(Inner_Cluster)
    Outer_Cluster = np.mean(Outer_Cluster)
    return (Outer_Cluster - Inner_Cluster) / max(Inner_Cluster, Outer_Cluster)
    
class KMeans:
    def __init__(self, k, Iteration):
        self.k = k
        self.Iteration = Iteration
        
        #indices of sample for clusters
        self.clusters = [[] for _ in range(self.k)]
        #centroids/mean of vector distances in every cluster
        self.centroids = []
    
    def predict(self,Feature_data_train):
        self.Feature_data_train = Feature_data_train
        self.num_sample,self.num_feature = Feature_data_train.shape
        #initialize centroids
        rand_idx = np.random.choice(self.num_sample,self.k, replace=False)
        self.centroids = [self.Feature_data_train.values[idx] for idx in rand_idx]
        #optimization centroids
        for _ in range(self.Iteration):
            #update clusters(which point goes to which clusters)
            self.clusters = self._createClusters(self.centroids)
            #update centroids(more optimize centroids to each point)
            prev_centroids = self.centroids
            self.centroids = self._getCentroid(self.clusters)
            #check if the centroids converged
            if(self._converged(prev_centroids,self.centroids)):
                break
        #return label
        return self.clusters

    def _converged(self,prev_centroids,centroids):
        distances = [EuclideanDistance(prev_centroids[i],centroids[i]) for i in range(self.k)]
        return (sum(distances) == 0)

    def _getCentroid(self,clusters):
        centroids = np.zeros((self.k,self.num_feature))
        for idx_clus in range(len(clusters)):
            centroids[idx_clus] = np.average(clusters[idx_clus],axis=0)
        return centroids
        #ini masih salah bre
        #mendingan besok cari mean per kolom dari cluster abis itu jadiin satu centroid baru
        #bikin array sebanyak K, abis itu adding array pake numpy.add terus di bagi banyak K
        #kalo ga pake sum(array) axis = 0
        # jadi entar taro dulu di array si cluster x abis itu ditambah terus dibagi panjang cluster            
        # for clus_idx,cluster in enumerate(clusters):
        #     clus_mean = np.mean(self.Feature_data_train[cluster],axis=0)
        #     centroids[clus_idx] = clus_mean
        # return centroids

    def _createClusters(self,centroids):
        clusters = [[] for _ in range(self.k)]
        #cari centroid terdekat dari masing masing data biar dijadiin cluster
        for i in range(len(self.Feature_data_train)):
            #print(self.Feature_data_train.values[i])
            centroids_idx = self._closestCentroid(self.Feature_data_train.values[i],centroids)
            clusters[centroids_idx].append(self.Feature_data_train.values[i])
        # for idx,data in enumerate(self.Feature_data_train):
        #     centroids_idx = self._closestCentroid(data.value[idx],centroids)
        #     clusters[centroids_idx].append(data.value[idx])
        return clusters
    
    def _closestCentroid(self,sample,centroids):
        distances = [EuclideanDistance(sample,point) for point in centroids]
        closest = np.argmin(distances)
        return closest

        

#Data Split between Feature and outcome and scaling data using standard scale
# Data_Train_Feature = D_Train.iloc[:,[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]]
Data_Train_Feature = D_Train.iloc[:1000,[2,21]]
Data_Test_Feature = D_Test.iloc[:500,[1,20]]
Data_Train_Feature = standardScaler(Data_Train_Feature)
Data_Test_Feature = standardScaler(Data_Test_Feature)

print("Ini Bagian Train data")
for i in range(2,11):
    model_Train = KMeans(i, Iteration=100)
    Labeled_Clusters_Train = model_Train.predict(Data_Train_Feature)
    Silhouette_Koef = Silhouette(Labeled_Clusters_Train)
    print("Cluster sebanyak:",i,"dengan Nilai koefisien silhoutte:",Silhouette_Koef)

print("Ini Bagian Test")
print("Masukan K dari elbow di atas: ")
BestK = int(input())
model_Test = KMeans(BestK,Iteration=100)
Labeled_Clusters_Test = model_Test.predict(Data_Test_Feature)
Silhouette_Koef_Test = Silhouette(Labeled_Clusters_Test)
print("Cluster Terbaik Setalah data di train adalah",BestK,"dan nilai koefisien dari data test adalah:",Silhouette_Koef_Test)

print("Ini bagian eksperiment dimana data yang tidak ada isinya akan di isi oleh 1 dan perubahan cluster menjadi 9 dan iterasi di ubah menjadi 300")
D_TrainCopy = D_TrainCopy.fillna(1)
D_TrainCopy = D_TrainCopy.replace(cleanup_nums_Train)
Data_Train_Feature_Experiment = D_TrainCopy.iloc[:2000,[2,21]]
Data_Train_Feature_Experiment = standardScaler(Data_Train_Feature_Experiment)

model_Train_Experiment = KMeans(9, Iteration=300)
Labeled_Clusters_Train_Experiment = model_Train_Experiment.predict(Data_Train_Feature_Experiment)
Silhouette_Koef_Experiment = Silhouette(Labeled_Clusters_Train_Experiment)
print("Cluster sebanyak: 9 ,dengan Nilai koefisien silhoutte:",Silhouette_Koef_Experiment)

#Pake silhouete untuk nyari k terbaik(model dan pred harus dalam loop) setelah dapat data k terbaik, maka k tersebut dipakai untuk pembuatan model dengan k tersebut
#seteleh dapat model dengan k tersebut(cluster) maka test akan bisa di bandingkan hasilnya dengan centroid terdekat menggunakan eulidean distance

#dibawah ini semua dahbener
#entar bikin plottingan baru,buat data test baru train soalnya
#plottingan dan scatter data
cmap = {0 : 'r',1 : 'b', 2: 'darkgreen', 3: 'm', 4: 'y', 5: 'darkorange', 6: 'c', 7: 'darkviolet', 8:'pink', 
9:'gold', 10:'hotpink', 11:'magenta',12:'orange'}
label_color = [cmap[l] for l in range(len(Labeled_Clusters_Test))]
label_color_EX = [cmap[l] for l in range(len(Labeled_Clusters_Train_Experiment))]


for idx_class in range(len(Labeled_Clusters_Test)):
    for idx_data in range(len(Labeled_Clusters_Test[idx_class])):
        plt.scatter(Labeled_Clusters_Test[idx_class][idx_data][0], Labeled_Clusters_Test[idx_class][idx_data][1] , c=label_color[idx_class])

for centroid in range(len(model_Test.centroids)):
    plt.scatter(model_Test.centroids[centroid][0], model_Test.centroids[centroid][1], marker="*", color='k', s=150, linewidths=5)

plt.show()

D_Train = standardScaler(D_Train)
D_Test = standardScaler(D_Test)
D_Train.to_csv(r'dataTrain.csv')
D_Test.to_csv(r'dataTest.csv')

# for idx_experiment in range(len(Labeled_Clusters_Train_Experiment)):
#     for idx_data_experiment in range(len(Labeled_Clusters_Train_Experiment[idx_experiment])):
#         plt.scatter(Labeled_Clusters_Train_Experiment[idx_experiment][idx_data_experiment][0],Labeled_Clusters_Train_Experiment[idx_experiment][idx_data_experiment][1],c=label_color_EX[idx_experiment])

# for centroid_experimen in range(len(model_Train_Experiment.centroids)):
#     plt.scatter(model_Train_Experiment.centroids[centroid_experimen][0], model_Train_Experiment.centroids[centroid_experimen][1], marker="*", color='k', s=150, linewidths=5)

# plt.show()