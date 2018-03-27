import pandas as pd
import numpy as np
import scipy
from sklearn.cluster import KMeans


def kmeans(K):

    ##
    Model = KMeans(n_clusters=K,max_iter=1000)
    Trained_Model = Model.fit(Train_Data)

    ##
    Predictions_Normal_Instance =  Trained_Model.predict(Normal_Instance)
    counts = np.bincount(Predictions_Normal_Instance)
    Cluster_Number_Normal = np.argmax(counts)

    ##
    Predictions_Test_Data = Trained_Model.predict(Test_Data)

    ##
    Predictions_Test_Data = np.vectorize(lambda x: 0 if x == Cluster_Number_Normal else 1)(Predictions_Test_Data)
    _, counts = np.unique(Predictions_Test_Data, return_counts=True)
    Num_Normal_Detected=counts[0]
    Num_Attack_Detected=counts[1]

    ##
    True_Normal_Detected = 0
    True_Attack_Detected = 0
    False_Positives = 0
    False_Negatives = 0

    ##
    assert len(Predictions_Test_Data) == len(Test_Data_labels)
    for i in range(len(Predictions_Test_Data)):
        if Predictions_Test_Data[i]==0 and Test_Data_labels[i]==0:
            True_Normal_Detected += 1
        if Predictions_Test_Data[i]==1 and Test_Data_labels[i]==1:
            True_Attack_Detected += 1
        if Predictions_Test_Data[i]==1 and Test_Data_labels[i]==0:
            False_Positives += 1
        if Predictions_Test_Data[i]==0 and Test_Data_labels[i]==1:
            False_Negatives += 1

    ##
    DR_Noraml = (True_Normal_Detected/Num_Normal_Detected)*100
    DR_Attack = (True_Attack_Detected/Num_Attack_Detected)*100
    FPR = (False_Positives/Num_Normal)*100
    FNR = (False_Negatives/Num_Attack)*100
    Efficiency = DR_Noraml + DR_Attack

    ##
    print("K: ",K)
    print("DR_Normal:",DR_Noraml,"DR_Attack:",DR_Attack,"FPR:",FPR,"FNR:",FNR,"Efficiency",Efficiency)
    print("########################")
    return DR_Noraml,DR_Attack,FPR,FNR,Efficiency


# Loading Data
Train_Data = pd.read_csv("Train_Data_preprocessed_Y_Included.csv" , header=None)
Test_Data = pd.read_csv("Test_Data_preprocessed_Y_Included.csv" , header=None)

# Save Difficulty Level
Difficulty_Level_Train = Train_Data.iloc[:][42]
Difficulty_Level_Test = Test_Data.iloc[:][42]
del Train_Data[42]
del Test_Data[42]

# Find index that point the labels
Labels_index = len(Train_Data.iloc[0][:])-1

# Filter normal instance for find Cluster number that represent normal label
Normal_Instance = Train_Data[Train_Data[Labels_index]==0]
del Normal_Instance[Labels_index]

# save Train labels and delete them from dataset
Train_Data_labels = Train_Data.iloc[:][Labels_index]
del Train_Data[Labels_index]

# Counting number of each class that will be used in calculating Accuracy
Num_Count = Test_Data[Labels_index].value_counts()
Num_Normal = Num_Count[0]
Num_Attack = Num_Count[1]

# save Test labels and delete them from dataset
Test_Data_labels = Test_Data.iloc[:][Labels_index]
del Test_Data[Labels_index]

# call kmeans function for different K
K_Values = [11,22,44,66,88]
Results =[]
for i in K_Values:
    Results.append(kmeans(i))

df = pd.DataFrame(Results,columns=["DR_Normal:","DR_Attack:","FPR:","FNR:","Efficiency"]).to_csv("Results.csv")