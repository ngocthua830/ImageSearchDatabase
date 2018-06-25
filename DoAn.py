import numpy as np
import cv2
from sklearn.cluster import KMeans
import os
import pickle
from sklearn.externals import joblib
from scipy import spatial
import sys

dataset_path = "/home/thua/git/Instance-Search/oxford/images"
model_path = '/home/thua/TVTTDPT/DoAn/' 
all_feature_list = []
all_feature_array = np.array([])
vocabulary_len = 100
file_feature_list = []
bag_of_words = []
sift = cv2.xfeatures2d.SIFT_create()
#---------------------------------------
###Computing SIFT features
if not os.path.isfile(model_path+"all_feature_array.data"):
        print("Computing sift feature...")
        for filename in os.listdir(dataset_path):
                img = cv2.imread(os.path.join(dataset_path, filename), 0)

                kp = sift.detect(img, None)
                kp, des = sift.compute(img, kp)
                file_feature_list.append((filename, des))
                for kp_feature in des:
                        all_feature_list.append(kp_feature)
                #print(des)
        all_feature_array = np.array(all_feature_list)
        pickle.dump(all_feature_array, open(model_path+"all_feature_array.data", "wb"))
        pickle.dump(file_feature_list, open(model_path+"file_feature_list.data", "wb"))
else:
        print("Load all_feature_array(sift feature) file...")
        all_feature_array = pickle.load(open(model_path+"all_feature_array.data", "rb"))
        file_feature_list = pickle.load(open(model_path+"file_feature_list.data", "rb"))
print("Sift features array:")       
print(all_feature_array)
#-------------------------------------------
###KMeans clustering
kmeans = KMeans(n_clusters=vocabulary_len)
if not os.path.isfile(model_path+"kmeans.model"):
        print("Computing KMeans...")
        kmeans = kmeans.fit(all_feature_array)
        joblib.dump(kmeans, model_path+"kmeans.model")
else:
        print("Load KMeans...")
        kmeans = joblib.load(model_path+"kmeans.model")
#print(kmeans.predict([all_feature_array[0], all_feature_array[1], all_feature_array[2]]))
#print("KMeans Centers:")
#print(kmeans.cluster_centers_)
#---------------------------
###Computing bag of words
for file_list in file_feature_list:
        tmp = np.zeros((vocabulary_len), dtype=int)
        for predict in kmeans.predict(file_list[1]):
                tmp[predict] = tmp[predict] + 1
        bag_of_words.append((file_list[0], tmp))
print(bag_of_words)
#---------------------------
#query_img = cv2.imread("./Dataset/0817.png", 0)
query_img = cv2.imread(sys.argv[1], 0)
kp, des = sift.detectAndCompute(query_img, None)
query_vector = np.zeros((vocabulary_len), dtype=int)
for predict in  kmeans.predict(des):
        query_vector[predict] = query_vector[predict] + 1
        
#----------------------------
angle_best = spatial.distance.cosine(query_vector, bag_of_words[0][1])
angle_list = []
for item in bag_of_words[1:]:
        angle = spatial.distance.cosine(query_vector, item[1])
        angle_list.append((angle, item[0]))
print(angle_list)
angle_list.sort(key=lambda x: x[0])
print(angle_list[0:10])
file_content = angle_list[0][1]
for item in angle_list[1:10]:
        file_content = file_content + "\n" + item[1]
file = open('/home/thua/git/gui-JavaFX/JavaFXApplication3/src/javafxapplication3/result.txt', 'w')
file.write(file_content)

