# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 19:22:38 2019

@author: DEEPAK
"""

#Citations https://stackoverflow.com/questions/14463277/how-to-disable-python-warnings
#suppress all warnings with this
import warnings
warnings.filterwarnings("ignore")

#Including the header from python Library
from sklearn.cluster import KMeans    
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from math import sqrt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB as NB
import statistics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from matplotlib.colors import ListedColormap 
import time
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,ReLU
from keras.layers import Dropout, Activation,BatchNormalization,MaxPool2D
from keras.utils import np_utils
from keras.layers import Convolution2D, MaxPooling2D

#citations https://stackoverflow.com/questions/55109716/c-argument-looks-like-a-single-numeric-rgb-or-rgba-sequence
wcss= []
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

def model_ann_def():
    model_class = Sequential()
    model_class.add(Dense(units =1500,init="uniform",activation ="relu",input_dim=3072))
    model_class.add(Dense(units =1500,init="uniform",activation ="relu"))
    model_class.add(Dense(units =10,init="uniform",activation ="sigmoid"))  
    model_class.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_class

def CIFAR_DATASET():    
    all_vars_0 = io.loadmat(r"C:\Users\DEEPESH\Desktop\Cifar_graduate_assignment\cifar-10-matlab\cifar-10-batches-mat\data_batch_1.mat")
    all_vars_1 = io.loadmat(r"C:\Users\DEEPESH\Desktop\Cifar_graduate_assignment\cifar-10-matlab\cifar-10-batches-mat\data_batch_2.mat")
    all_vars_2 = io.loadmat(r"C:\Users\DEEPESH\Desktop\Cifar_graduate_assignment\cifar-10-matlab\cifar-10-batches-mat\data_batch_3.mat")
    all_vars_3 = io.loadmat(r"C:\Users\DEEPESH\Desktop\Cifar_graduate_assignment\cifar-10-matlab\cifar-10-batches-mat\data_batch_4.mat")
    all_vars_4 = io.loadmat(r"C:\Users\DEEPESH\Desktop\Cifar_graduate_assignment\cifar-10-matlab\cifar-10-batches-mat\data_batch_5.mat")
    all_vars_test = io.loadmat(r"C:\Users\DEEPESH\Desktop\Cifar_graduate_assignment\cifar-10-matlab\cifar-10-batches-mat\test_batch.mat")

    train_0 = all_vars_0.get("data")
    train_1 = all_vars_1.get("data")
    train_2 = all_vars_2.get("data")
    train_3 = all_vars_3.get("data")
    train_4 = all_vars_4.get("data")
    test_5 = all_vars_test.get("data")

    train_dataset_1=np.append(train_0,train_1,axis=0)
    train_dataset_2=np.append(train_dataset_1,train_2,axis=0)
    train_dataset_3=np.append(train_dataset_2,train_3,axis=0)
    train_dataset_4=np.append(train_dataset_3,train_4,axis=0)
    train_dataset=np.append(train_dataset_4,test_5,axis=0)

    label_train_0 = all_vars_0.get("labels")
    label_train_1 = all_vars_1.get("labels")
    label_train_2 = all_vars_2.get("labels")
    label_train_3 = all_vars_3.get("labels")
    label_train_4 = all_vars_4.get("labels")
   
    label_test_5 = all_vars_test.get("labels")

    train_dataset_1_label=np.append(label_train_0,label_train_1,axis=0)
    train_dataset_2_label=np.append(train_dataset_1_label,label_train_2,axis=0)
    train_dataset_3_label=np.append(train_dataset_2_label,label_train_3,axis=0)
    train_dataset_4_label=np.append(train_dataset_3_label,label_train_4,axis=0)
    train_dataset_label=np.append(train_dataset_4_label,label_test_5,axis=0)
   
           
    return train_dataset , train_dataset_label

score_knn =[]
score_Qda =[]
score_Lda =[]
score_nb =[]
score_nb =[]
score_rf =[]
score_dt =[]
score_svm =[]


error_test =[]
error_knn =[]
error_Qda =[]
error_Lda =[]
error_nb =[]
error_rf =[]
error_dt =[]
error_svm =[]
score_test =[]

#Declaration of subplot variables
fig, ax1=plt.subplots()
fig, ax2=plt.subplots()
fig, ax3=plt.subplots()
fig, ax4=plt.subplots()
fig, ax5=plt.subplots()
fig, ax6=plt.subplots()

fig, ax7=plt.subplots()
fig, ax8=plt.subplots()
fig, ax9=plt.subplots()
fig, ax10=plt.subplots()
fig, ax11=plt.subplots()
fig, ax12=plt.subplots()



fig, ax13=plt.subplots()
fig, ax14=plt.subplots()
fig, ax15=plt.subplots()
fig, ax16=plt.subplots()
fig, ax17=plt.subplots()
fig, ax18=plt.subplots()

fig, ax19=plt.subplots()
fig, ax20=plt.subplots()
fig, ax21=plt.subplots()
fig, ax22=plt.subplots()
fig, ax23=plt.subplots()
fig, ax24=plt.subplots()



my_train_data=[]
my_train_label=[]
   
#WARNING: Critical function if called in between lines will erase all the array that holds value for
#plotting and error and scores  
def clear():
    score_knn.clear()
    score_Qda.clear()
    score_Lda.clear()
    score_nb.clear()
    score_rf.clear()
    score_dt.clear()
    score_svm.clear()

    error_rf.clear()
    error_dt.clear()
    error_svm.clear()    
    error_knn.clear()
    error_Qda.clear()
    error_Lda.clear()
    error_nb.clear()
   


#Function that fits scores and check the error for the model and returns score/accuracy and error that will be used in Loop_for_computataion
#Also be considered as the main logic or critical unit
#CITATIONS :https://www.youtube.com/watch?v=gJo0uNL-5Qw&t=470s took only concept references
def fit_predict_check_score_and_error(universalmodel,X_train,X_test,y_train,y_test):
    universalmodel.fit(X_train,y_train)
    prediction = universalmodel.predict(X_test)  
    return(metrics.accuracy_score(y_test,prediction),np.mean(y_test != prediction))
    #Note:I see that the  above function is performing similar to  
    #cross_validate(universalmodel,X, y, scoring="accuracy", cv=partitioner)
   
   
#Enable/call this function if you needed to check the Dimensional transform to be applied  
def Dimension_reduction_transform(universalmodel,X_train,X_test,y_train,y_test):
    X_train=universalmodel.fit_transform(X_train,y_train)
    X_test=universalmodel.transform(X_test)
    return X_train ,X_test

k_range = range(1,6,1)
knn_check = range(1,25,5)
   
#Loop for computation of stratified strait fit algorithmn and computes the
#LDA QDA NB and KNN for values 1,5,10
def Loop_for_computataion(my_train_data ,my_train_label,model_cnn,status,iris_cifar):
    #Applying the K Fold using 5 splits as mentioned question
    lda = LDA()
    qda = QDA()
    nb  = NB()
    rf =RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
    svm=SVC(kernel='rbf',random_state=0)
    dt = DecisionTreeClassifier(criterion='entropy',random_state=0)
       
   
    #CITATIONS:https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html
    #Even if i chnage the train and test size (Ex:Train 80% and test 20% I find slight variation in op I,e I mean
    #i have cross verified changing the sizes and fit it performs correctly)
    Kfold_stratified_shuffleop = StratifiedShuffleSplit(n_splits=5,train_size=0.8,test_size=0.2, random_state=0)  
    for training_values, testing_values in Kfold_stratified_shuffleop.split(my_train_data, my_train_label):
       
        #using the standard naming convention X_train X_test,y_train,y_test
        X_train, X_test = my_train_data[training_values], my_train_data[testing_values]
        y_train, y_test = my_train_label[training_values], my_train_label[testing_values]
       
        print("\n")
        print("TRAINING VALUES:", training_values, "TESTING VALUES:", testing_values)
        print("\n")

        if status == 3:
            print("ENABLING PCA")
            meshgrid_pca_analysis(X_train,X_test,y_train,y_test,lda,qda,nb,rf,dt,svm,1,iris_cifar)
        elif status ==1:
            compute_logic_supervised_learning(X_train,X_test,y_train,y_test,lda,qda,nb,rf,dt,svm,1)
        elif status ==2:
            cnn_split = list(StratifiedShuffleSplit(n_splits=2, test_size=0.1).split(X_train, y_train))
            idx_tr, idx_val = cnn_split[0]
            X_val, y_val = X_train[idx_val], y_train[idx_val]
            X_tr, y_tr = X_train[idx_tr], y_train[idx_tr]
       
            X_val = X_val.reshape(len(X_val),32,32,3)
            X_tr = X_tr.reshape(len(X_tr),32,32,3)
            X_test = X_test.reshape(len(X_test),32,32,3)
           
            y_val = np_utils.to_categorical(y_val, 10)
            y_tr = np_utils.to_categorical(y_tr, 10)
            model_cnn.fit(X_tr, y_tr, validation_data=(X_val, y_val))      
            model_cnn.predict(X_test)
        else:
            print("No proper selection")
             
#Function that defines the computations  
def compute_logic_supervised_learning(X_train,X_test,y_train,y_test,lda,qda,nb,rf,dt,svm,model_ann):    
        score,error = fit_predict_check_score_and_error(lda,X_train, X_test,y_train,y_test)
        score_Lda.append(score)
        error_Lda.append(error)
       
        score,error=fit_predict_check_score_and_error(qda,X_train, X_test,y_train,y_test)
        score_Qda.append(score)
        error_Qda.append(error)
             
        score,error=fit_predict_check_score_and_error(nb,X_train, X_test,y_train,y_test)
        error_nb.append(error)
        score_nb.append(score)
       
        score,error=fit_predict_check_score_and_error(rf,X_train, X_test,y_train,y_test)
        error_rf.append(error)
        score_rf.append(score)

        score,error=fit_predict_check_score_and_error(dt,X_train, X_test,y_train,y_test)
        error_dt.append(error)
        score_dt.append(score)
       
        score,error=fit_predict_check_score_and_error(svm,X_train, X_test,y_train,y_test)
        error_svm.append(error)
        score_svm.append(score)
       
       
#        This loop is applied as We need to validateKnn when neighbors is 1, 5, 10
        for i in range(1,15,5):    
            knn = KNeighborsClassifier(n_neighbors=i,n_jobs=-1)
            score,error=fit_predict_check_score_and_error(knn,X_train, X_test,y_train,y_test)
            error_knn.append(error)
            score_knn.append(score)
#           
           
#print functions that print all the values of score and error
def prints_func():      
   
#Enable if needed for detailing prints    
    print("------------------LDA-K-FOLD Mean Accuracy------------------------------\n")
    print(statistics.mean(score_Lda))
    print("------------------QDA-K-FOLD Mean Accuracy------------------------------\n")
    print(statistics.mean(score_Qda))
    print("------------------NB-K-FOLD Mean Accuracy--------------------------------\n")
    print(statistics.mean(score_nb))
    print("------------------KNN- 1-K-FOLD Mean Accuracy----------------------------\n")
    print(statistics.mean(score_knn[0:5]))
    print("------------------KNN- 5-K-FOLD Mean Accuracy----------------------------\n")
    print(statistics.mean(score_knn[5:10]))
    print("------------------KNN- 10-K-FOLD Mean Accuracy----------------------------\n")
    print(statistics.mean(score_knn[10:15]))
    print("------------------Random forest-K-FOLD Mean Accuracy------------------------------\n")
    print(statistics.mean(score_rf))
    print("------------------Decision Tree -K-FOLD Mean Accuracy------------------------------\n")
    print(statistics.mean(score_dt))
    print("------------------SVM -K-FOLD Mean Accuracy--------------------------------\n")
    print(statistics.mean(score_svm))
   
    print("------------------LDA-K-FOLD Mean Error----------------------------\n")
    print(statistics.mean(error_Lda))
    print("------------------QDA-K-FOLD Mean Error----------------------------\n")
    print(statistics.mean(error_Qda))
    print("------------------NB-K-FOLD Mean Error----------------------------\n")
    print(statistics.mean(error_nb))
    print("------------------KNN-K-FOLD Mean Error----------------------------\n")
    print(statistics.mean(error_knn[0:5]))
    print("------------------KNN-5-K-FOLD Mean Error----------------------------\n")
    print(statistics.mean(error_knn[5:10]))
    print("------------------KNN-10-K-FOLD Mean Error----------------------------\n")
    print(statistics.mean(error_knn[10:15]))
    print("------------------RF -K-FOLD Mean Error----------------------------\n")
    print(statistics.mean(error_rf))
    print("------------------DT -K-FOLD Mean Error----------------------------\n")
    print(statistics.mean(error_dt))
    print("------------------SVM-K-FOLD Mean Error----------------------------\n")
    print(statistics.mean(error_svm))
   
#function that is used for plotting the graphs
def plot_func(plot_area,kfoldvalue,error_score,check,Q3_Q1,label):
    plot_area.plot(kfoldvalue, error_score, label=label)
    plot_area.set_xlabel('Value of K fold')
 
    if(check==1):
        plot_area.set_ylabel('score plot')
    else:
        plot_area.set_ylabel('error plot')
    plot_area.legend()
       
error=0
score=1
   
#function that is used for calling the plot that takes the subplots as the arguments
#i find that plotting it in one graph is not viewable hence making
#GROUP-1-LDA AND KNN
#GROUP-2 NB
#GROUP-3 KNN 1,5,10
from sklearn.decomposition import IncrementalPCA

def PCA_reduction(final_zero_data):
    m, n = final_zero_data.shape
    n_components=2

    ipca = IncrementalPCA(
    copy=False,
    n_components=n_components,
    batch_size=(m)
    )
    X_train_recuced_ipca = ipca.fit_transform(final_zero_data)
    return X_train_recuced_ipca


def smaller_plot(ost_plot,tnd_plot,trd_plot,fth_plot,fvth_plot,cth_plot,svth_plot,eth_plot,nth_plot,tenth_plot,
                 ele_plot,Q3_Q1):
    plot_func(ost_plot,k_range, error_Lda,error,Q3_Q1,label='Plotting error w.r.t to Kfold LDA')
    plot_func(ost_plot,k_range, error_Qda,error,Q3_Q1,label='Plotting error w.r.t to Kfold QDA')
    plot_func(tnd_plot,k_range, error_nb,error,Q3_Q1,label='Plotting error w.r.t to Kfold NB-1')
    
    plot_func(trd_plot,k_range, error_knn[0:5],error,Q3_Q1,label='Plotting error w.r.t to Kfold KNN-1')
    plot_func(trd_plot,k_range, error_knn[5:10],error,Q3_Q1,label='Plotting error w.r.t to Kfold KNN-5')
    plot_func(trd_plot,k_range, error_knn[10:15],error,Q3_Q1,label='Plotting error w.r.t to Kfold KNN-10')
   
    plot_func(svth_plot,k_range, error_rf,error,Q3_Q1,label='Plotting error w.r.t to Kfold Random forest')
    plot_func(svth_plot,k_range, error_dt,error,Q3_Q1,label='Plotting error w.r.t to Kfold Decision tree')
    plot_func(eth_plot,k_range, error_svm,error,Q3_Q1,label='Plotting error w.r.t to Kfold SVM ') 

    plot_func(fth_plot,k_range, score_Lda,score,Q3_Q1,label='Plotting score w.r.t to Kfold LDA')
    plot_func(fth_plot,k_range, score_Qda,score,Q3_Q1,label='Plotting score w.r.t to Kfold QDA')
    plot_func(fvth_plot,k_range, score_nb,score,Q3_Q1,label='Plotting score w.r.t to Kfold NB')
    
    plot_func(cth_plot,k_range, score_knn[0:5],score,Q3_Q1,label='Plotting score w.r.t to Kfold KNN-1')
    plot_func(cth_plot,k_range, score_knn[5:10],score,Q3_Q1,label='Plotting score w.r.t to Kfold KNN-5')
    plot_func(cth_plot,k_range, score_knn[10:15],score,Q3_Q1,label='Plotting score w.r.t to Kfold KNN-10')
   
    plot_func(nth_plot,k_range, score_rf,score,Q3_Q1,label='Plotting score w.r.t to Kfold Random forest')
    plot_func(tenth_plot,k_range, score_dt,score,Q3_Q1,label='Plotting score w.r.t to Kfold Decision tree')
    plot_func(ele_plot,k_range, score_svm,score,Q3_Q1,label='Plotting score w.r.t to Kfold SVM ')

from sklearn.preprocessing import StandardScaler    


def meshgrid_pca_analysis(X_train,X_test,y_train,y_test,lda,qda,nb,rf,dt,svm,model_ann,value):
##applying PCA for dimensionality reduction before
##providing to supervised learning algorithm 
    sc = StandardScaler()    
    X_train = sc.fit_transform(X_train) 
    X_test = sc.transform(X_test) 
    compute_logic_supervised_learning(X_train,X_test,y_train,y_test,lda,qda,nb,rf,dt,svm,model_ann) 
    plotting_meshgrid(X_train,X_test,y_train,y_test,lda,"Liner Discrminant plot",value)
    plotting_meshgrid(X_train,X_test,y_train,y_test,qda,"Quadratic Discriminat plot",value)
    plotting_meshgrid(X_train,X_test,y_train,y_test,nb,"nayies bayes plot plot",value)
    plotting_meshgrid(X_train,X_test,y_train,y_test,rf,"Random forest plot",value)
    plotting_meshgrid(X_train,X_test,y_train,y_test,dt,"Decision treep lot plot",value)
    plotting_meshgrid(X_train,X_test,y_train,y_test,svm,"support vector plot",value)


#citations:https://www.geeksforgeeks.org/principal-component-analysis-with-python/    
def plotting_meshgrid(X_train,X_test,y_train,y_test,classifier,label,Iris_cifar):
    classifier.fit(X_train, y_train)

## Predicting the test set 
## result through scatter plot    
    X_set, y_set = X_test, y_test 
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, 
                     stop = X_set[:, 0].max() + 1, step = 0.01), 
                     np.arange(start = X_set[:, 1].min() - 1, 
                     stop = X_set[:, 1].max() + 1, step = 0.01)) 
    if Iris_cifar ==1:
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), 
             X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, 
              cmap = ListedColormap(('yellow','aquamarine','orange'))) 
    else:
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), 
             X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, 
             cmap = ListedColormap(('yellow','aquamarine','red','blue','orange','white','black','pink','purple','green'))) 
        
    
    plt.xlim(X1.min(), X1.max()) 
    plt.ylim(X2.min(), X2.max()) 
    
    if Iris_cifar ==1:
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
            c = ListedColormap(('green','purple','red'))(i), label = j) 
    else:
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
            c = ListedColormap(('green','purple','pink','black','white','orange','blue','red','aquamarine','yellow'))(i), label = j) 
       
    
    plt.title(label) 
    plt.xlabel('PC1') # for Xlabel 
    plt.ylabel('PC2') # for Ylabel 
    plt.legend() # to show legend 
    plt.show() 


Q3_Q1=0
#Citations:
#https://stackoverflow.com/questions/43357507/pca-memory-error-in-sklearn-alternative-dim-reduction#comment88630572_43513435

def pca_reduction(final_zero_data):
    m, n = final_zero_data.shape
    n_components=2
    ipca = IncrementalPCA(
            copy=False,
            n_components=n_components,
            batch_size=(m)
            )
    X_train_recuced_ipca = ipca.fit_transform(final_zero_data)
    return X_train_recuced_ipca


from sklearn.preprocessing import OneHotEncoder,LabelEncoder

def iris():
    iris_df  = pd.read_csv(r"C:\Users\DEEPESH\Desktop\Cifar_graduate_assignment\iris_data_set.csv")                                                                                                                                                                                                                            
    X =  iris_df[['sepal length', 'sepal width', 'petal length', 'petal width']]
    y =  iris_df[['species']]


    label_encoder_y = LabelEncoder()
    y = label_encoder_y.fit_transform(y)

    X=np.array(X)
    y=np.array(y)
    X=pca_reduction(X)
    Loop_for_computataion(X,y,0,3,1)
    prints_func()

    
##########################################################
#same values of Final One and Final Zero is used        
final_zero_data,final_one_label=CIFAR_DATASET()

complete_data=np.append(final_zero_data,final_one_label,axis=1)
complete_data=pd.DataFrame(complete_data)

df=complete_data

final_zero_data=df.iloc[:,0:3072]
final_one_label=df.iloc[:,-1]

final_zero_data=pd.DataFrame(final_zero_data)
final_one_label=pd.DataFrame(final_one_label)

final_zero_data=final_zero_data.iloc[:60000,:]
final_one_label=final_one_label.iloc[:,-1]
final_one_label=final_one_label.iloc[:60000,]
final_zero_data=np.array(final_zero_data)
final_one_label=np.array(final_one_label)
##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
##############################
#Implementation on Iris Datset to show the plots of each 
#algorithm and its working and measure Error and Accuracy
      
def iris_example(final_zero_data,final_one_label):
    clear()
    print("Timer started")
    iris()
    print("Timer ended and It took seconds")
    clear()

###############################
#CIFAR DATASET COMPUTATION WITHOUT PCA OR DIMENSIONALITY REDUCTION FOR 10000 SAMPLES
def without_pca1000smaples(final_zero_data,final_one_label):
    clear()
    print("Timer started")
    final_zero_data=pd.DataFrame(final_zero_data)
    final_one_label=pd.DataFrame(final_one_label)
    final_zero_data=final_zero_data.iloc[:10000,:]
    final_one_label=final_one_label.iloc[:,-1]
    final_one_label=final_one_label.iloc[:10000,]
    final_zero_data=np.array(final_zero_data)
    final_one_label=np.array(final_one_label)
    start = time.time()
    Loop_for_computataion(final_zero_data,final_one_label,0,1,1)
    smaller_plot(ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,Q3_Q1)
    prints_func()
    end = time.time()
    print("Timer ended and It took seconds")
    print(end - start)
    clear()


###############################
#CIFAR DATASET COMPUTATION WITH PCA FOR 10000 SAMPLES    
def with_pca1000samples(final_zero_data,final_one_label):
    clear()
    print("Timer started")
    final_zero_data=pd.DataFrame(final_zero_data)
    final_one_label=pd.DataFrame(final_one_label)
    final_zero_data=final_zero_data.iloc[:10000,:]
    final_one_label=final_one_label.iloc[:,-1]
    final_one_label=final_one_label.iloc[:10000,]
    final_zero_data=np.array(final_zero_data)
    final_one_label=np.array(final_one_label)
    print("Timer started")
    start = time.time()
    final_zero_data=pca_reduction(final_zero_data)
    Loop_for_computataion(final_zero_data,final_one_label,0,1,1)
    prints_func()
    end = time.time()
    print("Timer ended and It took seconds")
    print(end - start)
    clear()

###############################
#CIFAR DATASET COMPUTATION WITH PCA MESHGRID FOR 10000 SAMPLES    
def withpca10000samples_meshgrid(final_zero_data,final_one_label):
    clear()
    print("Timer started")
    final_zero_data=pd.DataFrame(final_zero_data)
    final_one_label=pd.DataFrame(final_one_label)
    final_zero_data=final_zero_data.iloc[:10000,:]
    final_one_label=final_one_label.iloc[:,-1]
    final_one_label=final_one_label.iloc[:10000,]
    final_zero_data=np.array(final_zero_data)
    final_one_label=np.array(final_one_label)
    start = time.time()
    final_zero_data=pca_reduction(final_zero_data)
    Loop_for_computataion(final_zero_data,final_one_label,0,3,0)
    prints_func()
    end = time.time()
    print("Timer ended and It took seconds")
    print(end - start)
    clear()

    
###############################
#CIFAR DATASET COMPUTATION WITH PCA  60000 SAMPLES    
def withpca60000samples(final_zero_data,final_one_label):    
    print("Timer started")
    start = time.time()
    final_zero_data=pd.DataFrame(final_zero_data)
    final_one_label=pd.DataFrame(final_one_label)
    final_zero_data=pca_reduction(final_zero_data)
    Loop_for_computataion(final_zero_data,final_one_label,0,1,0)
    smaller_plot(ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,Q3_Q1)
    prints_func()
    end = time.time()
    print("Timer ended and It took seconds")
    print(end - start)
    clear()

#compute the CNN
def cnn_model():
     
    model = Sequential()
    model.add(BatchNormalization()) 
    model.add(Conv2D(32,kernel_size=(3,3),input_shape=(32, 32, 3),activation='relu'))
    model.add(Conv2D(64,kernel_size=(3,3),input_shape=(32, 32, 3),activation='relu'))
    model.add(Conv2D(64,kernel_size=(3,3),input_shape=(32, 32, 3),activation='relu'))
    model.add(Conv2D(64,kernel_size=(3,3),input_shape=(32, 32, 3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(64,kernel_size=(3,3),input_shape=(32, 32, 3),activation='relu'))
    model.add(Flatten())
    model.add(Dense(units=10, activation="softmax"))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=["accuracy"])

    return model

########################## CNN CIFAR DATASET################################
def cnn_performance():
    complete_data=[]
    train_dataset,train_dataset_label=CIFAR_DATASET()
    model_cnn=cnn_model()

    dataset=pd.DataFrame(train_dataset)
    dataset_label=pd.DataFrame(train_dataset_label)
    complete_data=np.append(dataset,dataset_label,axis=1)
    complete_data=pd.DataFrame(complete_data)

    df = complete_data.sort_values(complete_data.columns[3072])

    dataset=df.iloc[:,0:3072]
    dataset_label=df.iloc[:,-1]

    dataset=pd.DataFrame(dataset)
    dataset_label=pd.DataFrame(dataset_label)

    train_data=dataset.iloc[:50000,:]
    train_label=dataset_label.iloc[:,-1]
    train_label=train_label.iloc[:50000,]

    train_data=np.array(train_data)
    train_label=np.array(train_label)

    train_label= np_utils.to_categorical(train_label, 10)
    train_data = train_data.reshape(len(train_data),32,32,3)

    test_data=dataset.iloc[50000:,:]
    test_label=dataset_label.iloc[:,-1]
    test_label=test_label.iloc[50000:,]

    test_data=np.array(test_data)
    test_label=np.array(test_label)

    test_label = np_utils.to_categorical(test_label, 10)
    test_data = test_data.reshape(len(test_data),32,32,3)

    test_data=np.array(test_data)
    test_label=np.array(test_label)

    model_cnn.fit(train_data,train_label,batch_size=32,epochs=15)
    loss,accuracy=model_cnn.evaluate(test_data,test_label)
    print("CNN ACCURACY")
    print(accuracy)
    print("CNN LOSS")
    print(loss)
    print("CNN ERROR")
    print(1-accuracy)


def Kmeans(final_zero_data):
    
    final_zero_data=pd.DataFrame(final_zero_data)
    final_zero_data=final_zero_data.iloc[:10000,:]
    X_train_recuced_ipca=PCA_reduction(final_zero_data)
#    meshgrid(X_train_recuced_ipca,final_one_label)
#    for i in range(1,100
#                   ):
#        print("KMEANS")
#        kmeans=KMeans(n_clusters =i,init='k-means++',max_iter =300,n_init=10,random_state=0)
#        kmeans.fit(X_train_recuced_ipca)
#        wcss.append(kmeans.inertia_)
#    
#   
#    plt.plot(range(1,100),wcss)
    #    plt.title("the elbow method")
    kmeans=KMeans(n_clusters =10,init='k-means++',max_iter =300,n_init=10,random_state=0)
    y_means=kmeans.fit_predict(X_train_recuced_ipca)

    plt.scatter(X_train_recuced_ipca[y_means ==0,0],X_train_recuced_ipca[y_means ==0,1],s=10,c='red',label= 'zero')
    plt.scatter(X_train_recuced_ipca[y_means ==1,0],X_train_recuced_ipca[y_means ==1,1],s=10,c='yellow',label= 'one')
    plt.scatter(X_train_recuced_ipca[y_means ==2,0],X_train_recuced_ipca[y_means ==2,1],s=10,c='blue',label= 'two')
    plt.scatter(X_train_recuced_ipca[y_means ==3,0],X_train_recuced_ipca[y_means ==3,1],s=10,c='green',label= 'three')
    plt.scatter(X_train_recuced_ipca[y_means ==4,0],X_train_recuced_ipca[y_means ==4,1],s=10,c='orange',label= 'four')
    plt.scatter(X_train_recuced_ipca[y_means ==5,0],X_train_recuced_ipca[y_means ==5,1],s=10,c='pink',label= 'five')
    plt.scatter(X_train_recuced_ipca[y_means ==6,0],X_train_recuced_ipca[y_means ==6,1],s=10,c='violet',label= 'six')
    plt.scatter(X_train_recuced_ipca[y_means ==7,0],X_train_recuced_ipca[y_means ==7,1],s=10,c='grey',label= 'seven')
    plt.scatter(X_train_recuced_ipca[y_means ==8,0],X_train_recuced_ipca[y_means ==8,1],s=10,c='purple',label= 'eight')
    plt.scatter(X_train_recuced_ipca[y_means ==9,0],X_train_recuced_ipca[y_means ==9,1],s=10,c='white',label= 'nine')
    
    
    plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,0],s=100,c='yellow',label='Centroids')


################################################################
################################################################
################################################################
################################################################
################################################################
 
#for better view Kindly please run the code by enbaling each function one after another     
iris_example(final_zero_data,final_one_label)
without_pca1000smaples(final_zero_data,final_one_label)
with_pca1000samples(final_zero_data,final_one_label)
withpca10000samples_meshgrid(final_zero_data,final_one_label)
withpca60000samples(final_zero_data,final_one_label) 
cnn_performance()
Kmeans(final_zero_data)



