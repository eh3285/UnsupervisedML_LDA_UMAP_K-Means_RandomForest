#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import random 
random.seed(109)

import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer 

music_data=os.getcwd()+'/Desktop/Capstone_project/musicData.csv'
m_data=pd.read_csv(music_data)

train_data_lst=[]
test_data_lst=[]

for each_genre in m_data['music_genre'].unique():
    g_data=m_data[m_data['music_genre']==each_genre]

    if len(g_data)>=5000:
        train_data=g_data.sample(n=4500,random_state=109)
        other=g_data.drop(train_data.index)
        test_data=other.sample(n=500,random_state=109)
        train_data_lst.append(train_data)
        test_data_lst.append(test_data)

df_train=pd.concat(train_data_lst, ignore_index=True).sample(frac=1,
                                                 random_state=109)
df_test=pd.concat(test_data_lst, ignore_index=True).sample(frac=1,
                                                 random_state=109)

print(df_train.columns)
        
not_including_col=["instance_id","artist_name","track_name","obtained_date",
                   "music_genre"]

X_train=df_train.drop(columns=not_including_col) # to separate the features
y_train=df_train['music_genre'] # to separate the output values 

X_test=df_test.drop(columns=not_including_col) # to separate the features
y_test=df_test['music_genre'] # to separate the output values 

X_train=X_train.replace('?',np.nan)
X_test=X_test.replace('?',np.nan)

categorical_feat=["key","mode"]
numerical_feat=["popularity","acousticness","danceability","duration_ms",
                "energy","instrumentalness","liveness","loudness",
                "speechiness","tempo","valence"]

categorical_imput=SimpleImputer(strategy='most_frequent')
X_train[categorical_feat]=( #only fit on the train set
            categorical_imput.fit_transform(X_train[categorical_feat]))
X_test[categorical_feat]=( #only fit on the train set
            categorical_imput.transform(X_test[categorical_feat]))
numerical_imput=SimpleImputer(strategy='median')
X_train[numerical_feat]=( #only fit on the train set
            numerical_imput.fit_transform(X_train[numerical_feat]))
X_test[numerical_feat]=( #only fit on the train set
            numerical_imput.transform(X_test[numerical_feat]))

X_train=pd.get_dummies(X_train,columns=["key","mode"],drop_first=True)
X_test=pd.get_dummies(X_test,columns=["key","mode"],drop_first=True)
X_train,X_test=X_train.align(X_test,join='left',axis=1,fill_value=0)

feature_names=X_train.columns
numerical_feat=[feat for feat in numerical_feat if feat in feature_names]
standard_scaler=StandardScaler()
X_train[numerical_feat]=standard_scaler.fit_transform(X_train[numerical_feat])
X_test[numerical_feat]=standard_scaler.transform(X_test[numerical_feat])

y_label_encoder=LabelEncoder()
y_encoded_train=y_label_encoder.fit_transform(y_train)
y_encoded_test=y_label_encoder.transform(y_test)

print("total genres count:",len(y_label_encoder.classes_))
print("X_train shape", X_train.shape)
print("X_test shape", X_test.shape)
print("y_train shape", y_encoded_train.shape)
print("y_test shape", y_encoded_test.shape)

print("Y test missing value count:",pd.isna(y_encoded_test).sum())
print("Y train missing value count:",pd.isna(y_encoded_train).sum())
print("X test missing value count:",pd.isna(X_test).sum())
print("X train missing value count:",pd.isna(X_train).sum())

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import silhouette_score, adjusted_rand_score

n_lda_comp=len(y_label_encoder.classes_)-1
lda_model=LDA(n_components=n_lda_comp)
X_lda_train=lda_model.fit_transform(X_train.values,y_encoded_train)
print("Explained ratio for LDA:",
      lda_model.explained_variance_ratio_)
print("Variance explained by LDA:",
      lda_model.explained_variance_ratio_.sum())

t_sne=TSNE(n_components=2, random_state=109,perplexity=50,max_iter=2000)
X_tsne_train=t_sne.fit_transform(X_lda_train)

print("Output TSNE shape",X_tsne_train.shape)


import umap

umap_model=umap.UMAP(n_neighbors=15,min_dist=0.1,random_state=109)
X_umap_train=umap_model.fit_transform(X_lda_train)

print("Output UMAP shape",X_umap_train.shape)

k_means=KMeans(n_clusters=10,random_state=109,n_init=10)
label_clusters=k_means.fit_predict(X_tsne_train)

label_clusters_2=k_means.fit_predict(X_umap_train)

eval_clustering=silhouette_score(X_tsne_train, label_clusters) 
match_true_genre=adjusted_rand_score(y_encoded_train,label_clusters)

eval_clustering_2=silhouette_score(X_umap_train, label_clusters_2) 
match_true_genre_2=adjusted_rand_score(y_encoded_train,label_clusters_2)

print("Silhouette score for t-SNE:",eval_clustering)
print("Adjusted random index for t-SNE:",match_true_genre)

print("Silhouette score for UMAP:",eval_clustering_2)
print("Adjusted random index for UMAP:",match_true_genre_2)

lst_silhouette_scores=[]
lst_ari_scores=[]
K_range=range(2,21)
for k_val in K_range:
    K_means=KMeans(n_clusters=k_val,random_state=109,n_init=10)
    predicted_labels=K_means.fit_predict(X_tsne_train)
    
    silhouette=silhouette_score(X_tsne_train,predicted_labels)
    ari=adjusted_rand_score(y_encoded_train,predicted_labels)
    
    lst_ari_scores.append(ari)
    lst_silhouette_scores.append(silhouette)

plt.plot(range(2,21),lst_silhouette_scores,'o-',)
plt.xlabel("K number of clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for different K values for t-SNE")
plt.tight_layout()
plt.show()

plt.plot(range(2,21),lst_ari_scores,'o-',)
plt.xlabel("K number of clusters")
plt.ylabel("ARI Score")
plt.title("Adjusted Random Score for different K values for t-SNE")
plt.tight_layout()
plt.show()

optimal_k_ari=K_range[np.argmax(lst_ari_scores)]
optimal_k_silh=K_range[np.argmax(lst_silhouette_scores)]
print("Optimal Ks when using t-SNE are:",optimal_k_ari,"max ari",
      optimal_k_silh,"max silhouette score")

lst_silhouette_scores_2=[]
lst_ari_scores_2=[]
K_range=range(2,21)
for k_val in K_range:
    K_means=KMeans(n_clusters=k_val,random_state=109,n_init=10)
    predicted_labels_2=K_means.fit_predict(X_umap_train)
    
    silhouette=silhouette_score(X_umap_train,predicted_labels_2)
    ari=adjusted_rand_score(y_encoded_train,predicted_labels_2)
    
    lst_ari_scores_2.append(ari)
    lst_silhouette_scores_2.append(silhouette)

plt.plot(range(2,21),lst_silhouette_scores_2,'o-',)
plt.xlabel("K number of clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for different K values for UMAP")
plt.tight_layout()
plt.show()

plt.plot(range(2,21),lst_ari_scores_2,'o-',)
plt.xlabel("K number of clusters")
plt.ylabel("ARI Score")
plt.title("Adjusted Random Score for different K values for UMAP")
plt.tight_layout()
plt.show()

optimal_k_ari_2=K_range[np.argmax(lst_ari_scores_2)]
optimal_k_silh_2=K_range[np.argmax(lst_silhouette_scores_2)]
print("Optimal Ks when using UMAP are:",optimal_k_ari_2,"max ari",
      optimal_k_silh_2, "max silhouette score")
figure1,axes1=plt.subplots(1,2,figsize=(20,8))

scatter_3=axes1[0].scatter(X_umap_train[:,0],X_umap_train[:,1],
                          c=y_encoded_train,cmap='tab10',alpha=0.6,s=10)
axes1[0].set_title('UMAP Visualization For Clusters In True Genre Colors')
axes1[0].set_xlabel('UMAP 1st Component')
axes1[0].set_ylabel('UMAP  2nd Component')

handle_3, _=scatter_3.legend_elements()
axes1[0].legend(handle_3,y_label_encoder.classes_,title="Genres", 
               loc='best',fontsize=8)
cmap=matplotlib.colormaps.get_cmap('tab10').resampled(10)
scatter_4=axes1[1].scatter(X_umap_train[:,0],X_umap_train[:,1],
                          c=label_clusters_2,cmap=cmap, 
                          alpha=0.6,s=10)
axes1[1].set_title('UMAP Visualization For Clusters In KMeans Colors')
axes1[1].set_xlabel('UMAP 1st Component')
axes1[1].set_ylabel('UMAP 2nd Component')
handle_4, _=scatter_4.legend_elements()
axes1[1].legend(handle_4,[f'Cluster{i}'for i in range (10)],title="Clusters", 
               loc='best',fontsize=8)

plt.tight_layout()
plt.show()

figure2,axes2=plt.subplots(1,2,figsize=(20,8))

scatter_1=axes2[0].scatter(X_tsne_train[:,0],X_tsne_train[:,1],
                          c=y_encoded_train,cmap='tab10',alpha=0.6,s=10)
axes2[0].set_title('t-SNE Visualization For Clusters In True Genre Colors')
axes2[0].set_xlabel('t-SNE 1st Component')
axes2[0].set_ylabel('t-SNE  2nd Component')

handle_1, _=scatter_1.legend_elements()
axes2[0].legend(handle_1,y_label_encoder.classes_,title="Genres", 
               loc='best',fontsize=8)
cmap=matplotlib.colormaps.get_cmap('tab10').resampled(10)
scatter_2=axes2[1].scatter(X_tsne_train[:,0],X_tsne_train[:,1],
                          c=label_clusters,cmap=cmap, 
                          alpha=0.6,s=10)
axes2[1].set_title('t-SNE Visualization For Clusters In KMeans Colors')
axes2[1].set_xlabel('t-SNE 1st Component')
axes2[1].set_ylabel('t-SNE 2nd Component')
handle_2, _=scatter_2.legend_elements()
axes2[1].legend(handle_1,[f'Cluster{i}'for i in range (10)],title="Clusters", 
               loc='best',fontsize=8)

plt.tight_layout()
plt.show()

'''
Notable output: (not including the full print statements)
Explained ratio for LDA: 
    [6.37551366e-01 1.70557243e-01 1.15827475e-01 3.75890519e-02
 2.40200392e-02 7.80291710e-03 5.76312980e-03 7.58430109e-04
 1.30347573e-04]
Variance explained by LDA: 0.9999999999999998
Output TSNE shape (45000, 2)
Output UMAP shape (45000, 2)
Silhouette score for t-SNE: 0.38919222
Adjusted random index for t-SNE: 0.16592026263283727
Silhouette score for UMAP: 0.43953508
Adjusted random index for UMAP: 0.1911517800592779
Optimal Ks when using t-SNE are: 4 max ari 3 max silhouette score
Optimal Ks when using UMAP are: 10 max ari 2 max silhouette score
'''
from sklearn.model_selection import RandomizedSearchCV

X_lda_test=lda_model.transform(X_test)
X_umap_test=umap_model.transform(X_lda_test)


X2_train, X_val, y2_train, y_val = train_test_split(X_umap_train, 
                                                    y_encoded_train, 
         test_size=0.2, stratify=y_encoded_train, random_state=109)

print("Val set size:",X_val.shape[0])
print("train set size:",X2_train.shape[0])
print("Test set size:",X_test.shape[0])

hyper_param_search={'n_estimators' : [100,200,300,500],
                    'max_depth' : [10,20,30, None],
                    'min_samples_split' : [2,5,10],
                    'min_samples_leaf' : [1,2,4],
                    'max_features' : ['sqrt','log2'],
                    'bootstrap' : [True],
                    'class_weight' : ['balanced', None]
    }

base_random_forest=RandomForestClassifier(random_state=109,n_jobs=-1,
                                          verbose=1)
parameter_search=RandomizedSearchCV(estimator=base_random_forest,
                                    param_distributions=hyper_param_search,
                                    n_iter=20, cv=3, 
                                    scoring='accuracy', n_jobs=-1,
                                    verbose=2, random_state=109,
                                    return_train_score=True)

parameter_search.fit(X2_train,y2_train)
print("best parameters:",parameter_search.best_params_)
print("best CV accuracy:",parameter_search.best_score_)

best_random_forest=parameter_search.best_estimator_
predictions_validation=best_random_forest.predict(X_val)
predictions_proba_val=best_random_forest.predict_proba(X_val)
accuracy_validation=accuracy_score(y_val, predictions_validation)

print("Validation Set Accuracy:",accuracy_validation)

validation_auc_roc=roc_auc_score(y_val, predictions_proba_val,
                          multi_class='ovr', average='weighted')

print("ROC AUC score for Validation Set:",validation_auc_roc)

predictions_testset=best_random_forest.predict(X_umap_test)
predictions_proba_test=best_random_forest.predict_proba(X_umap_test)
accuracy_testset=accuracy_score(y_encoded_test, predictions_testset)

print("Test Set Accuracy:",accuracy_testset)

testset_auc_roc=roc_auc_score(y_encoded_test, predictions_proba_test,
                          multi_class='ovr', average='weighted')

print("ROC AUC score for Test Set:",testset_auc_roc)

classes_unique=np.unique(y_encoded_train)
n_classes=10

from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc
from itertools import cycle 

binarize_y_val=label_binarize(y_val,classes=classes_unique)
binarize_y_test=label_binarize(y_encoded_test, classes=classes_unique)

val_fpr=dict()
val_tpr=dict()
val_roc_auc=dict()

for classes in range (n_classes):
    val_fpr[classes],val_tpr[classes],_=roc_curve(binarize_y_val[:,classes],
                                predictions_proba_val[:,classes])
    val_roc_auc[classes]=auc(val_fpr[classes],val_tpr[classes])
    
test_fpr=dict()
test_tpr=dict()
test_roc_auc=dict()

for classes in range (n_classes):
    test_fpr[classes],test_tpr[classes],_=roc_curve(binarize_y_test[:,classes],
                                predictions_proba_test[:,classes])
    test_roc_auc[classes]=auc(test_fpr[classes],test_tpr[classes])
    
plt.figure(figsize=(12,8))
colors=cycle(['purple','brown','pink','gray','olive','red','green',
              'aqua','darkorange','cornflowerblue'])

for idx,color in zip(range(10),colors):
    plt.plot(val_fpr[idx],val_tpr[idx],color=color,lw=2,
             label=f'{y_label_encoder.classes_[classes_unique[idx]]} Genre, AUC:{val_roc_auc[idx]}')

plt.plot([0,1],[0,1],'k--',lw=2,label="Random Classifier")
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Validation Set")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,8))
colors=cycle(['purple','brown','pink','gray','olive','red','green',
              'aqua','darkorange','cornflowerblue'])

for idx,color in zip(range(10),colors):
    plt.plot(test_fpr[idx],test_tpr[idx],color=color,lw=2,
             label=f'{y_label_encoder.classes_[classes_unique[idx]]} Genre, AUC:{test_roc_auc[idx]}')

plt.plot([0,1],[0,1],'k--',lw=2,label="Random Classifier")
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Test Set")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

'''
Notable output: (not including the full print statements)
best CV accuracy: 0.4929722222222222
Validation Set Accuracy: 0.4886666666666667
ROC AUC score for Validation Set: 0.8819791220850481
Test Set Accuracy: 0.4972
ROC AUC score for Test Set: 0.8822664000000001
'''

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
