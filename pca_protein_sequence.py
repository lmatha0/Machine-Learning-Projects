#load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load csv dataset (https://www.rcsb.org/) (Filtered for protein sequence by X-ray diffraction for human proteins. Collected first 2500 results.)
df = pd.read_csv('protein_sequence_data.csv', skiprows=1)
print(df.head()) #print first 5 data entries to visualize data

#define list of amino acids
amino_acid = list("ACDEFGHIKLMNPQRSTVWY") 

#calculate composition of each amino acid per sequence
def aa_composition(seq):
    counts = np.array([seq.count(aa) for aa in amino_acid], dtype=float)
    return counts / len(seq)

#format data used to perform PCA
aa_comp = np.vstack(df['Sequence'].apply(aa_composition)) #apply aa_composition to each sequence in the dataset 
aa_length = df['Polymer Entity Sequence Length'].values.reshape(-1,1) #format amino acid sequence length
aa_weight = df['Molecular Weight (Entity)'].values.reshape(-1,1) #format protein molecular weight

#stack features horizontally in a matrix
X = np.hstack([aa_comp, aa_length, aa_weight]) 

#normalize data 
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_norm = (X - X_mean) / X_std

def pca(X):
    #calculate variance of data in dataset
    X_var = X - np.mean(X, axis=0)

    #calculate covariance matrix
    X_cov = np.cov(X_var, rowvar=False) #set columns as variables

    #calculate eigenvalues and eigenvectors of covariance matrix
    eigen_value, eigen_vector = np.linalg.eigh(X_cov) 
    
    #sort eigenvalues and eigenvectors in descending order
    index = np.argsort(eigen_value)[::-1] #arranges index of eigenvalues in descending order
    eigen_value = eigen_value[index]
    eigen_vector = eigen_vector[:, index] #reorders eigenvectors with corresponding sorted eigenvalues

    #project data on principal components
    X_pca = np.dot(X_var, eigen_vector)

    return X_pca, eigen_value, eigen_vector

#test on normalized X data
X_pca, eigen_values, eigen_vectors = pca(X_norm)

#plot data
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Protein Features')
plt.show()

