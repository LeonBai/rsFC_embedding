# rsFC_embedding

## Included code snippets:

### 1. FID.py

Calculate the core FID distance between classes on the embedding space


### 2.relation-embedding.py

The core code to implement our six embedding options on any types of datasets with supervisions. 

Input: X [samples, features]

Output: Z[samples, reduced_dimensions == 2/3]

Choices: Gaussian, vMF embeddings
         with, without pairwise difference  
         Least (only know number of the classes in diagnostic labels), Median (the full diagnostic labels), Most (the full diagnostic labels + contrastive learning) types of diagnostic information 
Logic of code structure: auto-encoder structure 

### Required python3 version> 3.5 <3.8

### Required library 
0. Numpy: any versions are runnable here
1. Tensorflow == v2.0
