 # Create a random matrix of dimension DxN.
# 	X = np.random.randn(D, N)
# U, S, V = torch.svd_lowrank(X, q=512, niter=16)

import torch
import torch.nn.functional as F

import pandas as pd
import time

import numpy as np


# u = np.array([1,2,3])
# print("id(u) ", id(u))
# v = np.array([4,5,6])
# print("id(v) ", id(v))
# v = v + u
# print("v", v, "id(v) ", id(v))

# algorithm 1
start_time = time.time()

df = pd.read_csv("/root/USArrests_subset.csv", index_col=0)
print("data dimension:", df.shape) 
print("head of data:") 
print(df.head())

#matrix_numpy 特征×样本 与论文Principal Component Analysis Based on L1-Norm Maximization一致
matrix_numpy = df.values.T  # 此时是NumPy数组，在CPU上

# 2. 转换为PyTorch Tensor，并移动到GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 关键步骤：转换为Tensor并指定设备
matrix_tensor = torch.from_numpy(matrix_numpy).float().to(device)

# 3. 在GPU上进行SVD分解
U, S, Vh = torch.linalg.svd(matrix_tensor, full_matrices=False)


# 现在 U, S, Vh 都是GPU上的Tensor
print(f"U的设备: {U.device}")  # 应显示 'cuda:0'
print(f"奇异值: {S[:2]}")     # 查看前2个奇异值

def interate_w(X,normalized_w_col):
    euclidean_dist = 1.0
    threshold = 1e-6
    while euclidean_dist > threshold:        
        normalized_old_w_col = normalized_w_col.clone()
        
        outer_product_row = normalized_w_col.T @ X 
        signs_col = torch.sign(outer_product_row).T
        w_col =  X @ signs_col.float()
        normalized_w_col = F.normalize(w_col, dim=0, p=2)
        euclidean_dist = torch.norm(normalized_w_col - normalized_old_w_col, p=2)
        
       
    return normalized_w_col

X = matrix_tensor




normalized_w_col = U[:,0:1]               
normalized_w_col = interate_w(X,normalized_w_col)

print("normalized w_col:", normalized_w_col)


pca_dim = 2
j = 1
while j < pca_dim:    
               
    
    tmp1_row = normalized_w_col.T @ X
    X = X - normalized_w_col @ tmp1_row
    # w_col =U[:,j:j+1]
    # normalized_w_col = F.normalize(w_col, dim=0, p=2) 

    normalized_w_col = U[:,j:j+1] 
    normalized_w_col = interate_w(X,normalized_w_col)
    j = j +1 

print("normalized w_col:", normalized_w_col)


end_time = time.time()
elapsed_time = end_time - start_time
print(f"algorithm 1执行时间: {elapsed_time:.6f} 秒")
	












	
# algorithm 2
start_time = time.time()

df = pd.read_csv("/root/USArrests_subset.csv", index_col=0)
print("data dimension:", df.shape) 
print("head of data:") 
print(df.head())

#matrix_numpy 特征×样本 与论文Principal Component Analysis Based on L1-Norm Maximization一致
matrix_numpy = df.values.T  # 此时是NumPy数组，在CPU上

# 2. 转换为PyTorch Tensor，并移动到GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 关键步骤：转换为Tensor并指定设备
X= torch.from_numpy(matrix_numpy).float().to(device)
# normalized_w_col = torch.zeros((X.shape[0], 1), device=X.device)
# print(f"列向量形状: {normalized_w_col.shape}")
# print(f"列向量设备: {normalized_w_col.device}")
# # 3. 在GPU上进行SVD分解
# U, S, Vh = torch.linalg.svd(matrix_tensor, full_matrices=False)


# 现在 U, S, Vh 都是GPU上的Tensor
print(f"U的设备: {U.device}")  # 应显示 'cuda:0'
print(f"奇异值: {S[:2]}")     # 查看前2个奇异值

def interate_w(X):
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    normalized_w_col = U[:,0:1]  
    
    euclidean_dist = 1.0
    threshold = 1e-6

    while euclidean_dist > threshold:        
        normalized_old_w_col = normalized_w_col.clone()
        
        outer_product_row = normalized_w_col.T @ X 
        signs_col = torch.sign(outer_product_row).T
        w_col =  X @ signs_col.float()
        normalized_w_col = F.normalize(w_col, dim=0, p=2)
        euclidean_dist = torch.norm(normalized_w_col - normalized_old_w_col, p=2)
        
       
    return normalized_w_col

# X = matrix_tensor




# normalized_w_col = U[:,0:1]               
# normalized_w_col = interate_w(X,normalized_w_col)

# print("normalized w_col:", normalized_w_col)


pca_dim = 2
# principals = torch.zeros((X.shape[0], 1), device=X.device)
principals = []
# j = 1
# while j < pca_dim:
for i in range(pca_dim):               
    
    # w_col =U[:,j:j+1]
    # normalized_w_col = F.normalize(w_col, dim=0, p=2) 

    # normalized_w_col = U[:,j:j+1] 
    # normalized_w_col = interate_w(X,normalized_w_col)
    normalized_w_col = interate_w(X) 
     
    # principals = torch.cat([principals, normalized_w_col], dim=1)  # dim=1表示按列连接
    principals.append(normalized_w_col)
    tmp1_row = normalized_w_col.T @ X
    X = X - normalized_w_col @ tmp1_row  
    # j = j +1 

print("principals:", principals)


end_time = time.time()
elapsed_time = end_time - start_time
print(f"algorithm 2执行时间: {elapsed_time:.6f} 秒")



	

	
