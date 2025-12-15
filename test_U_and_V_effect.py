import torch
import torch.nn.functional as F

import pandas as pd
import time

import numpy as np



sum_abs_mult_4_pca_l1_with_l2_pca_init_list = []
time_4_pca_l1_with_l2_pca_init_list = []




def iter_w_with_new_l2_pc(X,normalized_w_col):
      
        
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


  
def pca_l1_with_l2_pca_init(data,pca_dim,init_pc):
    start_time = time.time()
    svd_total_time = 0


    # 2. 转换为PyTorch Tensor，并移动到GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 关键步骤：转换为Tensor并指定设备
    print(f"X的设备: {data.device}")
    X = data.to(device)
    print(f"X的设备: {X.device}")  # 应显示 'cuda:0'
    data =  X

    # U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    # # normalized_w_col_0 = U[:,0:1] 
    
    



    

    principal_list = []

    for i in range(pca_dim):       
        normalized_w_col = iter_w_with_new_l2_pc(X,init_pc[:,i:i+1])
        principal_list.append(normalized_w_col)
        tmp1_row = normalized_w_col.T @ X
        X = X - normalized_w_col @ tmp1_row
                
   
    principal_matrix = torch.cat(principal_list, dim=1)  # 沿第二维度（列方向）拼接

    print("principals 矩阵形状:", principal_matrix.shape)
    print("principals 矩阵:", principal_matrix)

    mult =  principal_matrix.T @ data
    print(f"mult: {mult}")
    abs_mult = torch.abs(mult)
    print(f"abs_mult: {abs_mult}")
    sum_abs_mult = torch.sum(abs_mult)
    print(f"sum_abs_mult: {sum_abs_mult}")
    

   
    sum_abs_mult_4_pca_l1_with_l2_pca_init_list.append(sum_abs_mult)

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"algorithm 2执行时间: {elapsed_time:.6f} 秒")

    time_4_pca_l1_with_l2_pca_init_list.append(elapsed_time)
    return principal_matrix  # 返回矩阵而不是列表    




df = pd.read_csv("/root/USArrests_subset.csv", index_col=0)
print("data dimension:", df.shape) 
print("head of data:") 
print(df.head())

#matrix_numpy 特征×样本 与论文Principal Component Analysis Based on L1-Norm Maximization一致
data =  torch.from_numpy(df.values.T).float() # 此时是NumPy数组，在CPU上
# data = torch.rand(4096, 4096)
    # pca_dim = 16

pca_dim = 2



# 2. 转换为PyTorch Tensor，并移动到GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 关键步骤：转换为Tensor并指定设备
print(f"X的设备: {data.device}")
X = data.to(device)
print(f"X的设备: {X.device}")  # 应显示 'cuda:0'
data =  X

data_t = data.T

U, S, Vh = torch.linalg.svd(data_t, full_matrices=False) #only differecce is data_t
pca_l1_with_l2_pca_init(data_t,pca_dim,U[:,0:pca_dim])
print("data_t U")
print(f"sum_abs_mult_4_pca_l1_with_l2_pca_init_list {sum_abs_mult_4_pca_l1_with_l2_pca_init_list}")


U, S, Vh = torch.linalg.svd(data, full_matrices=False) #orignal 
pca_l1_with_l2_pca_init(data_t,pca_dim,Vh.T[:,0:pca_dim]) #only difference is V
print("data V=Vh.T")
print(f"sum_abs_mult_4_pca_l1_with_l2_pca_init_list {sum_abs_mult_4_pca_l1_with_l2_pca_init_list}")







