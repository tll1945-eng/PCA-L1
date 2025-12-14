# Create a random matrix of dimension DxN.
# 	X = np.random.randn(D, N)
# U, S, V = torch.svd_lowrank(X, q=512, niter=16)


# R语言程序
# > mypcal1 <- pcal1(USArrests[,c("Murder", "Assault", "Rape")],  projDim=2, center=FALSE,  initialize="l2pca")
# > mypcal1
# $loadings
#             [,1]        [,2]
# [1,] -0.04521307  0.05326758
# [2,] -0.99134367 -0.12560502
# [3,] -0.12326194  0.99064925



import torch
import torch.nn.functional as F

import pandas as pd
import time

import numpy as np


  


sum_abs_mult_4_l2_pca_list = []
sum_abs_mult_4_pca_l1_with_l2_pca_init_list = []
sum_abs_mult_4_lpca_l1_with_gmo_init_list = []
sum_abs_mult_4_pca_l1_each_svd_iterate_w_list = []
sum_abs_mult_4_pca_l1_only_once_svd_simple_list = []


time_4_l2_pca_list = []
time_4_pca_l1_with_l2_pca_init_list = []
time_4_lpca_l1_with_gmo_init_list = []
time_4_pca_l1_each_svd_iterate_w_list = []
time_4_pca_l1_only_once_svd_simple_list = []



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


def l2_pca(data,pca_dim):
    start_time = time.time()
    svd_total_time = 0


    # 2. 转换为PyTorch Tensor，并移动到GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 关键步骤：转换为Tensor并指定设备
    print(f"X的设备: {data.device}")
    X = data.to(device)
    print(f"X的设备: {X.device}")  # 应显示 'cuda:0'
    data = X

    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    
    principal_list = []

    principal_matrix = U[:,0:pca_dim]

    print("principals 矩阵形状:", principal_matrix.shape)
    print("principals 矩阵:", principal_matrix)

    # 方法1：最常用 - 计算每列平方和
    col_squared_sums = torch.sum(principal_matrix ** 2, dim=0)
    # 或者等价地: (matrix ** 2).sum(dim=0)
    print("每列平方和:", col_squared_sums)


    mult =  principal_matrix.T @ data
    print(f"mult: {mult}")
    abs_mult = torch.abs(mult)
    print(f"abs_mult: {abs_mult}")
    sum_abs_mult = torch.sum(abs_mult)
    print(f"sum_abs_mult: {sum_abs_mult}")
    

   
    sum_abs_mult_4_l2_pca_list.append(sum_abs_mult)

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"algorithm 2执行时间: {elapsed_time:.6f} 秒")

    time_4_l2_pca_list.append(elapsed_time)




def pca_l1_with_l2_pca_init(data,pca_dim):
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

    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    # normalized_w_col_0 = U[:,0:1] 
    
    



    

    principal_list = []

    for i in range(pca_dim):       
        normalized_w_col = iter_w_with_new_l2_pc(X,U[:,i:i+1])
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


def lpca_l1_with_gmo_init(data,pca_dim):
    start_time = time.time()
    svd_total_time = 0


    # 2. 转换为PyTorch Tensor，并移动到GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 关键步骤：转换为Tensor并指定设备
    print(f"X的设备: {data.device}")

   
    X = data.to(device)
    print(f"X的设备: {X.device}")  # 应显示 'cuda:0'
    data = X

    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    normalized_w_col_0 = U[:,0:1] 
    

    principal_list = []

    for i in range(pca_dim):       
        normalized_w_col = iter_w_with_new_l2_pc(X,normalized_w_col_0)
        principal_list.append(normalized_w_col)
        tmp1_row = normalized_w_col.T @ X
        X = X - normalized_w_col @ tmp1_row
        coffs = U[:,i+1:i+2].T @ torch.cat(principal_list, dim=1)
        tmp_matrix = torch.cat(principal_list, dim=1) * coffs
        sum_tmp_matrix_cols = tmp_matrix.sum(dim=1)  # A.sum(dim=1) → shape (4,)，每一行是A所有列的和
        new_l2_pc_col_0 = U[:,i+1:i+2] - sum_tmp_matrix_cols
        
   
    principal_matrix = torch.cat(principal_list, dim=1)  # 沿第二维度（列方向）拼接

    print("principals 矩阵形状:", principal_matrix.shape)
    print("principals 矩阵:", principal_matrix)

    mult =  principal_matrix.T @ data
    print(f"mult: {mult}")
    abs_mult = torch.abs(mult)
    print(f"abs_mult: {abs_mult}")
    sum_abs_mult = torch.sum(abs_mult)
    print(f"sum_abs_mult: {sum_abs_mult}")
    

   
    sum_abs_mult_4_lpca_l1_with_gmo_init_list.append(sum_abs_mult)

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"algorithm 2执行时间: {elapsed_time:.6f} 秒")
    

    time_4_lpca_l1_with_gmo_init_list.append(elapsed_time)
    return principal_matrix  # 返回矩阵而不是列表
   

	
def pca_l1_each_svd_iterate_w(data,pca_dim):
    start_time = time.time()
    svd_total_time = 0


    # 2. 转换为PyTorch Tensor，并移动到GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 关键步骤：转换为Tensor并指定设备
    print(f"X的设备: {data.device}")
    X = data.to(device)
    print(f"X的设备: {X.device}")  # 应显示 'cuda:0'


    X = data.to(device)
    data = X

    print(f"X的设备: {X.device}")  # 应显示 'cuda:0'

    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    

    principal_list = []

    for i in range(pca_dim):
        normalized_w_col = interate_w(X)
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

    
    sum_abs_mult_4_pca_l1_each_svd_iterate_w_list.append(sum_abs_mult)

    end_time = time.time()
    elapsed_time = end_time - start_time
   
    print(f"algorithm 2执行时间: {elapsed_time:.6f} 秒")
   
    time_4_pca_l1_each_svd_iterate_w_list.append(elapsed_time)
    return principal_matrix  # 返回矩阵而不是列表
    
#########################################################################


def pca_l1_only_once_svd_simple(data,pca_dim):
    start_time = time.time()
    svd_total_time = 0


    # 2. 转换为PyTorch Tensor，并移动到GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 关键步骤：转换为Tensor并指定设备
    print(f"X的设备: {data.device}")
    X = data.to(device)
    print(f"X的设备: {X.device}")  # 应显示 'cuda:0'
    

    
    data = X


    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    new_l2_pc_col = U[:,0:1] 
    normalized_w_col = new_l2_pc_col 

    



    

    principal_list = []

    for i in range(pca_dim):
        normalized_w_col = normalized_w_col.clone()
        normalized_w_col = iter_w_with_new_l2_pc(X,normalized_w_col)
        principal_list.append(normalized_w_col)
        tmp1_row = normalized_w_col.T @ X
        X = X - normalized_w_col @ tmp1_row
        new_l2_pc_col = new_l2_pc_col - normalized_w_col
   
    principal_matrix = torch.cat(principal_list, dim=1)  # 沿第二维度（列方向）拼接

    print("principals 矩阵形状:", principal_matrix.shape)
    print("principals 矩阵:", principal_matrix)

    mult =  principal_matrix.T @ data
    print(f"mult: {mult}")
    abs_mult = torch.abs(mult)
    print(f"abs_mult: {abs_mult}")
    sum_abs_mult = torch.sum(abs_mult)
    print(f"sum_abs_mult: {sum_abs_mult}")
    

   
    sum_abs_mult_4_pca_l1_only_once_svd_simple_list.append(sum_abs_mult)

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"algorithm 2执行时间: {elapsed_time:.6f} 秒")
    
    time_4_pca_l1_only_once_svd_simple_list.append(elapsed_time)

    return principal_matrix  # 返回矩阵而不是列表
   
    

#########################################################################







    



if __name__ == "__main__":

    A = torch.tensor([[1, 2, 3], 
                    [4, 5, 6], 
                    [7, 8, 9]])  # shape: (3, 3)
    coeffs = torch.tensor([10, 20, 30])  # shape: (3,) → 对应3列的系数

    # 2. 广播相乘（推荐写法：隐式扩展维度）
    result = A * coeffs  # 核心！利用广播自动匹配维度

    # 等价写法（显式扩展维度，更直观，新手推荐）
    result = A * coeffs.unsqueeze(0)  # coeffs.unsqueeze(0) → shape (1, 3)，广播到(3,3)





    print("原始矩阵：\n", A)
    print("系数向量：", coeffs)
    # print("coeffs.unsqueeze(0):"coeffs.unsqueeze(0))
    print("结果矩阵：\n", result)
    
   
    


    
    df = pd.read_csv("/root/USArrests_subset.csv", index_col=0)
    print("data dimension:", df.shape) 
    print("head of data:") 
    print(df.head())

    #matrix_numpy 特征×样本 与论文Principal Component Analysis Based on L1-Norm Maximization一致
    data =  torch.from_numpy(df.values.T).float() # 此时是NumPy数组，在CPU上
    # data = torch.rand(4096, 4096)
    # pca_dim = 16

    pca_dim = 2

    l2_pca(data,pca_dim)
    print(f"sum_abs_mult_4_l2_pca_list {sum_abs_mult_4_l2_pca_list}")

    ##数值高 时间短
    pca_l1_with_l2_pca_init(data,pca_dim)
    print(f"sum_abs_mult_4_pca_l1_with_l2_pca_init_list {sum_abs_mult_4_pca_l1_with_l2_pca_init_list}")

    

    lpca_l1_with_gmo_init(data,pca_dim)
    print(f"sum_abs_mult_4_lpca_l1_with_gmo_init_list {sum_abs_mult_4_lpca_l1_with_gmo_init_list}")
  
    
    pca_l1_each_svd_iterate_w(data,pca_dim)
    print(f"sum_abs_mult_4_pca_l1_each_svd_iterate_w_list {sum_abs_mult_4_pca_l1_each_svd_iterate_w_list}") 

    pca_l1_only_once_svd_simple(data,pca_dim)
    print(f"sum_abs_mult_4_pca_l1_only_once_svd_simple_list {sum_abs_mult_4_pca_l1_only_once_svd_simple_list}")##效果不好  

    loop_num = 5
    for i in range(loop_num):
        data = torch.rand(4096, 4096)
        pca_dim = 16
        l2_pca(data,pca_dim)
        pca_l1_with_l2_pca_init(data,pca_dim) # 就简单
        lpca_l1_with_gmo_init(data,pca_dim)
        pca_l1_each_svd_iterate_w(data,pca_dim)
        pca_l1_only_once_svd_simple(data,pca_dim)  
    
    print(f"total_sum_abs_mult_4_l2_pca_list {sum(sum_abs_mult_4_l2_pca_list)}")
    print(f"total_time_4_l2_pca_list {sum(time_4_l2_pca_list)}")
   
    print(f"total_sum_abs_mult_4_pca_l1_with_l2_pca_init_list {sum(sum_abs_mult_4_pca_l1_with_l2_pca_init_list)}")
    print(f"total_time_4_pca_l1_with_l2_pca_init_list {sum(time_4_pca_l1_with_l2_pca_init_list)}")


   
    print(f"total_sum_abs_mult_4_lpca_l1_with_gmo_init_list {sum(sum_abs_mult_4_lpca_l1_with_gmo_init_list)}")
    print(f"total_time_4_lpca_l1_with_gmo_init_list {sum(time_4_lpca_l1_with_gmo_init_list)}")

    
    print(f"total_sum_abs_mult_4_pca_l1_each_svd_iterate_w_list {sum(sum_abs_mult_4_pca_l1_each_svd_iterate_w_list)}") 
    print(f"total_time_4_pca_l1_each_svd_iterate_w_list {sum(time_4_pca_l1_each_svd_iterate_w_list)}") 
    
    
    print(f"total_sum_abs_mult_4_pca_l1_only_once_svd_simple_list {sum(sum_abs_mult_4_pca_l1_only_once_svd_simple_list)}")
    print(f"total_time_4_pca_l1_only_once_svd_simple_list {sum(time_4_pca_l1_only_once_svd_simple_list)}")
    


    print("ok")

        