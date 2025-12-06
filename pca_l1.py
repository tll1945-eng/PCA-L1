 # Create a random matrix of dimension DxN.
# 	X = np.random.randn(D, N)
# U, S, V = torch.svd_lowrank(X, q=512, niter=16)

import torch
import torch.nn.functional as F





def interate_w(X,normalized_w):
    euclidean_dist = 1.0
    threshold = 1e-6
    while euclidean_dist > threshold:        
        normalized_old_w = normalized_w.clone()
        outer_product = X.T @ normalized_w
        signs = torch.sign(outer_product)
        w = X @ signs.float()
        normalized_w = F.normalize(w, dim=0, p=2)
        euclidean_dist = torch.norm(normalized_w - normalized_old_w, p=2)
        
        
    return normalized_w

X = torch.tensor([[-6, -5, -4, -3, -2, 10, 0, 1, 2, 3, 4],[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]]).float()

w = torch.tensor( [[1],
                  [1]]).float()      # 2行1列的列向量
normalized_w = F.normalize(w, dim=0, p=2) 
               
normalized_w = interate_w(X,normalized_w)
print("normalized w:", normalized_w)
		
		
tmp1 = torch.matmul(normalized_w.T, X) 
X = X - torch.matmul(normalized_w, tmp1) 
	

	
