import torch

BLOCK_M = 4
BLOCK_K = 6
stride_x_m = BLOCK_K
stride_x_k = 1

# Like matmul, each row is contiguous

# offs_x_m = torch.arange(0, BLOCK_M)
offs_x_m = torch.randint(0, 100, (BLOCK_M,))
print(offs_x_m)

offs_x_m = torch.arange(0, BLOCK_M)
print(offs_x_m)

offs_x_k = torch.arange(0, BLOCK_K)

print(offs_x_m[:, None] * stride_x_m )
print(offs_x_m[:, None] + offs_x_k[None, :] * stride_x_k)


""""""

print('Each column is sort of contiguous (strided access)')
offs_x_m = torch.arange(0, BLOCK_M)

offs_x_k = torch.randint(0, 100, (BLOCK_K,))
# offs_x_k = torch.arange(0, BLOCK_K)

print(offs_x_m[:, None])
a = offs_x_m[:, None] * stride_x_m + offs_x_k[None, :] * stride_x_k
print(a)
t = torch.as_strided(a, (BLOCK_M,), (BLOCK_K, ))
print(t)

""""""

# Multiple rows
# Cut!

print('Multiple rows')

offs_x_m = torch.arange(0, BLOCK_M) % 2

offs_x_k = torch.arange(0, BLOCK_K)

print(offs_x_m[:, None] * stride_x_m + offs_x_k[None, :] * stride_x_k)

""""""

# Multiple rows
# Cut!

print('Multiple rows / cols')

offs_x_m = torch.arange(0, BLOCK_M) % 2

offs_x_k = torch.arange(0, BLOCK_K) % 4

print(offs_x_m[:, None] * stride_x_m + offs_x_k[None, :] * stride_x_k)
