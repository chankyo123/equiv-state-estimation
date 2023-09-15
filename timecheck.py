import torch
import time
import timeit

# import psutil

# Define matrix sizes
size2 = (204800, 3)
size1 = (16 * 1024, 204800)

# Create random matrices
A_torch = torch.rand(size1)
B_torch = torch.rand(size2)

# Start measuring time
start_time = time.time()

# Perform matrix multiplication
# result = torch.matmul(A_torch, B_torch)


# End measuring time
end_time = time.time()


# Get memory usage
# process = psutil.Process()
# memory_usage = process.memory_info().rss / (1024 ** 2)  # in MB


pytorch_product = 'torch.matmul(A_torch, B_torch)'
print('\npytorch time: {:.2f} seconds'.format(timeit.timeit(pytorch_product, number=100, globals=globals())))