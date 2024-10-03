import torch

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use the GPU
    print("CUDA is available! Using GPU.")
else:
    device = torch.device("cpu")  # Use the CPU
    print("CUDA is not available. Using CPU.")

# Create a tensor and move it to the GPU
x = torch.randn(3, 3)  # Create a random tensor on CPU
x = x.to(device)  # Move tensor to GPU
print("Tensor on device:", x)

# Perform some operations on the GPU
y = torch.randn(3, 3, device=device)  # Another tensor on GPU
result = x + y  # Element-wise addition
print("Result of addition on device:", result)

# Move the result back to CPU
result_cpu = result.to("cpu")  # Move result back to CPU
print("Result moved back to CPU:", result_cpu)
