import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of vectors and the length of each vector
numVectors = 10000
vectorLength = 100

# Create a matrix with random values and normalize it along rows
bigMat = torch.randn(numVectors, vectorLength, device=device)
bigMat /= bigMat.norm(p=2, dim=1, keepdim=True)
# Enable gradient tracking
bigMat.requires_grad_(True)

# Compute the dot product matrix
dotProduct = bigMat @ bigMat.T
# Calculate norms for normalization
norms = torch.sqrt(torch.diag(dotProduct))
# Normalize the dot product to compute cosine similarities
normedDotProduct = dotProduct / torch.outer(norms, norms)
# Convert cosine similarities to angles in degrees
anglesDeg = torch.rad2deg(torch.acos(normedDotProduct.detach()))

# Create a mask to exclude self-angles (diagonal entries)
selfOrthogonalityMask = ~(torch.eye(numVectors, numVectors, device=device).bool())

# Plot the histogram of angles excluding the self-angles
plt.hist(anglesDeg[selfOrthogonalityMask].cpu().numpy().ravel(), bins=100, range=(0, 180))
plt.grid(True)
plt.title("Initial Angle Distribution of Embedding Vectors")
plt.xlabel("Angle (Degrees)")
plt.ylabel("Frequency")
plt.show()

# Optimizer setup
optimizer = torch.optim.Adam([bigMat], lr=0.01)
maxSteps = 500
losses = []
# Cutoff for dot product differences
dotDiffCutoff = 0.01
# Identity matrix for ideal dot product structure
bigIdentity = torch.eye(numVectors, numVectors, device=device)

# Optimization loop
for step in tqdm(range(maxSteps), desc="Optimizing Embedding Vectors"):
    optimizer.zero_grad()

    # Recompute the dot product matrix
    dotProduct = bigMat @ bigMat.T
    # Calculate the difference from the ideal identity matrix
    diff = dotProduct - bigIdentity
    
    # Loss function includes penalties for off-diagonal differences and diagonal deviations
    loss = (diff.abs() - dotDiffCutoff).relu().sum()
    loss += numVectors * diff.diag().pow(2).sum()
    
    # Backpropagation and optimization step
    loss.backward()
    optimizer.step()

    # Track the loss for visualization
    losses.append(loss.item())

# Plot the loss curve
plt.plot(losses)
plt.grid(True)
plt.title("Loss Curve During Optimization")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.show()

# Final analysis
# Recompute the dot product and normalize it
finalDotProduct = bigMat @ bigMat.T
finalNorms = torch.sqrt(torch.diag(finalDotProduct))
finalNormedDotProduct = finalDotProduct / torch.outer(finalNorms, finalNorms)
# Convert to angles
finalAnglesDeg = torch.rad2deg(torch.acos(finalNormedDotProduct.detach()))

# Plot the histogram of final angles excluding self-angles
plt.hist(finalAnglesDeg[selfOrthogonalityMask].cpu().numpy().ravel(), bins=100, range=(87, 92))
plt.grid(True)
plt.title("Final Angle Distribution of Embedding Vectors")
plt.xlabel("Angle (Degrees)")
plt.ylabel("Frequency")
plt.show()
