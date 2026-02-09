import numpy as np
import utils

# Set random seed for reproducibility
np.random.seed(42)

# Machine Learning pipeline

# Stage 1 & 2: Data Preparation and ingestion

# Distances in miles for recent bike deliveries
distances = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)

# Corresponding delivery times in minutes
times = np.array([[6.96], [12.11], [16.77], [22.21]], dtype=np.float32)

# Stage 3: Model Building

# Create a simple linear model with one input (distance) and one output (time)
# Initialize weights and bias
weights = np.random.randn(1, 1).astype(np.float32) * 0.01
bias = np.zeros((1, 1), dtype=np.float32)

# Stage 4: Training

# Hyperparameters
learning_rate = 0.01
epochs = 500

# Training loop
for epoch in range(epochs):
    # Forward pass
    outputs = np.dot(distances, weights) + bias

    # Calculate loss (Mean Squared Error)
    loss = np.mean((outputs - times) ** 2)

    # Backward pass
    # Calculate gradients
    batch_size = distances.shape[0]
    grad_output = 2 * (outputs - times) / batch_size

    # Gradient w.r.t weights: X^T @ grad_output
    grad_weights = np.dot(distances.T, grad_output)

    # Gradient w.r.t bias: sum of grad_output
    grad_bias = np.sum(grad_output, axis=0, keepdims=True)

    # Update weights and bias using SGD
    weights -= learning_rate * grad_weights
    bias -= learning_rate * grad_bias

    # Print the loss every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}: Loss = {loss:.6f}")

# Stage 5: Prediction

distance_to_predict = 7.0

# Convert the python variable into a 2D NumPy array that the model expects
new_distance = np.array([[distance_to_predict]], dtype=np.float32)

# Pass the new data to the trained model to get a prediction
predicted_time = np.dot(new_distance, weights) + bias

# Extract the scalar value for printing
predicted_time_scalar = predicted_time.item()
print(f"Prediction for a {distance_to_predict}-mile delivery: {predicted_time_scalar:.1f} minutes")

if predicted_time_scalar > 30:
    print("\nDecision: Do NOT take the job. You will likely be late.")
else:
    print("\nDecision: Take the job. You can make it!")

# Stage 6: Inspection

# Get the weights and biases
print(f"Weight: {weights}")
print(f"Bias: {bias}")

# Stage 7: Testing

# Combined dataset: bikes for short distances, cars for longer ones
new_distances = np.array([
    [1.0], [1.5], [2.0], [2.5], [3.0], [3.5], [4.0], [4.5], [5.0], [5.5],
    [6.0], [6.5], [7.0], [7.5], [8.0], [8.5], [9.0], [9.5], [10.0], [10.5],
    [11.0], [11.5], [12.0], [12.5], [13.0], [13.5], [14.0], [14.5], [15.0], [15.5],
    [16.0], [16.5], [17.0], [17.5], [18.0], [18.5], [19.0], [19.5], [20.0]
], dtype=np.float32)

# Corresponding delivery times in minutes
new_times = np.array([
    [6.96], [9.67], [12.11], [14.56], [16.77], [21.7], [26.52], [32.47], [37.15], [42.35],
    [46.1], [52.98], [57.76], [61.29], [66.15], [67.63], [69.45], [71.57], [72.8], [73.88],
    [76.34], [76.38], [78.34], [80.07], [81.86], [84.45], [83.98], [86.55], [88.33], [86.83],
    [89.24], [88.11], [88.16], [91.77], [92.27], [92.13], [90.73], [90.39], [92.98]
], dtype=np.float32)

# Make predictions on new data
predictions = np.dot(new_distances, weights) + bias

# Calculate the new loss
new_loss = np.mean((predictions - new_times) ** 2)
print(f"Loss on new, combined data: {new_loss:.2f}")

utils.plot_nonlinear_comparisons(predictions, new_distances, new_times)