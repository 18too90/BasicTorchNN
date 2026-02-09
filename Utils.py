import torch
import matplotlib.pyplot as plt
import numpy

def plot_results(model, distances, times):
    """
    Args:
        model: The trained machine learning model to use for predictions.
        distances: The input data points (features) for the model.
        times: The target data points (labels) for the plot. 
    """

    model.eval()

    with torch.no_grad():
        predicted_times = model(distances)
    
    plt.figure(figsize=(8,6))

    plt.plot(distances.numpy(), times.numpy(), color = 'orange', 
             marker='o', linestyle = "None", label = "Actual Delivery Times")
    
    plt.plot(distances.numpy(), predicted_times.numpy(), color = 'green', 
             marker='None', linestyle = "None", label = "Predicted Times")
    
    plt.title('Actual vs. Predicted Delivery Times')
    plt.xlabel('Distance (miles)')
    plt.ylabel('Time (minutes)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_nonlinear_comparison(model, new_distances, new_times):
    """
    Args:
        model: The trained model to be evaluated.
        new_distances: The new input data for generating predictions.
        new_times: The actual target values for comparison.
    """
    model.eval()

    with torch.no_grad():
        predictions = model(new_distances)
    
    plt.figure(figsize=(8, 6))

    plt.plot(new_distances.numpy(), new_times.numpy(), color='orange',
             marker='o', linestyle='None', label='Actual Data (Bikes & Cars)')
    
    plt.plot(new_distances.numpy(), predictions.numpy(), color='green',
             marker='None', label='Linear Model Predictions')
    
    plt.title('Linear Model vs. Non-Linear Reality')
    plt.xlabel('Distance (miles)')
    plt.ylabel('Time (minutes)')
    plt.legend()
    plt.grid(True)
    plt.show()
