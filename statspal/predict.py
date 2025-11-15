#Crucial Note: Please ensure your converted model, statspal_v2.onnx, is saved in the same directory as this predict.py file.

import numpy as np
import onnxruntime as rt
import os
import importlib.resources

def _downsample_data(data_array: np.ndarray, FIXED_SEQUENCE_LENGTH = 1024) -> np.ndarray:
    """
    Downsamples the input data to exactly 1024 points using stratified 
    random sampling based on an 'auto-binned' histogram.
    This preserves the Probability Density Function (PDF) of the original data.
    """
    N_original = len(data_array)
    
    if N_original == FIXED_SEQUENCE_LENGTH:
        return data_array.astype(np.float32)
    
    # NOTE: Since the model was trained on 1024 points, this method 
    # should only be used for N_original >= 1024.
    if N_original < FIXED_SEQUENCE_LENGTH:
        # Fallback for upsampling: If the input is too short, we fall back 
        # to the safest method: linear interpolation, or return an error.
        # For now, we'll return an error since the histogram approach is designed 
        # for downsampling.
        raise ValueError(
            f"Input size {N_original} is too small. Stratified sampling requires N >= 1024."
        )


    # 1. Calculate the histogram using 'auto' bins to determine bin edges and counts
    # The 'bins' array contains the edges, and 'hist' contains the counts (bin_height)
    hist, bins = np.histogram(data_array, bins='auto')
    
    # Calculate the desired scaling ratio
    scale_factor = FIXED_SEQUENCE_LENGTH / N_original
    
    # 2. Calculate the target number of samples for each bin (k[i])
    # The floor operation creates the remainder that needs 'fudging'
    k_target = np.floor(hist * scale_factor).astype(int)
    
    # 3. Handle the remainder (fudging) to ensure the total is exactly 1024
    current_total = np.sum(k_target)
    remainder = FIXED_SEQUENCE_LENGTH - current_total
    
    if remainder > 0: #IN THEORY THIS WOULD FALL SHORT BY NOT TOPPING UP THE MODE BINS PROPORTIONALLY MORE THAN OTHER BINS
        # Distribute the remainder (D) into the bins that had the largest truncation.
        # This is calculated by the fractional part of (hist * scale_factor).
        fractions = (hist * scale_factor) - k_target
        
        # Get the indices of the bins with the largest fractional remainder
        # We need the 'remainder' number of top indices
        top_indices = np.argsort(fractions)[::-1][:remainder]
        
        # Add 1 to the k_target for those bins
        k_target[top_indices] += 1

    # 4. Perform stratified random sampling
    downsampled_points = []
    
    for i in range(len(hist)):
        # Define the lower and upper bounds of the current bin
        bin_min = bins[i]
        bin_max = bins[i+1]
    
        # 1. Use exclusive upper bound (<) for all bins except the last one
        if i < len(hist) - 1:
            # Bin bounds: [min, max)
            indices_in_bin = np.where((data_array >= bin_min) & (data_array < bin_max))[0]
        else:
            # 2. For the last bin, include the maximum edge [min, max]
            indices_in_bin = np.where((data_array >= bin_min) & (data_array <= bin_max))[0]
        
        # Select k_target[i] points at random from the original data in this bin
        num_to_sample = k_target[i]
        
        # Use np.random.choice to get the random indices
        # If the number of points in the bin is less than num_to_sample (shouldn't happen 
        # if the scaling is correct, but safe to use replace=True)
        sampled_indices = np.random.choice(
            indices_in_bin, 
            size=num_to_sample, 
            replace=True if num_to_sample > len(indices_in_bin) else False
        )
        
        # Extract the actual data points and append
        downsampled_points.append(data_array[sampled_indices])

    # 5. Concatenate all samples and shuffle to destroy bin order (if necessary for the model)
    final_array = np.concatenate(downsampled_points)
    
    # IMPORTANT: Shuffle the array to ensure the model sees a random sequence 
    # of the downsampled points, NOT the strict bin order.
    np.random.shuffle(final_array)
    
    # Final check: Must be exactly 1024
    if len(final_array) != FIXED_SEQUENCE_LENGTH:
         raise RuntimeError(f"Downsampled size mismatch: Expected 1024, got {len(final_array)}")

    return final_array.astype(np.float32)

def _upsample_data(data_array: np.ndarray, FIXED_SEQUENCE_LENGTH = 1024, UP_SAMPLE_BINS = 100) -> np.ndarray:
# 1. Calculate the PDF using a fixed number of bins for stability with sparse data
    hist, bins = np.histogram(data_array, bins=UP_SAMPLE_BINS)
    
    # 2. Normalize to get the probability distribution over the bins
    bin_probabilities = hist / np.sum(hist)
    
    # 3. Use np.random.choice to select 1024 bins, weighted by probability
    # The result is 1024 bin indices, indicating where the synthetic points should land
    bin_indices = np.arange(UP_SAMPLE_BINS)
    synthetic_bin_assignment = np.random.choice(
        bin_indices, 
        size=FIXED_SEQUENCE_LENGTH, 
        p=bin_probabilities
    )
    
    # 4. Generate a synthetic data point for each assigned bin
    synthetic_points = np.zeros(FIXED_SEQUENCE_LENGTH, dtype=np.float32)
    
    for i, bin_idx in enumerate(synthetic_bin_assignment):
        # Find the boundaries for the selected bin
        bin_min = bins[bin_idx]
        bin_max = bins[bin_idx + 1]
        
        # Generate a random float uniformly within the bin boundaries
        # This retains the original distribution shape while creating 1024 points
        synthetic_points[i] = np.random.uniform(bin_min, bin_max)
        
    final_array = synthetic_points
    # No need to shuffle, as they were generated randomly already
    
    # Final check and return
    if len(final_array) != FIXED_SEQUENCE_LENGTH:
        # This should only happen if there's a serious mathematical error, but good to check
        raise RuntimeError(f"Resampled size mismatch: Expected 1024, got {len(final_array)}")
    
    return final_array.astype(np.float32)

def _resize_to_1024(arr):
    if len(arr) == 1024:
        return arr
    elif len(arr) > 1024:
        return _downsample_data(arr)
    elif len(arr) < 1024:
        return _upsample_data(arr)

try:
    # Use files() from importlib.resources to get a Path object to the model
    MODEL_PATH = str(importlib.resources.files('statspal') / 'statspal_v2.onnx')
except Exception:
    # Fallback to local path (only works reliably during development/testing)
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'statspal_v2.onnx')

def predict(arr):
    arr = _resize_to_1024(arr)
    
    model_path = "statspal_v2.onnx"
    sess = rt.InferenceSession(MODEL_PATH, providers=rt.get_available_providers())
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    input_data = arr.reshape(1, 1024, 1)
    model_output = sess.run([output_name], {input_name: input_data})
    prediction_nodes = model_output[0][0] #output of 21 softmax nodes
    return prediction_nodes

def predict_max(arr, verbose=False):
    prediction = predict(arr)
    distribution_num = np.argmax(prediction)
    if verbose == True:
        print("Predicted Distribution: " + str(distribution_num) + " "
              + keys()[distribution_num][1]
              + "\n" + "Probability: " + str(max(prediction)))
    return np.argmax(prediction), max(prediction)

def keys():
    discrete_keys = [[0, "bernoulli"],
                    [1, "betabinom"],
                    [2, "betanbinom"],
                    [3, "binom"],
                    [4, "boltzmann"],
                    [5, "dlaplace"],
                    [6, "geom"],
                    [7, "hypergeom"],
                    [8, "logser"],
                    [9, "nbinom"],
                    [10, "nchypergeom_fisher"],
                    [11, "nchypergeom_wallenius"],
                    [12, "nhypergeom"],
                    [13, "planck"],
                    [14, "poisson"],
                    [15, "poisson_binom"],
                    [16, "randint"],
                    [17, "skellam"],
                    [18, "yulesimon"],
                    [19, "zipf"],
                    [20, "zipfian"]]
    return discrete_keys
