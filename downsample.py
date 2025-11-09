import numpy as np

def downsample_data(data_array, target_length=1024):
    """
    Downsamples a data array longer than 1024 elements by binning and averaging.
    
    Args:
        data_array (np.ndarray): The input 1D array (shape > 1024,).
        target_length (int): The required length (1024).
        
    Returns:
        np.ndarray: The downsampled array of shape (1024,).
    """
    input_length = len(data_array)
    
    if input_length < target_length:
        raise ValueError(f"Input length ({input_length}) is shorter than the target length ({target_length}). Use the upsampling method instead.")

    # Calculate the ratio and the number of elements per bin.
    ratio = input_length / target_length
    
    # We use integer bin size for simplicity, but if the ratio is not an integer
    # (e.g., 1500 -> 1024), we use numpy's reshape to create equal-sized chunks
    # or handle the remainder separately.
    
    # A cleaner, generalized method for downsampling:
    # 1. Calculate how many original samples contribute to each final sample.
    samples_per_output = int(np.floor(input_length / target_length))
    
    if samples_per_output == 0:
        # This should have been caught by the ValueError, but acts as a safeguard.
        return data_array[:target_length] 
    
    # Trim the input array so it's perfectly divisible by the ratio
    trimmed_length = samples_per_output * target_length
    trimmed_data = data_array[:trimmed_length]
    
    # Reshape the data into (1024, samples_per_output) and average across the rows.
    downsampled_data = trimmed_data.reshape(target_length, samples_per_output).mean(axis=1)
    
    return downsampled_data
