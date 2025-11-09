from scipy.interpolate import interp1d
import numpy as np

def upsample_data(data_array, target_length=1024):
    """
    Upsamples a data array shorter than 1024 elements using linear interpolation.
    
    Args:
        data_array (np.ndarray): The input 1D array (shape < 1024,).
        target_length (int): The required length (1024).
        
    Returns:
        np.ndarray: The upsampled array of shape (1024,).
    """
    input_length = len(data_array)
    
    if input_length > target_length:
        raise ValueError(f"Input length ({input_length}) is longer than the target length ({target_length}). Use the downsampling method instead.")

    # 1. Define the original X coordinates (0 to N-1)
    original_x = np.linspace(0, 1, input_length)
    
    # 2. Define the new X coordinates (0 to target_length-1)
    new_x = np.linspace(0, 1, target_length)
    
    # 3. Create the interpolation function
    # 'kind=linear' creates straight lines between points.
    interpolator = interp1d(original_x, data_array, kind='linear')
    
    # 4. Generate the new, upsampled data
    upsampled_data = interpolator(new_x).astype(np.float32)
    
    return upsampled_data
