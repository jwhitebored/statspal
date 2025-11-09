import numpy as np
from scipy.interpolate import interp1d

# Assuming model is already loaded and upsample/downsample functions are available

def statspal_prediction(model, raw_data_array, target_length=1024):
    """
    Handles variable-length input data, resizes it to 1024, and makes a prediction.

    Args:
        model: The trained Keras model.
        raw_data_array (np.ndarray): The user's input array of any length.
        target_length (int): The fixed input length of the model (1024).

    Returns:
        np.ndarray: The prediction output (e.g., class probabilities).
    """
    input_length = len(raw_data_array)
    data = raw_data_array.astype(np.float32)

    # 1. Resize the data
    if input_length > target_length:
        # Downsample using averaging
        print(f"Input is {input_length} elements. Downsampling by averaging...")
        data = downsample_data(data, target_length)
    elif input_length < target_length:
        # Upsample using linear interpolation
        print(f"Input is {input_length} elements. Upsampling via interpolation...")
        data = upsample_data(data, target_length)
    else:
        print("Input length is exactly 1024. No resizing needed.")

    # 2. Reshape to required Keras format (1, 1024, 1)
    # Add Feature dimension (1024, 1)
    reshaped_data = data.reshape(target_length, 1)
    
    # Add Batch dimension (1, 1024, 1)
    data_for_prediction = np.expand_dims(reshaped_data, axis=0)

    # 3. Predict
    prediction = model.predict(data_for_prediction)
    
    return prediction
