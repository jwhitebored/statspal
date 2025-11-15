import numpy as np
import onnxruntime as rt
import os
import importlib.resources as pkg_resources

# --- Configuration (These values MUST match your trained model) ---
FIXED_SEQUENCE_LENGTH = 1024
NUM_CLASSES = 21 # Set this to the actual number of distribution classes
UP_SAMPLE_BINS = 100 
MODEL_NAME = 'statspal_v2.onnx' # Filename for the bundled ONNX model

# --- ONNX Setup: Initialized globally using package resources (THE ONLY WAY) ---
ONNX_SESSION = None
INPUT_NAME = None
OUTPUT_NAME = None

try:
    # Use the package name ('statspal') and the resource name ('statspal_v2.onnx')
    with pkg_resources.open_binary('statspal', MODEL_NAME) as model_stream:
        # Load the ONNX model from the file stream's binary content
        ONNX_SESSION = rt.InferenceSession(model_stream.read()) 
    
    INPUT_NAME = ONNX_SESSION.get_inputs()[0].name
    OUTPUT_NAME = ONNX_SESSION.get_outputs()[0].name
    
except Exception as e:
    # This ensures that if loading fails, the code doesn't crash later
    print(f"CRITICAL ERROR loading ONNX model '{MODEL_NAME}': {e}")


# --- Helper Functions (Resampling Logic) ---

def _resample_data(data_array: np.ndarray) -> np.ndarray:
    """
    Resamples the unordered data to exactly 1024 points, preserving the PDF shape.
    
    If N >= 1024: Uses Stratified Random Subsampling (Downsampling).
    If N < 1024: Uses Synthetic PDF Generation (Upsampling).
    """
    N_original = len(data_array)
    
    if N_original == FIXED_SEQUENCE_LENGTH:
        return data_array.astype(np.float32)

    # ----------------------------------------------------------------------
    # CASE 1: DOWNSAMPLING (N_original > 1024) - Stratified Subsampling
    # ----------------------------------------------------------------------
    if N_original > FIXED_SEQUENCE_LENGTH:
        
        hist, bins = np.histogram(data_array, bins='auto')
        scale_factor = FIXED_SEQUENCE_LENGTH / N_original
        k_target = np.floor(hist * scale_factor).astype(int)
        
        # Fudge the remainder
        remainder = FIXED_SEQUENCE_LENGTH - np.sum(k_target)
        if remainder > 0:
            fractions = (hist * scale_factor) - k_target
            top_indices = np.argsort(fractions)[::-1][:remainder]
            k_target[top_indices] += 1

        downsampled_points = []
        for i in range(len(hist)):
            bin_min = bins[i]
            bin_max = bins[i+1]
            
            if i < len(hist) - 1:
                indices_in_bin = np.where((data_array >= bin_min) & (data_array < bin_max))[0]
            else:
                indices_in_bin = np.where((data_array >= bin_min) & (data_array <= bin_max))[0]
            
            num_to_sample = k_target[i]
            
            sampled_indices = np.random.choice(
                indices_in_bin, 
                size=num_to_sample, 
                replace=True if num_to_sample > len(indices_in_bin) else False
            )
            downsampled_points.append(data_array[sampled_indices])

        final_array = np.concatenate(downsampled_points)
        np.random.shuffle(final_array)
        
    # ----------------------------------------------------------------------
    # CASE 2: UPSAMPLING (N_original < 1024) - Synthetic PDF Generation
    # ----------------------------------------------------------------------
    else: # N_original < FIXED_SEQUENCE_LENGTH
        
        if N_original == 0:
             return np.zeros(FIXED_SEQUENCE_LENGTH, dtype=np.float32)

        # 1. Calculate the PDF 
        hist, bins = np.histogram(data_array, bins=UP_SAMPLE_BINS)
        
        # 2. Normalize to get the probability distribution over the bins
        bin_probabilities = hist / np.sum(hist)
        
        # 3. Select 1024 bins, weighted by probability
        bin_indices = np.arange(UP_SAMPLE_BINS)
        synthetic_bin_assignment = np.random.choice(
            bin_indices, 
            size=FIXED_SEQUENCE_LENGTH, 
            p=bin_probabilities
        )
        
        # 4. Generate a synthetic data point for each assigned bin
        synthetic_points = np.zeros(FIXED_SEQUENCE_LENGTH, dtype=np.float32)
        
        for i, bin_idx in enumerate(synthetic_bin_assignment):
            bin_min = bins[bin_idx]
            bin_max = bins[bin_idx + 1]
            synthetic_points[i] = np.random.uniform(bin_min, bin_max)
            
        final_array = synthetic_points
        
    if len(final_array) != FIXED_SEQUENCE_LENGTH:
        raise RuntimeError(f"Resampled size mismatch: Expected 1024, got {len(final_array)}")

    return final_array.astype(np.float32)


def _run_onnx_inference(input_tensor: np.ndarray) -> np.ndarray:
    """
    Runs the forward pass using the initialized global ONNX Runtime session.
    """
    if ONNX_SESSION is None:
        # Avoid crashing if the session failed to initialize due to critical errors
        raise RuntimeError("ONNX Runtime session is not initialized. Cannot run inference.")

    # Run the model using the global session, input name, and output name
    result = ONNX_SESSION.run([OUTPUT_NAME], {INPUT_NAME: input_tensor})
    
    return result[0].flatten()


# --- Main API Function ---

def predict(data_array: np.ndarray | list) -> tuple[np.ndarray, int]:
    """
    Main prediction function for the statspal package.
    Returns: (probabilities array, predicted_index integer)
    """
    if not isinstance(data_array, np.ndarray):
        data_array = np.array(data_array)

    # 1. Convert the data points into the 1024-length sample array
    try:
        resampled_data = _resample_data(data_array.flatten())
    except Exception:
        # Fallback if resampling logic fails unexpectedly
        return np.zeros(NUM_CLASSES), -1 

    # 2. Reshape input tensor to (1, 1024, 1)
    input_tensor = resampled_data.reshape(1, FIXED_SEQUENCE_LENGTH, 1).astype(np.float32)

    # 3. Run the forward pass using the global session
    try:
        logits = _run_onnx_inference(input_tensor)
    except RuntimeError:
        # Handles the case where ONNX_SESSION is None (model load failed at start)
        return np.zeros(NUM_CLASSES), -1 
    except Exception as e:
        print(f"Inference error during model execution: {e}")
        return np.zeros(NUM_CLASSES), -1 
    
    # 4. Apply Softmax to get probabilities
    exp_logits = np.exp(logits - np.max(logits)) 
    probabilities = exp_logits / np.sum(exp_logits)

    # 5. Determine the predicted class index
    predicted_index = np.argmax(probabilities)
    
    return probabilities, predicted_index

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
