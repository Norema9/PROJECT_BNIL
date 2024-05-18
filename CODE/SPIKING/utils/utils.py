
import snntorch.spikegen as spikegen
import numpy as np

def normalize(data):
    """ 
    The values of the inputs are normalized to a range suitable for 
    the SNN, typically between 0 and 1. This helps in mapping the inputs 
    values to a spike rate effectively.
    
    Parameters:
    data (list or numpy array): The input data to be normalized.
    
    Returns:
    list or numpy array: Normalized data with values between 0 and 1.
    """
    min_val = min(data)
    max_val = max(data)
    normal_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normal_data


def convert2spike_rate(normalized_data, time_window):
    """
    This function converts the normalized input values into spike rate.
    
    Parameters:
    normalized_data (list or numpy array): The normalized input data with values between 0 and 1.
    time_window (int): The duration over which the spike rate is calculated.
    
    Returns:
    numpy array: A 2D array representing spike trains for each input value.
    """
    # Convert the normalized data to spike rates
    spike_trains = []
    for value in normalized_data:
        spike_train = spikegen.rate(value, num_steps = time_window)
        spike_trains.append(spike_train)
    
    return np.array(spike_trains)
    

def generate_spike():
    """
    This generates spikes at each time step for the duration 
    of the simulation, based on the spike rates """
    pass