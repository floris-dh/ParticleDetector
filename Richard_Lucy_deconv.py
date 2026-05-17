import numpy as np
import scipy.signal as signal

def richard_lucy_deconvolution(measurement, response_function, iterations=20):
    #Normalise response function. 
    response_function = response_function / np.sum(response_function)
    initial_guess = np.copy(measurement).astype(float)  
    #flip the response function for convolution
    flipped_response = response_function[::-1]
    
    for i in range(iterations):
        #prediction based on current guess
        og_prediction = signal.convolve(initial_guess, response_function, mode='same')
        ratio = measurement / (og_prediction + 1e-10)  # avoid division by zero
        
        correction = signal.convolve(ratio, flipped_response, mode='same')
        initial_guess *= correction  
    return initial_guess

def gauss(length, sigma):
    #apparantly u cant have floats in np.linspace, therefore we // in order to get an integer.
    x=np.linspace(-length//2, length//2, length)
    gauss = np.exp((-0.5)*(x**2)/(sigma**2))
    norm_gauss = gauss / np.sum(gauss)
    return norm_gauss