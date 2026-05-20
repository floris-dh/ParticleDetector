import numpy as np
import pandas as pd

"""Here I will attempt to extract the useful data from out between the noise.
A couple methods will be implemented.
These being: 
Analysing the gradient of the pulse, if this is beneath a certain treshold we clasify it as noise.
Checking the time interval of the pulse to long and it is likely noise, to short and it is noise, 
there will still be some room for alpha build up.
Many of the threshold values are copied from Olivers code. So big shout out.
The min and max length need to be adjusted to the sampling rate of the data, which is 50 kHz, so 1 sample is 0.02 ms.
"""

def filter_data(file_path, THL=-300, sample_rate_kHz=48):
    scale_factor = sample_rate_kHz / 48.0
    min_g = -20/scale_factor         #  minimal steepness of the pulse, it is vert big but that is expected for alpha particles
    max_g = -3300/scale_factor       # upper limit for the steepness, if it is steeper than this it is likely noise or some other bs
    min_length = int(round(44 *scale_factor))    # about 0.9 ms 
    max_length = int(round(120 *scale_factor))    # about 2.5 ms, tolerates some alpha-pileup
    min_skip = min_length 

    cleaned_data = []
    try:
        df = pd.read_pickle(file_path)
        for i, row in df.iterrows():
            y_raw = np.asarray(row['pulse'], dtype='int16')
            y = y_raw.copy() #considering the data will be altered in the future, we keep the original data for reference.
            dydx = np.gradient(y)

            gy = dydx.min()   
            gx = dydx.argmin() 
        
            while True:
                if y[0] > THL and y[0] < np.abs(THL) and y.min() < THL and y[gx] <= np.abs(THL) and gy < min_g and gy > max_g:
                    trigx=np.where(dydx < min_g)[0]
                    if len(trigx) == 0:
                        break
                    peakx1 = trigx[0]

                    crossing_check = y[peakx1 + min_length:] > y[peakx1]
                    crossing_x1 = crossing_check.argmax() if crossing_check.any() else -1
                    if crossing_x1 > 0:
                        peakx2=peakx1+crossing_x1+min_length #because crossing_x1 is offset by min_length!
                        diff = np.abs(peakx2 - peakx1)
                        peak = y[peakx1:peakx2].min()
                        if min_length <= diff <= max_length and peak < THL:
                            pure_peak = np.abs(peak) + y[peakx1]
                            cleaned_data.append(pure_peak)

                        peakx2_step = peakx1 + min_skip
                    else:
                        peakx2_step = peakx1 + min_skip
                else:
                        if len(y) < min_length:
                            break
                        peakx2_step = min_skip
                y = np.delete(y, slice(0, peakx2_step))
                if len(y) < min_length:
                        break
                else:
                    dydx = np.delete(dydx, slice(0, peakx2_step))
                    gy = dydx.min() 
                    gx = dydx.argmin()
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    return cleaned_data