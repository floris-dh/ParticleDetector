import time
import pickle
import dwfpy as dwf

# --- CONFIGURATION PARAMETERS ---
NUM_PULSES_TO_RECORD = 100         # Stop recording after this many pulses
OUTPUT_FILENAME = "recorded_pulses.pkl"

# Channel 1 parameters based on the image
CHANNEL_INDEX = 0                  # Channel 1 is index 0
CHANNEL_RANGE = 0.5                # Total voltage range (Volts)
CHANNEL_OFFSET = 0.0               # Volts offset

# Trigger parameters matching the image
TRIGGER_LEVEL = -0.055             # -55 mV trigger level
TRIGGER_SLOPE = "falling"          # Negative-going pulse trigger
TRIGGER_HYSTERESIS = 0.005         # 5 mV hysteresis to handle noise range

# Acquisition timing parameters
SAMPLE_RATE = 20e6                 # 20 MHz sampling frequency
# Window length per pulse (e.g., 500 microseconds = 10,000 samples)
PULSE_WINDOW_SAMPLES = 10000       


def main():
    print(f"DWF version: {dwf.Application.get_version()}")
    captured_waveforms = []

    print("Opening device...")
    with dwf.Device() as device:
        scope = device.analog_input
        
        # 1. Setup Channel 1 parameters
        scope[CHANNEL_INDEX].setup(range=CHANNEL_RANGE, offset=CHANNEL_OFFSET)
        
        # 2. Setup the Edge Trigger mimicking the image
        scope.setup_edge_trigger(
            mode="normal",               # Hardware halts stream until a pulse happens
            channel=CHANNEL_INDEX, 
            slope=TRIGGER_SLOPE, 
            level=TRIGGER_LEVEL, 
            hysteresis=TRIGGER_HYSTERESIS
        )
        
        print(f"\nStarting continuous streaming capture...")
        pulse_count = 0
        
        # We start a streaming record session. 
        # By setting a long or infinite total time, Python continually polls the instrument.
        # The hardware handles the triggering instantly at the silicon level.
        for samples in scope.record(sample_rate=SAMPLE_RATE, record_length=0.0):
            
            # Check if the hardware state registers a trigger event
            if scope.status == dwf.Status.TRIGGERED or scope.status == dwf.Status.DONE:
                # Extract the active channel data slice from the streamed chunk
                channel_data = list(samples[CHANNEL_INDEX])
                
                # If the chunk contains enough data for our window, isolate it
                if len(channel_data) >= PULSE_WINDOW_SAMPLES:
                    pulse_data = {
                        "pulse_id": pulse_count,
                        "timestamp": time.time(),
                        "data": channel_data[:PULSE_WINDOW_SAMPLES]
                    }
                    captured_waveforms.append(pulse_data)
                    pulse_count += 1
                    
                    print(f" Captured pulse #{pulse_count}")
            
            # Break out of the infinite record loop once our quota is fulfilled
            if pulse_count >= NUM_PULSES_TO_RECORD:
                break

    # 3. Save data out to a binary pickle file
    print(f"\nSaving data to {OUTPUT_FILENAME}...")
    with open(OUTPUT_FILENAME, "wb") as f:
        pickle.dump(captured_waveforms, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"Successfully finalized. Recorded {len(captured_waveforms)} pulses.")

if __name__ == "__main__":
    main()