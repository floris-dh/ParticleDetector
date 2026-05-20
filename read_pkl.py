import pickle as pkl
import pandas as pd

def read_pickle_file(pickle_file_path):
    with open(pickle_file_path, "rb") as f:
        data = pkl.load(f)
    return data

data = pd.DataFrame(read_pickle_file("combined_waveforms.pkl"))

print(data['metadata'][0])