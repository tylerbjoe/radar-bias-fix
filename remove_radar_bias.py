import numpy as np
import zipfile
import io
import matplotlib.pyplot as plt

def load_npy_from_zip(zip_path, npy_file_path_in_zip):
    """
    Loads a .npy file from within a zip archive without extracting it.
    
    Parameters:
    zip_path (str): The path to the zip file.
    npy_file_path_in_zip (str): The path to the .npy file within the zip archive.
    
    Returns:
    numpy.ndarray: The array stored in the .npy file inside the zip.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        with zip_file.open(npy_file_path_in_zip) as npy_file:
            file_data = npy_file.read()
            np_array = np.load(io.BytesIO(file_data))
            return np_array
        
zip_path = r'C:\\Users\\TJoe\\Documents\\Radar Offset Fix\\test_npy_10_15 1.zip'
npy_file_path = 'test_npy_10_15/Ava/Ava1/Radar_1/Radar_1_6.npy'

array_data = load_npy_from_zip(zip_path, npy_file_path)


plt.plot(array_data)

i=2
while (i <= 2069):
    npy_filt_path = f''