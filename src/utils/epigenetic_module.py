

import os
import sys
import tqdm
import numpy as np
import math

# Add built C++ module to the search path
sys.path.append('./src/cpp_module/build')

import bigwig_cpp_reader


def exist_npz_file(config: dict, dataset: dict, npz_path: str) -> bool:
    if not os.path.exists(npz_path):
        return False
    else:
        signal_arrays = np.load(npz_path)["signal_arrays"]
        if (signal_arrays.shape[0] != len(dataset["chrom"])) or (signal_arrays.shape[1] != (config["window_size"]*2 // config["bin_size"])):
            return False
        else:
            print("Skipping BigWig data processing as the npz file already exists and is valid.")
            return True


def bin_average_array(signal_array: np.ndarray, start: int, end: int, bin_size: int) -> np.ndarray:
    # Consider nans in the signal array
    if signal_array.size == 0:
        return np.full(math.ceil((end - start) / bin_size), np.nan, dtype=np.float32)
    
    num_bins = math.ceil((end - start) / bin_size)
    if num_bins <= 0:
        return np.full(math.ceil((end - start) / bin_size), np.nan, dtype=np.float32)
    
    # Aggregate the single row by reshaping sequrence
    array_size = len(signal_array)
    padded_size = (array_size // bin_size) * bin_size
    reshaped_signal_array = signal_array[:padded_size].reshape(-1, bin_size)
    # Cakculate the mean of each bin
    binned_signal_array = np.nanmean(reshaped_signal_array, axis=1)
    
    # Process the last bin if it is not full
    remainder_size = array_size % bin_size
    if remainder_size > 0:
        remainder_bin = signal_array[padded_size:]
        binned_signal_array = np.append(binned_signal_array, np.nanmean(remainder_bin))
    
    return binned_signal_array


def save_bigwig_data_as_npz(config: dict, dataset: dict, bigwig_path: str, npz_path: str) -> None:
    if not os.path.exists(bigwig_path):
        raise FileNotFoundError(f"BigWig file not found: {bigwig_path}")
    print(f"Processing BigWig data from {bigwig_path} and saving to {npz_path}")
    chrom_list = dataset["chrom"]
    center_pos  = [(int(start) + int(end)) // 2 for start, end in zip(dataset["start_pos"], dataset["end_pos"])]
    strand_list = dataset["strand"]
    
    type_of_data = config["type_of_data"]
    window_size = config["parameters"]["window_size"][type_of_data] # ±
    bin_size = config["parameters"]["bin_size"][type_of_data]

    # Create a BigWigReader instance
    bw_reader = bigwig_cpp_reader.BigWigReader(bigwig_path)
    
    import time
    start_time = time.time()
    
    signal_arrays = np.zeros((len(dataset["chrom"]), window_size*2 // bin_size), dtype=np.float32)
    for i, (pos, chrom, strand) in tqdm.tqdm(enumerate(zip(center_pos, chrom_list, strand_list)), total=len(chrom_list), desc="Processing BigWig data"):
        start = pos - window_size
        end = pos + window_size
        signal_array, actual_start, actual_end = bw_reader.get_values(chrom, start, end)
        signal_array = bigwig_cpp_reader.adjust_array_length(signal_array, start, end, actual_start, actual_end)
        signal_array = bin_average_array(signal_array, start, end, bin_size)
        if strand == "+":
            signal_arrays[i] = signal_array
        elif strand == "-":
            signal_arrays[i] = signal_array[::-1]
    
    end_time = time.time()
    print(f"Processed {len(chrom_list)} regions in {end_time - start_time:.2f} seconds")
            
    # Save the signal arrays to a .npz file
    np.savez(npz_path, signal_arrays=signal_arrays)
    # signal_arrays = np.load(npz_path)["signal_arrays"]
    print(f"Saved BigWig data to {npz_path} with shape {signal_arrays.shape}")


def load_npz(signal_array_path_list: list) -> np.ndarray:
    if len(signal_array_path_list) == 1:
        with np.load(signal_array_path_list[0]) as data:
            signal_arrays = data["signal_arrays"]
    else:
        with np.load(signal_array_path_list[0]) as data:
            signal_arrays = data["signal_arrays"]
        signal_arrays = np.load(signal_array_path_list[0])["signal_arrays"]
        for path in signal_array_path_list[1:]:
            temp_signal_arrays = np.load(path)["signal_arrays"]
            if temp_signal_arrays.shape != signal_arrays.shape:
                raise ValueError(f"Signal arrays in {path} have different shape.")
            signal_arrays += temp_signal_arrays
        signal_arrays /= len(signal_array_path_list)
    # Removing outliers
    q1 = np.nanpercentile(signal_arrays, 25)
    q3 = np.nanpercentile(signal_arrays, 75)
    iqr = q3 - q1
    lower_bound = q1 - 2 * iqr
    upper_bound = q3 + 2 * iqr
    signal_arrays[signal_arrays < lower_bound] = lower_bound
    signal_arrays[signal_arrays > upper_bound] = upper_bound

    # Z-score normalization
    # signal_arrays = np.log1p(signal_arrays)  # log1p transformation
    mean = np.nanmean(signal_arrays)
    std = np.nanstd(signal_arrays)
    signal_arrays = (signal_arrays - mean) / std
    signal_arrays = np.array(signal_arrays, dtype=np.float16)
    return signal_arrays


def load_epigenetic_feature(config: dict, dataset_dict: dict) -> dict:
    epigenetic_features_data = {}
    for type_of_data in config["using_epi_data"]:
        window_size = config["parameters"]["window_size"][type_of_data]
        bin_size = config["parameters"]["bin_size"][type_of_data]
        signal_array_path_list = config["paths"]["epigenetic"][type_of_data]["npz_current"]
        signal_array = load_npz(signal_array_path_list)
        if signal_array.shape != (len(dataset_dict["sgrna"]), window_size*2 // bin_size):
            raise ValueError(f"Signal array shape {signal_array.shape} does not match the expected shape {(len(dataset_dict['sgrna']), window_size*2 // bin_size)} for {type_of_data}.")
        signal_array = np.nan_to_num(signal_array, nan=0.0)
        epigenetic_features_data[type_of_data] = signal_array
    dataset_dict.update({"epigenetic_features": epigenetic_features_data})
    return dataset_dict