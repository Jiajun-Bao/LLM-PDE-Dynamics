import numpy as np
from dataclasses import dataclass
from transformers import AutoTokenizer

@dataclass
class SimpleSerializerSettings:
    """
    Configuration settings for serializing and deserializing 2D numeric arrays.
    Attributes:
        space_sep (str): Separator between values within the same row
        time_sep (str): Separator between different rows (time steps)
    """
    space_sep: str = ","
    time_sep: str = ";"


def scale_value_to_150_850(value: float, v_min: float, v_max: float) -> int:
    """
    Scales a single float value (assumed in [v_min, v_max]) to an integer in [150, 850]
    Returns 500 if v_min == v_max (constant data).
    """
    if np.isclose(v_min, v_max):
        return 500
    ratio = (value - v_min) / (v_max - v_min)
    scaled = 150 + ratio * (850 - 150)

    return int(round(scaled))


def unscale_value_from_150_850(scaled_val: int, v_min: float, v_max: float) -> float:
    """
    Inverts the scale_value_to_150_850 operation, mapping an integer in [150, 850]
    back to a float in [v_min, v_max].
    """
    if np.isclose(v_min, v_max):
        return v_min
    ratio = (scaled_val - 150) / (850 - 150)
    
    return v_min + ratio * (v_max - v_min)


def scale_2d_array(data: np.ndarray, v_min=None, v_max=None) -> tuple:
    """
    Scales a 2D float array data to integer values in [150, 850].
    If v_min, v_max are not provided, uses the global min/max of data.
    Input:
        data (np.ndarray): 2D array of shape (time, space) with no missing values
        v_min (float or None): Optional lower bound
        v_max (float or None): Optional upper bound
    Output:
        scaled_array (np.ndarray): 2D int array, same shape, in [150,850]
        v_min (float): The actual min used
        v_max (float): The actual max used
    """
    if data.ndim != 2:
        raise ValueError("scale_2d_array expects a 2D array.")
    if v_min is None:
        v_min = np.min(data)
    if v_max is None:
        v_max = np.max(data)
    scaled_array = np.zeros_like(data, dtype=int)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            scaled_array[i, j] = scale_value_to_150_850(val, v_min, v_max)

    return scaled_array, v_min, v_max


def unscale_2d_array(scaled_data: np.ndarray, v_min: float, v_max: float) -> np.ndarray:
    """
    Unscales a 2D array of integers in [150, 850] back to float values in [v_min, v_max].
    Input:
        scaled_data (np.ndarray): 2D integer array
        v_min (float): The min used during scaling
        v_max (float): The max used during scaling
    Output:
        unscaled (np.ndarray): 2D float array of the same shape
    """
    if scaled_data.ndim != 2:
        raise ValueError("unscale_2d_array expects a 2D array.")
    unscaled = np.zeros_like(scaled_data, dtype=float)
    for i in range(scaled_data.shape[0]):
        for j in range(scaled_data.shape[1]):
            val = scaled_data[i, j]
            unscaled[i, j] = unscale_value_from_150_850(val, v_min, v_max)

    return unscaled


def serialize_2d_integers(scaled_data: np.ndarray, settings: SimpleSerializerSettings) -> str:
    """
    Serializes a 2D integer array (in [150,850]) into a single string.
    Input:
        scaled_data (np.ndarray): 2D integer array (time, space) in [150,850]
        settings (SimpleSerializerSettings): Provides separators
    Output:
        str: The 2D array in a textual form
    """
    if scaled_data.ndim != 2:
        raise ValueError("serialize_2d_integers expects a 2D array.")
    rows_str = []
    for row in scaled_data:
        items = [str(val) for val in row]
        row_str = settings.space_sep.join(items)
        rows_str.append(row_str)

    return settings.time_sep.join(rows_str)


def deserialize_2d_integers(text: str, settings: SimpleSerializerSettings) -> np.ndarray:
    """
    Converts a serialized string of integers back to a 2D integer array.
    Input:
        text (str): Textual representation from serialize_2d_integers
        settings (SimpleSerializerSettings): Provides separators
    Output:
        scaled_data (np.ndarray): 2D integer array (time, space)
    """
    if not text.strip():
        return np.array([], dtype=int)
    time_blocks = text.split(settings.time_sep)
    all_rows = []
    for block in time_blocks:
        block = block.strip()
        if not block:
            continue
        str_vals = block.split(settings.space_sep)
        row_vals = [int(s.strip()) for s in str_vals]
        all_rows.append(row_vals)

    return np.array(all_rows, dtype=int)


def extract_training_and_test(full_serialized_data, input_time_steps, settings):
    """
    Extracts input and test data from a serialized 2D integer dataset produced by serialize_2d_integers.
    The dataset is assumed to be a single string with rows (time slices) separated by settings.time_sep.
    Input:
        full_serialized_data (str): The serialized 2D integer array (time x space)
        input_time_steps (int): Number of initial time slices to use for input
        settings (SimpleSerializerSettings): Serialization settings
    Output:
        train_serial (str): Serialized string of the first input_time_steps time slices.
        test_serial (str): Serialized string of the (input_time_steps+1)-th time slice.
    Raises:
        ValueError: If input_time_steps is not less than the total number of time slices.
    """
    # Filtering out any accidental empty rows (e.g. due to a trailing separator)
    rows = [row.strip() for row in full_serialized_data.split(settings.time_sep) if row.strip()]
    num_rows = len(rows)
    if input_time_steps >= num_rows:
        raise ValueError(
            f"Invalid input_time_steps={input_time_steps}. "
            f"Data only has {num_rows} time slices. "
            "Ensure input_time_steps < total number of time slices."
        )
    # Add the time separator after each row so that the formatting matches the original
    train_serial = settings.time_sep.join(rows[:input_time_steps]) + settings.time_sep
    test_serial = rows[input_time_steps]
    
    return train_serial, test_serial


def check_tokenization(MODEL_NAME):
    """
    Checks that all three-digit numbers ("000" to "999"), along with
    the ("," ';') character, tokenize to exactly one token
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    good_tokens = {}
    warnings = []
    total_checked = 0

    for i in range(1000):
        s = f"{i:03d}"
        tokenized = tokenizer(s, add_special_tokens=False)
        token_ids = tokenized.input_ids
        total_checked += 1
        if len(token_ids) != 1:
            warning_message = f"Warning: '{s}' tokenized to {len(token_ids)} tokens: {token_ids}"
            print(warning_message)
            warnings.append(warning_message)
        else:
            good_tokens[s] = token_ids[0]
            
    for symbol in [",", ';']:
        tokenized = tokenizer(symbol, add_special_tokens=False)
        token_ids = tokenized.input_ids
        total_checked += 1
        if len(token_ids) != 1:
            warning_message = f"Warning: '{symbol}' tokenized to {len(token_ids)} tokens: {token_ids}"
            print(warning_message)
            warnings.append(warning_message)
        else:
            good_tokens[symbol] = token_ids[0]

    print("\n=== Summary Status ===")
    print(f"Total items checked: {total_checked}")
    print(f"Successfully tokenized items (exactly one token): {len(good_tokens)}")
    print(f"Total warnings: {len(warnings)}")
    if warnings:
        print("Warnings were issued for the above items that did not tokenize as expected.")

    return good_tokens