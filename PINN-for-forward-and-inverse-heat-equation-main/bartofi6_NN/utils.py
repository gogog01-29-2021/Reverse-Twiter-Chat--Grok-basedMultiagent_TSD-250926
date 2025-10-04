__all__ = [
    "find_first_missing_number_in_file_names",
    "proccess_loss_history_for_plotting",
    "clear_directory",
    "print_tensor",
    "print_shape_and_value",
    "find_max_number_file",
    "print_tensor_ref",
]

def print_tensor_ref(arg, name_str):
    import tensorflow as tf

    message = f"{name_str}.ref: {arg.ref}"
    tf.print(message)


def print_tensor(tensor):
    import tensorflow as tf

    tf.print(tensor, summarize=-1, end="\n\n")


def print_shape_and_value(arg, name=""):
    import tensorflow as tf

    tf.print(name + " shape:")
    tf.print(arg.shape)
    tf.print(name + ":")
    print_tensor(arg)


def find_first_missing_number_in_file_names(directory):
    import os
    import re

    # Extract the starting number from each file in the directory
    numbers = []
    for filename in os.listdir(directory):
        match = re.match(r"^(\d+)", filename)
        if match:
            numbers.append(int(match.group(1)))

    # Sort the list of numbers
    numbers.sort()

    # remove duplicates from the list
    numbers = list(set(numbers))

    # Find the first missing number in the sequence
    missing_number = 1
    for number in numbers:
        if number == missing_number:
            missing_number += 1
        else:
            break

    return missing_number


def proccess_loss_history_for_plotting(list_of_numbers, chunk_size):
    import numpy as np

    numbers_array = np.array(list_of_numbers)

    number_of_full_chunks = len(numbers_array) // chunk_size
    means = np.empty(1 + number_of_full_chunks + 1)
    means[0] = numbers_array[0]

    x_values = np.empty(1 + number_of_full_chunks + 1)
    x_values[0] = 1

    for i in range(number_of_full_chunks):
        means[i+1] = np.mean(numbers_array[i*chunk_size : (i+1)*chunk_size])
        x_values[i+1] = (i+1)*chunk_size

    means[-1] = np.mean(numbers_array[number_of_full_chunks*chunk_size : ])
    x_values[-1] = len(numbers_array)

    return x_values, means


def clear_directory(directory):
    import os

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        try:
            # Check if the current path is a file
            if os.path.isfile(filepath):
                # print(f"clearing {filepath}")
                # Remove the file
                os.remove(filepath)
            # If it's a directory, recursively call clear_directory
            elif os.path.isdir(filepath):
                clear_directory(filepath)
        except Exception as e:
            print(f"Failed to delete {filepath}: {e}")

def find_max_number_file(directory, prefix, file_type=None):
    import os
    import re

    max_number = -1
    max_file_name = None

    if file_type != None:
        pattern = re.compile(prefix + r'(\d+)\.' + file_type)
    else:
        pattern = re.compile(prefix + r'(\d+)')
    
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            number = int(match.group(1))
            if number > max_number:
                max_number = number
                max_file_name = filename
    
    return max_file_name


