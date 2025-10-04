import tensorflow as tf
import numpy as np
from collections.abc import Iterable
import itertools
import os
from pathlib import Path
import csv

# __package__ = ["training_settings_generator"]


def L1_norm(x):
    return tf.reduce_mean(tf.abs(x))


def L2_norm(x):
    return tf.reduce_mean(tf.square(x))


def max_norm(x):
    return tf.reduce_max(tf.abs(x))


def assert_equal_shapes(x, y, **kwargs):
    return tf.debugging.assert_equal(tf.shape(x), tf.shape(y), **kwargs)


def assert_finite_values(x):
    return tf.debugging.assert_all_finite(x, "Tensor must have finite values.")

def generate_letter_from_activation_function(activation_function):
    if activation_function == "relu" or activation_function == tf.keras.activations.relu:
        return "R"
    elif activation_function == "sigmoid" or activation_function == tf.keras.activations.sigmoid:
        return "S"
    elif activation_function == "tanh" or activation_function == tf.keras.activations.tanh:
        return "T"
    elif activation_function == None:
        return "I"
    else:
        return f"Conversion into letter from activation function not implemented for {activation_function}."


def check_stop_value_for_range_generation(start, stop, step):
    """
    If range wouldn't generate any number from these start, stop and step values,
    the function changes stop such that range returns the start value.
    Returns the changed stop value.
    .
    """
    if step >= 0:
        if stop <= start:
            stop = (start + step)
    else:
        if stop >= start:
            stop = (start + step)
    return stop


def _range_from_tuple(converted_tuple: tuple):
    # Checking if float or integer range should be created
    create_float_range = False
    for value in converted_tuple:
        if isinstance(value, float) and not value.is_integer():
            create_float_range = True

    start_value = converted_tuple[0]
    stop_value = converted_tuple[1]
    step_value = converted_tuple[2]
    stop_value = check_stop_value_for_range_generation(start_value, stop_value, step_value)

    if create_float_range:
        return np.arange(start_value, stop_value, step_value)
    else:
        return range(int(start_value), int(stop_value), int(step_value))


def training_settings_generator(config: dict, parameter_names_for_custom_conversion: list | tuple, function_to_convert_custom_parameters):
    """
    Generates combinations of training settings from training parameters.
    If some parameters need custom conversion into iterables, specify their names and provide a funtion that does desired conversion.
    
    Args:
    config (dict):
        - A dictionary containing training parameter names and their values.
        - Keys: Training parameter names.
        - Values:
            - Tuple of length 3 will be converted into range.
                - Interpreted as (start, stop, step).
                - Automatic determination for int and float ranges.
            - List for discrete choices.
            - If the value isn't an iterable, it will be put into a list so that it becomes an iterable for itertools.

    parameter_names_for_custom_conversion:
        - A list or tuple containing the names of parameters that are supposed to have custom conversion logic implemented.

    function_to_convert_custom_parameters(config, parameter_names_for_custom_conversion, training_setting_dictionary)
                                         -> tuple(new_parameter_names, iterable_with_combinations_of_values_for_new_parameters):
        - Function that provides the logic for custom conversion of specified parameters into iterable containing combinations of the new parameter values.
        - First all non-custom parameters are converted into iterables and then the custom parameters are converted using this provided function.
        - The single combinations of training settings already generated from non-custom parameters are available as a dictionary in the argument training_setting_dictionary.
            - For example, the dictionary training_setting_dictionary could contain pair "num_of_epochs": 50.
        - Args:
            - config: Same config as is passed into training_settings_generator.
            - parameter_names_for_custom_conversion: tuple or list of strings with the names of parameters intended for custom conversion.
            - training_setting_dictionary: dictionary containing parameter names and corresponding parameter values for given training setting which were already generated from non-custom parameters.
        - Returns tuple containing the folowing:
            - new_parameter_names: A list or tuple containing names of new parameters, whose values are supposed to be added to the training_setting_dictionary.
            - iterable_with_combinations_of_values_for_new_parameters: An iterable containing combinations of the values for the new parameters.
                - Each combination should have length equal to the number of names in new_parameter_names, so that it can be zipped together with the names and be added into the training_setting_dictionary.
                - The i-th value should of course be the value for the i-th parameter specified by its name, the i-th name in new_parameter_names.

    Yields:
    dict: A dictionary containing one training setting.
        - Keys: Training parameter names.
        - Values: Training parameters' values for a given training setting.
    """
    dict_with_iterables_made_from_parameter_values = dict()


    ############################### CONVERTING VALUES INTO CORRECT ITERABLES FOR ITERTOOLS.PRODUCT ###############################
    for parameter_name, parameter_value in config.items():
        if parameter_name in parameter_names_for_custom_conversion:
            continue

        elif isinstance(parameter_value, tuple) and len(parameter_value) == 3:
            dict_with_iterables_made_from_parameter_values[parameter_name] = _range_from_tuple(parameter_value)

        elif not isinstance(parameter_value, Iterable) or isinstance(parameter_value, str):
            dict_with_iterables_made_from_parameter_values[parameter_name] = [parameter_value]
            # String needs to be put into an iterable (list for example) so that itertools.product doesn't turn it into single letters.

        else:
            dict_with_iterables_made_from_parameter_values[parameter_name] = parameter_value


    ############################### GENERATING TRAINING SETTINGS ###############################
    names_of_training_parameters = tuple(dict_with_iterables_made_from_parameter_values.keys())
    values_for_training_parameters = dict_with_iterables_made_from_parameter_values.values()

    for single_combination_of_values_for_training_parameters in itertools.product(*values_for_training_parameters):

        training_setting_dictionary = dict(zip(names_of_training_parameters, single_combination_of_values_for_training_parameters))

        new_parameter_names, iterable_with_combinations_of_values_for_new_parameters = function_to_convert_custom_parameters(config, parameter_names_for_custom_conversion, training_setting_dictionary)

        for single_combination_of_values_for_new_parameters in iterable_with_combinations_of_values_for_new_parameters:
            
            new_entry_to_training_setting_dictionary = dict(zip(new_parameter_names, single_combination_of_values_for_new_parameters))
            training_setting_dictionary.update(new_entry_to_training_setting_dictionary)
            yield training_setting_dictionary


# Přidat tuhle funkci později do training_settings_generátoru: v generátoru vytvořím ten csv file a directory
#   a přidám je do configu, tim je budu mít k dispozici furt.
# Stejně tak přidat vytváření individual training run directory (možná přejmenovat: single místo individual):
#   ve funkci generátoru to vytvořit a přidat to do toho výslednýho generátoru, aby to bylo k dispozici.
def create_main_save_dir_and_main_csv_file(config: dict, data: dict):
    main_save_dir = config["main_save_dir"]
    if not os.path.exists(main_save_dir):
        os.makedirs(main_save_dir)

    main_csv_file = os.path.join(main_save_dir, "main.csv")
    main_csv_file_path = Path(main_csv_file)
    if config["override_main_csv_file"] and not main_csv_file_path.exists():
        main_csv_file_path.touch()

    config["main_csv_file"] = main_csv_file
    return
