import sys
import tensorflow as tf

def print_tensor_shape_and_type(tensor_as_fstring_formatted_with_equal_sign):
    # tensor_name_in_code_and_value_as_string = f"{tensor=}"
    # # For example:
    # #   a = 42
    # #   f"{a=}"
    # # returns the string a=42. From that I can extract the name of the variable as it is written in code,
    # # and also the value of the variable.

    # Extracting the name, as is written in code, and the value:
    tensor_name, tensor_value = tensor_as_fstring_formatted_with_equal_sign.split("=", 1)

    print(f"{tensor_name+':'}{tensor_value}")
    return


def tfprint_tensor_shape_and_type(tensor_name, tensor):
    tf.print(tensor_name + ' dtype: ', tensor.dtype, ", shape: ", tf.shape(tensor), sep="", output_stream=sys.stdout)
    return


def tfprint_tensor_values(tensor_name, tensor, summarize=None):
    return tf.print(tensor_name + ":", tensor, summarize=summarize, output_stream=sys.stdout)


# def tf_print_tensor_shape_and_type(tensor_as_fstring_formatted_with_equal_sign: str):
#     # tensor_name_in_code_and_value_as_string = f"{tensor=}"
#     # # For example:
#     # #   a = 42
#     # #   f"{a=}"
#     # # returns the string a=42. From that I can extract the name of the variable as it is written in code,
#     # # and also the value of the variable.

#     # Extracting the name, as is written in code, and the value:
#     tensor_name, tensor = tensor_as_fstring_formatted_with_equal_sign.split("=", 1)

#     tf.print(tensor_name + " shape: ", tf.shape(tensor), ", dtype: ", tensor.dtype, output_stream=sys.stdout, sep="")
    
#     # tf.print(tf.shape(tensor), end=", ")
#     # tf.print(f"{tensor_name+': '}{tf.shape(tensor)}")
#     return


def raise_error(error_type, error_message):
    raise error_type(error_message)
