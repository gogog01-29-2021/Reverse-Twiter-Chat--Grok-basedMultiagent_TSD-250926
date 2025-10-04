import itertools
import tensorflow as tf
import vtk
import os
import numpy as np
from ....PINN.HE_2D_Pde_Constrained_optimalization_on_circle import *
from ..general import _range_from_tuple


def setup_data(data, config):
    """
    Implements additional data processing desired or needed for given problem.
    .
    """
    circle_radius = data["circle_radius"]
    circle_center_in_xy = data["circle_center_in_xy"]
    x_start = circle_center_in_xy[0] - circle_radius
    x_stop = circle_center_in_xy[0] + circle_radius
    y_start = circle_center_in_xy[1] - circle_radius
    y_stop = circle_center_in_xy[1] + circle_radius
    t_start = data["t_start"]
    t_stop = data["t_stop"]
    num_of_x_plot_points = data["num_of_x_plot_points"]
    num_of_y_plot_points = data["num_of_y_plot_points"]
    num_of_time_plots = data["num_of_time_plots"]

    x = tf.linspace(x_start, x_stop, num=num_of_x_plot_points)
    y = tf.linspace(y_start, y_stop, num=num_of_y_plot_points)
    t = tf.linspace(t_start, t_stop, num=num_of_time_plots)
    # x = x[1:-1]
    # y = y[1:-1]
    # #   The condition for circle doesn't include points from the edges of the grid.
    # # If points in the circle are taken from the whole grid, the resulting ragged tensor will contain useless empty parts []
    # # corresponding to the points on the edges of the grid, as none of the points from any edge were included in the final circle.
    # # Thus we throw the edges of the grid out before we make the circle in order to not have empty parts (empty rows/columns)
    # # in the final Ragged tensor.

    X_plot, Y_plot, T_plot = tf.meshgrid(x, y, t, indexing='ij')
    # Indexování 'ij' zařídí, že bod na indexu [i,j,k] odpovídá bodu s hodnotami x[i], y[j], t[k].
    # Tedy prostě standardní kartészké indexování: z hodnot x, y a t se vygeneruje 3D mřížka a X, Y a T určují
    # hodnoty x, y a t v daných bodech. Např. hodnota x v bodě s indexy i,j,k je X[i,j,k], atd.

    data["X_plot"] = X_plot
    data["Y_plot"] = Y_plot
    data["T_plot"] = T_plot


    data["dictionary_with_norms_after_training_for_different_cost_functional_coeficients"] = {}


def _convert_activation_functions(config, parameter_names_for_custom_conversion, training_setting_dictionary, network_name: str):
    num_of_layers = training_setting_dictionary[f"{network_name}_num_of_layers"]
    list_with_activation_functions = config[f"{network_name}_activation_functions_list"]
    iterable_containing_list_with_activation_functions = (list_with_activation_functions,)

    if not config[f"{network_name}_repeat_activation_functions"]:
        iterable_with_settings_for_activation_functions = itertools.product(list_with_activation_functions, repeat=num_of_layers)
    else:
        long_list_with_activation_functions = list_with_activation_functions * num_of_layers
        precise_length_list_with_activation_functions = long_list_with_activation_functions[:num_of_layers]
        iterable_with_settings_for_activation_functions = (precise_length_list_with_activation_functions,)
        # In this case, there is only one setting, put into tuple so it is iterable.
        # It could have been also put into a list or different iterable, that doesn't really matter much.

    return iterable_containing_list_with_activation_functions, iterable_with_settings_for_activation_functions


def function_to_convert_custom_parameters(config, parameter_names_for_custom_conversion, training_setting_dictionary):
    (iterable_with_u_nn_activation_functions_list,
        iterable_with_settings_for_u_nn_activation_functions) = _convert_activation_functions(config,
                                                                                              parameter_names_for_custom_conversion,
                                                                                              training_setting_dictionary,
                                                                                              "u_nn")

    (iterable_with_q_nn_activation_functions_list,
        iterable_with_settings_for_q_nn_activation_functions) = _convert_activation_functions(config,
                                                                                              parameter_names_for_custom_conversion,
                                                                                              training_setting_dictionary,
                                                                                              "q_nn")


    new_parameter_names = ["list_with_activation_functions_to_use_in_general_for_u_nn",
                           "list_with_current_activation_functions_for_u_nn",
                           "list_with_activation_functions_to_use_in_general_for_q_nn",
                           "list_with_current_activation_functions_for_q_nn"]
    iterable_with_combinations_of_values_for_new_parameters = itertools.product(iterable_with_u_nn_activation_functions_list,
                                                                                iterable_with_settings_for_u_nn_activation_functions,
                                                                                iterable_with_q_nn_activation_functions_list,
                                                                                iterable_with_settings_for_q_nn_activation_functions)


    ### Comment valid only if one argument for itertools.product is given.
    # If we want to generate combinations from just one iterable, we could just put it into tuple or something and return it that way,
    #   instead of passing it into itertools.product.
    # However if we wanted to generate multiple combinations, we would need itertools.product,
    #   thus I am keeping the use of itertools.product even for just one iterable, so that in the future,
    #   the addition of more iterables is easy.

    return (new_parameter_names, iterable_with_combinations_of_values_for_new_parameters)


def function_to_convert_activation_functions_and_num_of_training_points(config, parameter_names_for_custom_conversion, training_setting_dictionary):
    (iterable_with_u_nn_activation_functions_list,
        iterable_with_settings_for_u_nn_activation_functions) = _convert_activation_functions(config,
                                                                                              parameter_names_for_custom_conversion,
                                                                                              training_setting_dictionary,
                                                                                              "u_nn")

    (iterable_with_q_nn_activation_functions_list,
        iterable_with_settings_for_q_nn_activation_functions) = _convert_activation_functions(config,
                                                                                              parameter_names_for_custom_conversion,
                                                                                              training_setting_dictionary,
                                                                                              "q_nn")


    new_parameter_names = ["list_with_activation_functions_to_use_in_general_for_u_nn",
                           "list_with_current_activation_functions_for_u_nn",
                           "list_with_activation_functions_to_use_in_general_for_q_nn",
                           "list_with_current_activation_functions_for_q_nn"]
    iterable_with_combinations_of_settings_for_activation_functions = itertools.product(iterable_with_u_nn_activation_functions_list,
                                                                                        iterable_with_settings_for_u_nn_activation_functions,
                                                                                        iterable_with_q_nn_activation_functions_list,
                                                                                        iterable_with_settings_for_q_nn_activation_functions)
    iterable_with_combinations_of_settings_for_activation_functions = list(iterable_with_combinations_of_settings_for_activation_functions)

    if config["put_num_of_t_training_points_as_num_of_x_and_y_training_points"]:
        setting_value_for_num_of_t_training_points = config["num_of_t_training_points"]

        if len(setting_value_for_num_of_t_training_points) == 3 and isinstance(setting_value_for_num_of_t_training_points, tuple):
            iterable_with_number_of_training_points = _range_from_tuple(setting_value_for_num_of_t_training_points)
        elif isinstance(setting_value_for_num_of_t_training_points, list):
            iterable_with_number_of_training_points = setting_value_for_num_of_t_training_points

        iterable_with_combinations_of_values_for_new_parameters = []
        for combination_of_settings_for_activation_functions in iterable_with_combinations_of_settings_for_activation_functions:
            for number_of_training_points in iterable_with_number_of_training_points:
                # Each combination_of_settings_for_activation_functions is a tuple because of itertools.product().
                list_with_number_of_training_points = [number_of_training_points] * 3
                new_setting_combination = list(combination_of_settings_for_activation_functions) + list_with_number_of_training_points
                iterable_with_combinations_of_values_for_new_parameters.append(tuple(new_setting_combination))

        names_for_number_of_training_points_parameter = ["num_of_t_training_points", "num_of_grid_x_training_points", "num_of_grid_y_training_points"]
        new_parameter_names.extend(names_for_number_of_training_points_parameter)

    ### Comment valid only if one argument for itertools.product is given.
    # If we want to generate combinations from just one iterable, we could just put it into tuple or something and return it that way,
    #   instead of passing it into itertools.product.
    # However if we wanted to generate multiple combinations, we would need itertools.product,
    #   thus I am keeping the use of itertools.product even for just one iterable, so that in the future,
    #   the addition of more iterables is easy.

    return (new_parameter_names, iterable_with_combinations_of_values_for_new_parameters)


def save_weighted_norms_calculated_after_training(training_setting_dictionary: dict, data: dict):
    if "dictionary_with_weighted_norms_history" not in data.keys():
        data["dictionary_with_weighted_norms_history"] = {
            "L1_equation_norm": [],
            "L2_equation_norm": [],
            "max_equation_norm": [],
            "L1_init_cond_norm": [],
            "L2_init_cond_norm": [],
            "max_init_cond_norm": [],
            "L1_boundary_cond_norm": [],
            "L2_boundary_cond_norm": [],
            "max_boundary_cond_norm": [],
            "cost_functional_loss": [],
        }
        save_weighted_norms_calculated_after_training.training_number_of_penultimate_training_instance = 0

    dictionary_with_weighted_norms_history = data["dictionary_with_weighted_norms_history"]
    dictionary_with_weighted_norms_after_last_training = data["dictionary_with_norms_after_training"]

    training_number_of_last_training_instance = training_setting_dictionary["training_number"]
    all_cost_functional_weights_were_tried_and_new_training_runs_with_new_training_number_have_started = \
        save_weighted_norms_calculated_after_training.training_number_of_penultimate_training_instance != training_number_of_last_training_instance

    if all_cost_functional_weights_were_tried_and_new_training_runs_with_new_training_number_have_started:
        for norm_name in dictionary_with_weighted_norms_history.keys():
            dictionary_with_weighted_norms_history[norm_name] = []


    for norm_name, norm_value in dictionary_with_weighted_norms_after_last_training.items():
        dictionary_with_weighted_norms_history[norm_name].append(norm_value)

    save_weighted_norms_calculated_after_training.training_number_of_penultimate_training_instance = training_number_of_last_training_instance
    return


def save_3D_plot_data_in_vtm_format_as_multiblock_with_time_information(
    X_plot,
    Y_plot,
    T_plot,
    tensor_with_data_to_save,
    file_and_data_name,
    file_save_dir,
):
    x_step = X_plot[1, 0, 0] - X_plot[0, 0, 0]
    y_step = Y_plot[0, 1, 0] - Y_plot[0, 0, 0]
    t_step = T_plot[0, 0, 1] - T_plot[0, 0, 0]

    number_of_time_plots = T_plot.shape[2]

    multiblock = vtk.vtkMultiBlockDataSet()
    multiblock.SetNumberOfBlocks(number_of_time_plots)

    writer = vtk.vtkXMLMultiBlockDataWriter()
    writer.SetDataModeToAscii()
    writer.SetFileName(os.path.join(file_save_dir, f"{file_and_data_name}.vtm"))

    for time_index in range(number_of_time_plots):
        slice_of_tensor_with_data_to_save = tensor_with_data_to_save[:, :, time_index:time_index+1]

        image_data = vtk.vtkImageData()
        image_data.SetDimensions(X_plot.shape[0], Y_plot.shape[1], 1)
        image_data.SetSpacing(x_step, y_step, t_step)

        vtk_array = vtk.util.numpy_support.numpy_to_vtk(slice_of_tensor_with_data_to_save.ravel("F"), deep=True)
        vtk_array.SetName(f"{file_and_data_name}")

        image_data.GetPointData().SetScalars(vtk_array)

        multiblock.SetBlock(time_index, image_data)

    time_values = T_plot[0, 0, :]
    time_array = vtk.vtkDoubleArray()
    time_array.SetName("TimeValues")
    time_array.SetNumberOfTuples(number_of_time_plots)
    for i, t in enumerate(time_values):
        time_array.SetValue(i, t)

    multiblock.GetFieldData().AddArray(time_array)

    writer.SetInputData(multiblock)
    writer.Write()


def save_3D_plot_data_in_vti_format(
    X_plot,
    Y_plot,
    T_plot,
    tensor_with_data_to_save,
    file_and_data_name,
    file_save_dir,
):
    x_step = X_plot[1, 0, 0] - X_plot[0, 0, 0]
    y_step = Y_plot[0, 1, 0] - Y_plot[0, 0, 0]
    t_step = T_plot[0, 0, 1] - T_plot[0, 0, 0]

    image_data = vtk.vtkImageData()
    image_data.SetDimensions(X_plot.shape[0], Y_plot.shape[1], T_plot.shape[2])
    image_data.SetSpacing(x_step, y_step, t_step)

    vtk_array = vtk.util.numpy_support.numpy_to_vtk(tensor_with_data_to_save.ravel("F"), deep=True)
    vtk_array.SetName(file_and_data_name)

    image_data.GetPointData().SetScalars(vtk_array)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetDataModeToAscii()
    writer.SetFileName(os.path.join(file_save_dir, f"{file_and_data_name}.vti"))
    writer.SetInputData(image_data)
    writer.Write()


def save_plot_data_in_vt_format(training_setting_dictionary: dict, data: dict):
    X_plot = data["X_plot"]
    Y_plot = data["Y_plot"]
    T_plot = data["T_plot"]
    u_nn = data["u_nn"]
    q_nn = data["q_nn"]
    desired_function_at_final_time = data["desired_function_at_final_time"]
    circle_radius = data["circle_radius"]
    circle_center_in_xy = data["circle_center_in_xy"]
    save_dir = training_setting_dictionary["save_dir_for_current_specific_training_setting"]


    x_eval = tf.reshape(X_plot, shape=[-1])
    y_eval = tf.reshape(Y_plot, shape=[-1])
    t_eval = tf.reshape(T_plot, shape=[-1])

    u_nn_values = u_nn(tf.stack([t_eval, x_eval, y_eval], axis=-1), training=False)
    q_nn_values = q_nn(tf.stack([t_eval, x_eval, y_eval], axis=-1), training=False)

    u_nn_plot = tf.reshape(u_nn_values, shape=X_plot.shape)
    q_nn_plot = tf.reshape(q_nn_values, shape=X_plot.shape)


    x_eval_on_unit_circle = (x_eval - circle_center_in_xy[0]) / circle_radius
    y_eval_on_unit_circle = (y_eval - circle_center_in_xy[1]) / circle_radius

    radial_distance = tf.math.sqrt(x_eval_on_unit_circle**2 + y_eval_on_unit_circle**2)
    polar_angle = tf.math.atan2(y_eval_on_unit_circle, x_eval_on_unit_circle)

    desired_function_at_final_time_values = desired_function_at_final_time(radial_distance, polar_angle)
    desired_function_at_final_time_plot = tf.reshape(desired_function_at_final_time_values, shape=X_plot.shape)


    mask_points_outside_circle_domain = (X_plot - circle_center_in_xy[0])**2 + (Y_plot - circle_center_in_xy[1])**2 <= circle_radius**2
    mask_points_outside_circle_domain = mask_points_outside_circle_domain.numpy()

    absolute_u_nn_error = u_nn_plot - desired_function_at_final_time_plot
    absolute_u_nn_error_plot = np.where(mask_points_outside_circle_domain, absolute_u_nn_error.numpy(), np.nan)

    relative_u_nn_error = (u_nn_plot - desired_function_at_final_time_plot) / tf.math.reduce_max(desired_function_at_final_time_plot)
    relative_u_nn_error_plot = np.where(mask_points_outside_circle_domain, relative_u_nn_error.numpy(), np.nan)

    X_plot = X_plot.numpy()
    Y_plot = Y_plot.numpy()
    T_plot = T_plot.numpy()
    u_nn_plot = np.where(mask_points_outside_circle_domain, u_nn_plot.numpy(), np.nan)
    q_nn_plot = np.where(mask_points_outside_circle_domain, q_nn_plot.numpy(), np.nan)
    desired_function_at_final_time_plot = np.where(mask_points_outside_circle_domain, desired_function_at_final_time_plot.numpy(), np.nan)

    # print("X_plot: ", X_plot)
    # print("Y_plot: ", Y_plot)
    # print("T_plot: ", T_plot)
    # print("u_nn_plot: ", u_nn_plot)
    # print("q_nn_plot: ", q_nn_plot)
    # print("desired_function_at_final_time_plot: ", desired_function_at_final_time_plot)

    # print("X shape:", X_plot.shape)
    # print("Y shape:", Y_plot.shape)
    # print("T shape:", T_plot.shape)
    # print("u_nn_plot shape:", u_nn_plot.shape)
    # print("q_nn_plot shape:", q_nn_plot.shape)
    # print("desired_function_at_final_time_plot shape:", desired_function_at_final_time_plot.shape)
    # print("mask_points_outside_circle_domain shape:", mask_points_outside_circle_domain.shape)
    # print("absolute_u_nn_error_plot shape:", absolute_u_nn_error_plot.shape)
    # print("relative_u_nn_error_plot shape:", relative_u_nn_error_plot.shape)
    # print("VTK Dimensions:", image_data.GetDimensions())

    save_3D_plot_data_in_vti_format(X_plot, Y_plot, T_plot, u_nn_plot, "u_nn", save_dir)
    save_3D_plot_data_in_vti_format(X_plot, Y_plot, T_plot, q_nn_plot, "q_nn", save_dir)
    save_3D_plot_data_in_vti_format(X_plot, Y_plot, T_plot, desired_function_at_final_time_plot, "fitted function", save_dir)
    save_3D_plot_data_in_vti_format(X_plot, Y_plot, T_plot, absolute_u_nn_error_plot, "absolute error (u_nn - u_f)", save_dir)
    save_3D_plot_data_in_vti_format(X_plot, Y_plot, T_plot, relative_u_nn_error_plot, "relative error ((u_nn - u_f)÷max(u_f))", save_dir)

    save_3D_plot_data_in_vtm_format_as_multiblock_with_time_information(X_plot, Y_plot, T_plot, u_nn_plot, "u_nn", save_dir)
    save_3D_plot_data_in_vtm_format_as_multiblock_with_time_information(X_plot, Y_plot, T_plot, q_nn_plot, "q_nn", save_dir)
    save_3D_plot_data_in_vtm_format_as_multiblock_with_time_information(X_plot, Y_plot, T_plot, desired_function_at_final_time_plot,
                                                                        "fitted function", save_dir)
    save_3D_plot_data_in_vtm_format_as_multiblock_with_time_information(X_plot, Y_plot, T_plot, absolute_u_nn_error_plot,
                                                                        "absolute error (u_nn - u_f)", save_dir)
    save_3D_plot_data_in_vtm_format_as_multiblock_with_time_information(X_plot, Y_plot, T_plot, relative_u_nn_error_plot,
                                                                        "relative error ((u_nn - u_f)÷max(u_f))", save_dir)


def process_results(training_setting_dictionary: dict, data: dict):
    save_weighted_norms_calculated_after_training(training_setting_dictionary, data)


def delete_model(training_setting_dictionary: dict, data: dict):
    del data["u_nn"]
    del data["q_nn"]
