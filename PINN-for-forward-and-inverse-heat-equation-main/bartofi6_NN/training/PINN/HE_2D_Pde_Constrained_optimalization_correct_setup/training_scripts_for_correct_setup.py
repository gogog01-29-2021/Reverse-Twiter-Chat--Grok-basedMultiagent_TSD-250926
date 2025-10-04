import tensorflow as tf
import gc
import os
import math
from ..general import L2_norm, \
                      create_main_save_dir_and_main_csv_file, \
                      training_settings_generator
from .data_processing import setup_data, \
                             function_to_convert_activation_functions_in_custom_way, \
                             function_to_convert_activation_functions_and_num_of_training_points, \
                             process_results, \
                             delete_model, \
                             save_plot_data_in_vt_format
from .file_and_folder_management import write_main_csv_file_header_row, \
                                        create_save_dir_and_csv_file_for_current_general_training_setting, \
                                        create_txt_file_with_parameters_for_current_training_setting, \
                                        create_save_dir_for_current_specific_training_setting, \
                                        append_this_trainings_results_into_csv_file_with_results_for_single_training_setting, \
                                        append_this_trainings_results_into_main_csv_file
from .plotting import plot_and_save_figs, \
                      plot_and_save_norm_values_after_all_cost_functional_coeficients
from .training_functions import train_model_for_correct_setup, \
                                save_model


def call_main_training_script_for_correct_setup(
    data,
    training_configuration,
    parameter_names_for_custom_conversion,
    function_to_convert_custom_parameters,
):
    setup_data(data, training_configuration)

    create_main_save_dir_and_main_csv_file(training_configuration, data)

    if training_configuration["override_main_csv_file"]:
        write_main_csv_file_header_row(training_configuration, data)


    for training_setting_dictionary in training_settings_generator(training_configuration,
                                                                   parameter_names_for_custom_conversion,
                                                                   function_to_convert_custom_parameters):

        create_save_dir_and_csv_file_for_current_general_training_setting(training_setting_dictionary, data)

        create_txt_file_with_parameters_for_current_training_setting(training_setting_dictionary, data)

        num_of_training_runs_for_one_setting = training_setting_dictionary["num_of_training_runs_for_one_setting"]

        for training_number in range(num_of_training_runs_for_one_setting):
            for cost_functional_loss_coef in training_setting_dictionary["iterable_with_cost_functional_loss_coefs"]:

                training_setting_dictionary["training_number"] = training_number
                training_setting_dictionary["cost_functional_loss_coef"] = cost_functional_loss_coef

                create_save_dir_for_current_specific_training_setting(training_setting_dictionary, data)

                train_model_for_correct_setup(training_setting_dictionary, data)

                append_this_trainings_results_into_csv_file_with_results_for_single_training_setting(training_setting_dictionary, data)

                append_this_trainings_results_into_main_csv_file(training_setting_dictionary, data)

                process_results(training_setting_dictionary, data)

                if training_setting_dictionary["save_weights"]:
                    save_model(training_setting_dictionary, data)

                if training_setting_dictionary["save_plots"]:
                    plot_and_save_figs(training_setting_dictionary, data)

                # save_plot_data_in_vt_format(training_setting_dictionary, data)

                delete_model(training_setting_dictionary, data)
            plot_and_save_norm_values_after_all_cost_functional_coeficients(training_setting_dictionary, data)
        tf.keras.backend.clear_session()
        gc.collect()


def dimensionless_equation_rhs_function(t, x, y, u, heat_coef):
    return 0


def dimensionless_initial_condition_function(x, y, u, heat_coef):
    return 0


Zernike_polynomial_scale = 50


def dimensionless_Zernike_polynomial(radial_distance, polar_angle):
    return radial_distance**2 * tf.math.sin(2*polar_angle)


#   These coeficients rescale the units, so for example if the spacial_scale_coef is 10^-3,
# then the spacial unit will be mm instead of a m.
L = 0.00014
final_time = 0.1
spacial_scale_coef = L
time_scale_coef = final_time
polynomial_scale_coef = Zernike_polynomial_scale
u_scale_coef = polynomial_scale_coef
RHS_scale_coef = u_scale_coef / time_scale_coef
q_scale_coef = u_scale_coef / spacial_scale_coef
initial_condition_scale_coef = u_scale_coef
cost_functional_scale_coef = polynomial_scale_coef**2 * spacial_scale_coef**2
# Dimensionless variables:
x_start = -L / spacial_scale_coef
x_stop  =  L / spacial_scale_coef
y_start = -L / spacial_scale_coef
y_stop  =  L / spacial_scale_coef
circle_radius = 0.000125 / spacial_scale_coef
circle_center_x = 0.0 / spacial_scale_coef
circle_center_y = 0.0 / spacial_scale_coef
t_start =  0.0 / time_scale_coef
t_stop  =  final_time / time_scale_coef
K = 9.4e-8
heat_coef = time_scale_coef * K / (spacial_scale_coef**2 * u_scale_coef)

xy_domain_area = (x_stop - x_start) * (y_stop - y_start)
circle_domain_area = math.pi * circle_radius**2
base_functional_loss_coef = 1.0


data = {
    "heat_coef": heat_coef,
    "alpha":     1.0,  # Currently redundant, as the boundary integral regularization is turned off.
    "x_start":   x_start,
    "x_stop":    x_stop,
    "y_start":   y_start,
    "y_stop":    y_stop,
    "t_start":   t_start,
    "t_stop":    t_stop,
    "circle_radius":            circle_radius,
    "circle_center_in_xy":      [circle_center_x, circle_center_y],
    "num_of_time_plots":        10,
    "num_of_x_plot_points":     200,
    "num_of_y_plot_points":     200,
    "equation_rhs_function": dimensionless_equation_rhs_function,
    "initial_condition_function": dimensionless_initial_condition_function,
    "desired_function_at_final_time": dimensionless_Zernike_polynomial,
    "spacial_scale_coef": spacial_scale_coef,
    "time_scale_coef": time_scale_coef,
    "polynomial_scale_coef": polynomial_scale_coef,
    "u_scale_coef": u_scale_coef,
    "RHS_scale_coef": RHS_scale_coef,
    "q_scale_coef": q_scale_coef,
    "initial_condition_scale_coef": initial_condition_scale_coef,
    "cost_functional_scale_coef": cost_functional_scale_coef,
}


def train_different_activation_functions():
    training_configuration = {
        "main_save_dir":                            os.path.join(".", "activation functions"),
        "override_main_csv_file":                   True,
        "save_weights":                             True,
        "save_plots":                               True,
        "num_of_training_runs_for_one_setting":     1,
        "shuffle_training_points_each_epoch":       True,
        "u_nn_kernel_initializer":                  tf.keras.initializers.glorot_uniform,
        "q_nn_kernel_initializer":                  tf.keras.initializers.glorot_uniform,
        "loss_fn":                                  L2_norm,
        "dtype":                                    tf.float32,
        "optimizer":                                tf.keras.optimizers.Adam,
        "learning_rate":                            0.001,
        "initial_epoch":                            0,
        "num_of_epochs":                            10000,
        "write_loss_values_every_x_epochs":         1000,
        "iterable_with_cost_functional_loss_coefs": [[base_functional_loss_coef * 1.0]],
        "init_cond_loss_coef":                      (1.0, 1.0, 1.0),
        "boundary_loss_coef":                       (1.0, 1.0, 1.0),
        "q_nn_cooling_penalty_weight":              (1.0, 1.0, 1.0),
        "num_of_t_training_points":                 (14, 14+1, 14),
        "num_of_grid_x_training_points":            (14, 14+1, 14),
        "num_of_grid_y_training_points":            (14, 14+1, 14),
        "num_of_x_points_for_integral_evaluation":  100,
        "num_of_y_points_for_integral_evaluation":  100,
        "u_nn_num_of_layers":                       (2, 2+1, 2),
        "u_nn_num_of_neurons_in_layer":             (50, 50+1, 50),
        "q_nn_num_of_layers":                       (2, 2+1, 2),
        "q_nn_num_of_neurons_in_layer":             (50, 50+1, 50),
        "num_of_batches":                           (1, 1, 1),
        "q_nn_activation_functions_list":           ["tanh", "tanh"],
        "q_nn_repeat_activation_functions":         True,
        "u_nn_activation_functions_list":           ["tanh", "sigmoid", "relu"],
        "u_nn_repeat_activation_functions":         False,
        "q_nn_final_activation_function":           None
    }

    parameter_names_for_custom_conversion = [
        "u_nn_activation_functions_list",
        "u_nn_repeat_activation_functions",
        "q_nn_activation_functions_list",
        "q_nn_repeat_activation_functions",
    ]

    call_main_training_script_for_correct_setup(
        data,
        training_configuration,
        parameter_names_for_custom_conversion,
        function_to_convert_activation_functions_in_custom_way,
    )

    training_configuration["override_main_csv_file"] = False

    training_configuration["q_nn_activation_functions_list"] = ["tanh", "sigmoid"]
    call_main_training_script_for_correct_setup(
        data,
        training_configuration,
        parameter_names_for_custom_conversion,
        function_to_convert_activation_functions_in_custom_way,
    )

    training_configuration["q_nn_activation_functions_list"] = ["tanh", "relu"]
    call_main_training_script_for_correct_setup(
        data,
        training_configuration,
        parameter_names_for_custom_conversion,
        function_to_convert_activation_functions_in_custom_way,
    )

    training_configuration["q_nn_activation_functions_list"] = ["sigmoid", "tanh"]
    call_main_training_script_for_correct_setup(
        data,
        training_configuration,
        parameter_names_for_custom_conversion,
        function_to_convert_activation_functions_in_custom_way,
    )

    training_configuration["q_nn_activation_functions_list"] = ["sigmoid", "sigmoid"]
    call_main_training_script_for_correct_setup(
        data,
        training_configuration,
        parameter_names_for_custom_conversion,
        function_to_convert_activation_functions_in_custom_way,
    )

    training_configuration["q_nn_activation_functions_list"] = ["sigmoid", "relu"]
    call_main_training_script_for_correct_setup(
        data,
        training_configuration,
        parameter_names_for_custom_conversion,
        function_to_convert_activation_functions_in_custom_way,
    )

    training_configuration["q_nn_activation_functions_list"] = ["relu", "tanh"]
    call_main_training_script_for_correct_setup(
        data,
        training_configuration,
        parameter_names_for_custom_conversion,
        function_to_convert_activation_functions_in_custom_way,
    )

    training_configuration["q_nn_activation_functions_list"] = ["relu", "sigmoid"]
    call_main_training_script_for_correct_setup(
        data,
        training_configuration,
        parameter_names_for_custom_conversion,
        function_to_convert_activation_functions_in_custom_way,
    )

    training_configuration["q_nn_activation_functions_list"] = ["relu", "relu"]
    call_main_training_script_for_correct_setup(
        data,
        training_configuration,
        parameter_names_for_custom_conversion,
        function_to_convert_activation_functions_in_custom_way,
    )


def train_different_and_higher_number_of_training_points():
    training_configuration = {
        "main_save_dir":                            os.path.join(".", "number of training points"),
        "override_main_csv_file":                   True,
        "save_weights":                             True,
        "save_plots":                               True,
        "num_of_training_runs_for_one_setting":     1,
        "shuffle_training_points_each_epoch":       True,
        "u_nn_kernel_initializer":                  tf.keras.initializers.glorot_uniform,
        "q_nn_kernel_initializer":                  tf.keras.initializers.glorot_uniform,
        "loss_fn":                                  L2_norm,
        "dtype":                                    tf.float32,
        "optimizer":                                tf.keras.optimizers.Adam,
        "learning_rate":                            (0.00075, 0.00075, 0.00075),
        "initial_epoch":                            0,
        "num_of_epochs":                            (10000, 10000, 10000),
        "write_loss_values_every_x_epochs":         1000,
        "iterable_with_cost_functional_loss_coefs": [[base_functional_loss_coef * 0.1,
                                                      base_functional_loss_coef * 1.0,
                                                      base_functional_loss_coef * 10.0,
                                                      base_functional_loss_coef * 100.0]],
        "init_cond_loss_coef":                      (1.0, 1.0, 1.0),
        "boundary_loss_coef":                       (1.0, 1.0, 1.0),
        "q_nn_cooling_penalty_weight":              (1.0, 1.0, 1.0),
        "num_of_t_training_points":                 (75, 100+1, 25),
        "num_of_grid_x_training_points":            (75, 100+1, 25),
        "num_of_grid_y_training_points":            (75, 100+1, 25),
        "num_of_x_points_for_integral_evaluation":  100,
        "num_of_y_points_for_integral_evaluation":  100,
        "u_nn_num_of_layers":                       (12, 12+1, 3),
        "u_nn_num_of_neurons_in_layer":             (50, 50+1, 50),
        "q_nn_num_of_layers":                       (12, 12+1, 3),
        "q_nn_num_of_neurons_in_layer":             (50, 50+1, 50),
        "num_of_batches":                           (6, 6, 6),
        "u_nn_activation_functions_list":           ["tanh"],
        "u_nn_repeat_activation_functions":         True,
        "q_nn_activation_functions_list":           ["tanh"],
        "q_nn_repeat_activation_functions":         True,
        "q_nn_final_activation_function":           None
    }

    parameter_names_for_custom_conversion = [
        "u_nn_activation_functions_list",
        "u_nn_repeat_activation_functions",
        "q_nn_activation_functions_list",
        "q_nn_repeat_activation_functions",
    ]

    call_main_training_script_for_correct_setup(
        data,
        training_configuration,
        parameter_names_for_custom_conversion,
        function_to_convert_activation_functions_in_custom_way,
    )


def train_different_number_of_training_points_relative_to_network_size():
    put_num_of_t_training_points_as_num_of_x_and_y_training_points = True
    training_configuration = {
        "main_save_dir":                            os.path.join(".", "number of training points relative to network size"),
        "override_main_csv_file":                   True,
        "save_weights":                             True,
        "save_plots":                               True,
        "num_of_training_runs_for_one_setting":     1,
        "shuffle_training_points_each_epoch":       True,
        "u_nn_kernel_initializer":                  tf.keras.initializers.glorot_uniform,
        "q_nn_kernel_initializer":                  tf.keras.initializers.glorot_uniform,
        "loss_fn":                                  L2_norm,
        "dtype":                                    tf.float32,
        "optimizer":                                tf.keras.optimizers.Adam,
        "learning_rate":                            (0.0009, 0.0009, 0.0009),
        "initial_epoch":                            0,
        "num_of_epochs":                            (10000, 10000, 10000),
        "write_loss_values_every_x_epochs":         1000,
        "iterable_with_cost_functional_loss_coefs": [[base_functional_loss_coef * 0.01,
                                                      base_functional_loss_coef * 1.0,
                                                      base_functional_loss_coef * 100.0]],
        "init_cond_loss_coef":                      (1.0, 1.0, 1.0),
        "boundary_loss_coef":                       (1.0, 1.0, 1.0),
        "q_nn_cooling_penalty_weight":              (1.0, 1.0, 1.0),
        "num_of_t_training_points":                 [11, 24, 51],
        "num_of_grid_x_training_points":            [11, 24, 51],
        "num_of_grid_y_training_points":            [11, 24, 51],
        ("put_num_of_t_training_points_as_"
         "num_of_x_and_y_training_points"):         put_num_of_t_training_points_as_num_of_x_and_y_training_points,
        "num_of_x_points_for_integral_evaluation":  100,
        "num_of_y_points_for_integral_evaluation":  100,
        "u_nn_num_of_layers":                       (6, 6+1, 3),
        "u_nn_num_of_neurons_in_layer":             (50, 50+1, 50),  # 13 001 weights
        "q_nn_num_of_layers":                       (6, 6+1, 3),
        "q_nn_num_of_neurons_in_layer":             (50, 50+1, 50),
        "num_of_batches":                           (1, 1, 1),
        "u_nn_activation_functions_list":           ["tanh"],
        "u_nn_repeat_activation_functions":         True,
        "q_nn_activation_functions_list":           ["tanh"],
        "q_nn_repeat_activation_functions":         True,
        "q_nn_final_activation_function":           None
    }

    parameter_names_for_custom_conversion = [
        "u_nn_activation_functions_list",
        "u_nn_repeat_activation_functions",
        "q_nn_activation_functions_list",
        "q_nn_repeat_activation_functions",
    ]
    if put_num_of_t_training_points_as_num_of_x_and_y_training_points:
        parameter_names_for_custom_conversion.extend(["num_of_t_training_points",
                                                      "num_of_grid_x_training_points",
                                                      "num_of_grid_y_training_points",
                                                      "put_num_of_t_training_points_as_num_of_x_and_y_training_points",])

    call_main_training_script_for_correct_setup(
        data,
        training_configuration,
        parameter_names_for_custom_conversion,
        function_to_convert_activation_functions_and_num_of_training_points,
    )


def train_different_number_of_layers():
    training_configuration = {
        "main_save_dir":                            os.path.join(".", "number of layers | for bigger network and more training points"),
        "override_main_csv_file":                   True,
        "save_weights":                             True,
        "save_plots":                               True,
        "num_of_training_runs_for_one_setting":     1,
        "shuffle_training_points_each_epoch":       True,
        "u_nn_kernel_initializer":                  tf.keras.initializers.glorot_uniform,
        "q_nn_kernel_initializer":                  tf.keras.initializers.glorot_uniform,
        "loss_fn":                                  L2_norm,
        "dtype":                                    tf.float32,
        "optimizer":                                tf.keras.optimizers.Adam,
        "learning_rate":                            (0.001, 0.001, 0.001),
        "initial_epoch":                            0,
        "num_of_epochs":                            (10000, 10000, 10000),
        "write_loss_values_every_x_epochs":         1000,
        "iterable_with_cost_functional_loss_coefs": [[base_functional_loss_coef * 0.1,
                                                      base_functional_loss_coef * 1.0,
                                                      base_functional_loss_coef * 10.0]],
        "init_cond_loss_coef":                      (1.0, 1.0, 1.0),
        "boundary_loss_coef":                       (1.0, 1.0, 1.0),
        "q_nn_cooling_penalty_weight":              (1.0, 1.0, 1.0),
        "num_of_t_training_points":                 (50, 50+1, 50),
        "num_of_grid_x_training_points":            (50, 50+1, 50),
        "num_of_grid_y_training_points":            (50, 50+1, 50),
        "num_of_x_points_for_integral_evaluation":  100,
        "num_of_y_points_for_integral_evaluation":  100,
        "u_nn_num_of_layers":                       (6, 9+1, 3),
        "u_nn_num_of_neurons_in_layer":             (50, 50+1, 50),
        "q_nn_num_of_layers":                       (6, 9+1, 3),
        "q_nn_num_of_neurons_in_layer":             (50, 50+1, 50),
        "num_of_batches":                           (4, 4, 4),
        "u_nn_activation_functions_list":           ["tanh"],
        "u_nn_repeat_activation_functions":         True,
        "q_nn_activation_functions_list":           ["tanh"],
        "q_nn_repeat_activation_functions":         True,
        "q_nn_final_activation_function":           None
    }

    parameter_names_for_custom_conversion = [
        "u_nn_activation_functions_list",
        "u_nn_repeat_activation_functions",
        "q_nn_activation_functions_list",
        "q_nn_repeat_activation_functions",
    ]

    call_main_training_script_for_correct_setup(
        data,
        training_configuration,
        parameter_names_for_custom_conversion,
        function_to_convert_activation_functions_in_custom_way,
    )
