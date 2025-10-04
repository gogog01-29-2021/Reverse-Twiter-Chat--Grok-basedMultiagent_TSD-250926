from . import *
from ..general import *
import tensorflow as tf
import gc
from ..general import L2_norm
# from .data_processing import function_to_convert_custom_parameters, function_to_convert_activation_functions_and_num_of_training_points


def call_main_training_script_for_circle_domain(
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

        num_of_training_runs_for_one_setting = training_setting_dictionary["num_of_training_runs_for_one_setting"]

        for training_number in range(num_of_training_runs_for_one_setting):
            for cost_functional_loss_coef in training_setting_dictionary["iterable_with_cost_functional_loss_coefs"]:

                training_setting_dictionary["training_number"] = training_number
                training_setting_dictionary["cost_functional_loss_coef"] = cost_functional_loss_coef

                create_save_dir_for_current_specific_training_setting(training_setting_dictionary, data)

                train_model_on_circle_domain(training_setting_dictionary, data)

                append_this_trainings_results_into_csv_file_with_individual_trainings_results(training_setting_dictionary, data)

                append_this_trainings_results_into_main_csv_file(training_setting_dictionary, data)

                process_results(training_setting_dictionary, data)

                if training_setting_dictionary["save_weights"]:
                    save_model(training_setting_dictionary, data)

                if training_setting_dictionary["save_plots"]:
                    plot_and_save_figs(training_setting_dictionary, data)

                save_plot_data_in_vt_format(training_setting_dictionary, data)

                delete_model(training_setting_dictionary, data)
            plot_and_save_norm_values_after_all_cost_functional_coeficients(training_setting_dictionary, data)
        tf.keras.backend.clear_session()
        gc.collect()


def train_different_activation_functions():
    training_configuration = {
        "main_save_dir":                            os.path.join(".", "training runs", "activation functions"),
        "override_main_csv_file":                   True,
        "save_weights":                             True,
        "save_plots":                               True,
        "num_of_training_runs_for_one_setting":     1,
        "shuffle_training_points_each_epoch":       True,
        "u_nn_kernel_initializer":                  tf.keras.initializers.glorot_uniform,
        "q_nn_kernel_initializer":                  tf.keras.initializers.glorot_uniform,
        "loss_fn":                                  L2_norm,
        "optimizer":                                tf.keras.optimizers.Adam,
        "learning_rate":                            (0.0009, 0.0009, 0.0009),
        "initial_epoch":                            0,
        "num_of_epochs":                            (8000, 8000, 8000),
        "write_loss_values_every_x_epochs":         1000,
        "iterable_with_cost_functional_loss_coefs": [[1.0]],
        "init_cond_loss_coef":                      (1.0, 1.0, 1.0),
        "boundary_loss_coef":                       (1.0, 1.0, 1.0),
        "num_of_t_training_points":                 (20, 20+1, 20),
        "num_of_grid_x_training_points":            (20, 20+1, 20),
        "num_of_grid_y_training_points":            (20, 20+1, 20),
        "num_xy_training_points_on_boundary":       (100, 100, 100),
        "u_nn_num_of_layers":                       (3, 3+1, 3),
        "u_nn_num_of_neurons_in_layer":             (30, 30+1, 30),
        "q_nn_num_of_layers":                       (3, 3+1, 3),
        "q_nn_num_of_neurons_in_layer":             (30, 30+1, 30),
        "num_of_batches":                           (2, 2, 2),
        "u_nn_activation_functions_list":           ["tanh", "sigmoid", "relu"],
        "u_nn_repeat_activation_functions":         False,
        "q_nn_activation_functions_list":           ["tanh", "sigmoid", "relu"],
        "q_nn_repeat_activation_functions":         False,
        "q_nn_final_activation_function":           ["relu"]
    }

    parameter_names_for_custom_conversion = [
        "u_nn_activation_functions_list",
        "u_nn_repeat_activation_functions",
        "q_nn_activation_functions_list",
        "q_nn_repeat_activation_functions",
    ]

    def equation_rhs_function(t, x, y, u, heat_coef):
        return 0

    def initial_condition_function(x, y, u, heat_coef):
        return 0

    def Zernike_polynomial(radial_distance, polar_angle):
        return radial_distance**2 * tf.math.sin(2*polar_angle)

    data = {
        "heat_coef": 1.0,
        "alpha":     0.4,
        "t_start":   0.0,
        "t_stop":    1.0,
        "circle_radius":            1.0,
        "circle_center_in_xy":      [0.0, 0.0],
        "num_of_time_plots":        10,
        "num_of_x_plot_points":     200,
        "num_of_y_plot_points":     200,
        "equation_rhs_function": equation_rhs_function,
        "initial_condition_function": initial_condition_function,
        "desired_function_at_final_time": Zernike_polynomial,
    }

    call_main_training_script_for_circle_domain(
        data,
        training_configuration,
        parameter_names_for_custom_conversion,
        function_to_convert_custom_parameters,
    )


def train_different_and_higher_number_of_training_points():
    training_configuration = {
        "main_save_dir":                            os.path.join(".", "training runs", "number of training points"),
        "override_main_csv_file":                   True,
        "save_weights":                             True,
        "save_plots":                               True,
        "num_of_training_runs_for_one_setting":     1,
        "shuffle_training_points_each_epoch":       True,
        "u_nn_kernel_initializer":                  tf.keras.initializers.glorot_uniform,
        "q_nn_kernel_initializer":                  tf.keras.initializers.glorot_uniform,
        "loss_fn":                                  L2_norm,
        "optimizer":                                tf.keras.optimizers.Adam,
        "learning_rate":                            (0.00075, 0.00075, 0.00075),
        "initial_epoch":                            0,
        "num_of_epochs":                            (10000, 10000, 10000),
        "write_loss_values_every_x_epochs":         1000,
        "iterable_with_cost_functional_loss_coefs": [[0.1, 1.0, 10.0, 100.0]],
        "init_cond_loss_coef":                      (1.0, 1.0, 1.0),
        "boundary_loss_coef":                       (1.0, 1.0, 1.0),
        "num_of_t_training_points":                 (75, 100+1, 25),
        "num_of_grid_x_training_points":            (75, 100+1, 25),
        "num_of_grid_y_training_points":            (75, 100+1, 25),
        "num_xy_training_points_on_boundary":       (1000, 1000, 1000),
        "u_nn_num_of_layers":                       (12, 12+1, 3),
        "u_nn_num_of_neurons_in_layer":             (50, 50+1, 50),
        "q_nn_num_of_layers":                       (12, 12+1, 3),
        "q_nn_num_of_neurons_in_layer":             (50, 50+1, 50),
        "num_of_batches":                           (6, 6, 6),
        "u_nn_activation_functions_list":           ["tanh"],
        "u_nn_repeat_activation_functions":         True,
        "q_nn_activation_functions_list":           ["tanh"],
        "q_nn_repeat_activation_functions":         True,
        "q_nn_final_activation_function":           ["relu"]
    }

    parameter_names_for_custom_conversion = [
        "u_nn_activation_functions_list",
        "u_nn_repeat_activation_functions",
        "q_nn_activation_functions_list",
        "q_nn_repeat_activation_functions",
    ]

    def equation_rhs_function(t, x, y, u, heat_coef):
        return 0

    def initial_condition_function(x, y, u, heat_coef):
        return 0

    def Zernike_polynomial(radial_distance, polar_angle):
        return radial_distance**2 * tf.math.sin(2*polar_angle)

    data = {
        "heat_coef": 1.0,
        "alpha":     0.4,
        "t_start":   0.0,
        "t_stop":    1.0,
        "circle_radius":            1.0,
        "circle_center_in_xy":      [0.0, 0.0],
        "num_of_time_plots":        10,
        "num_of_x_plot_points":     200,
        "num_of_y_plot_points":     200,
        "equation_rhs_function": equation_rhs_function,
        "initial_condition_function": initial_condition_function,
        "desired_function_at_final_time": Zernike_polynomial,
    }

    call_main_training_script_for_circle_domain(
        data,
        training_configuration,
        parameter_names_for_custom_conversion,
        function_to_convert_custom_parameters,
    )




def train_different_number_of_training_points_relative_to_network_size():
    put_num_of_t_training_points_as_num_of_x_and_y_training_points = True
    training_configuration = {
        "main_save_dir":                            os.path.join(".", "training runs", "number of training points relative to network size"),
        "override_main_csv_file":                   True,
        "save_weights":                             True,
        "save_plots":                               True,
        "num_of_training_runs_for_one_setting":     1,
        "shuffle_training_points_each_epoch":       True,
        "u_nn_kernel_initializer":                  tf.keras.initializers.glorot_uniform,
        "q_nn_kernel_initializer":                  tf.keras.initializers.glorot_uniform,
        "loss_fn":                                  L2_norm,
        "optimizer":                                tf.keras.optimizers.Adam,
        "learning_rate":                            (0.0009, 0.0009, 0.0009),
        "initial_epoch":                            0,
        "num_of_epochs":                            (10000, 10000, 10000),
        "write_loss_values_every_x_epochs":         1000,
        "iterable_with_cost_functional_loss_coefs": [[0.01, 1.0, 100.0]],
        "init_cond_loss_coef":                      (1.0, 1.0, 1.0),
        "boundary_loss_coef":                       (1.0, 1.0, 1.0),
        "num_of_t_training_points":                 [11, 24, 51],
        "num_of_grid_x_training_points":            [11, 24, 51],
        "num_of_grid_y_training_points":            [11, 24, 51],
        ("put_num_of_t_training_points_as_"
         "num_of_x_and_y_training_points"):         put_num_of_t_training_points_as_num_of_x_and_y_training_points,
        "num_xy_training_points_on_boundary":       [11**2, 24**2, 51**2],
        "u_nn_num_of_layers":                       (6, 6+1, 3),
        "u_nn_num_of_neurons_in_layer":             (50, 50+1, 50),  # 13 001 weights
        "q_nn_num_of_layers":                       (6, 6+1, 3),
        "q_nn_num_of_neurons_in_layer":             (50, 50+1, 50),
        "num_of_batches":                           (2, 2, 2),
        "u_nn_activation_functions_list":           ["tanh"],
        "u_nn_repeat_activation_functions":         True,
        "q_nn_activation_functions_list":           ["tanh"],
        "q_nn_repeat_activation_functions":         True,
        "q_nn_final_activation_function":           ["relu"]
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

    def equation_rhs_function(t, x, y, u, heat_coef):
        return 0

    def initial_condition_function(x, y, u, heat_coef):
        return 0

    def Zernike_polynomial(radial_distance, polar_angle):
        return radial_distance**2 * tf.math.sin(2*polar_angle)

    data = {
        "heat_coef": 1.0,
        "alpha":     0.4,
        "t_start":   0.0,
        "t_stop":    1.0,
        "circle_radius":            1.0,
        "circle_center_in_xy":      [0.0, 0.0],
        "num_of_time_plots":        10,
        "num_of_x_plot_points":     200,
        "num_of_y_plot_points":     200,
        "equation_rhs_function": equation_rhs_function,
        "initial_condition_function": initial_condition_function,
        "desired_function_at_final_time": Zernike_polynomial,
    }

    call_main_training_script_for_circle_domain(
        data,
        training_configuration,
        parameter_names_for_custom_conversion,
        function_to_convert_activation_functions_and_num_of_training_points,
    )


def train_different_number_of_layers():
    training_configuration = {
        "main_save_dir":                            os.path.join(".", "training runs", "number of layers | for bigger network and more training points"),
        "override_main_csv_file":                   True,
        "save_weights":                             True,
        "save_plots":                               True,
        "num_of_training_runs_for_one_setting":     1,
        "shuffle_training_points_each_epoch":       True,
        "u_nn_kernel_initializer":                  tf.keras.initializers.glorot_uniform,
        "q_nn_kernel_initializer":                  tf.keras.initializers.glorot_uniform,
        "loss_fn":                                  L2_norm,
        "optimizer":                                tf.keras.optimizers.Adam,
        "learning_rate":                            (0.00075, 0.00075, 0.00075),
        "initial_epoch":                            0,
        "num_of_epochs":                            (10000, 10000, 10000),
        "write_loss_values_every_x_epochs":         1000,
        "iterable_with_cost_functional_loss_coefs": [[0.1, 1.0, 10.0]],
        "init_cond_loss_coef":                      (1.0, 1.0, 1.0),
        "boundary_loss_coef":                       (1.0, 1.0, 1.0),
        "num_of_t_training_points":                 (50, 50+1, 50),
        "num_of_grid_x_training_points":            (50, 50+1, 50),
        "num_of_grid_y_training_points":            (50, 50+1, 50),
        "num_xy_training_points_on_boundary":       (1000, 1000, 1000),
        "u_nn_num_of_layers":                       (6, 9+1, 3),
        "u_nn_num_of_neurons_in_layer":             (50, 50+1, 50),
        "q_nn_num_of_layers":                       (6, 9+1, 3),
        "q_nn_num_of_neurons_in_layer":             (50, 50+1, 50),
        "num_of_batches":                           (4, 4, 4),
        "u_nn_activation_functions_list":           ["tanh"],
        "u_nn_repeat_activation_functions":         True,
        "q_nn_activation_functions_list":           ["tanh"],
        "q_nn_repeat_activation_functions":         True,
        "q_nn_final_activation_function":           ["relu"]
    }

    parameter_names_for_custom_conversion = [
        "u_nn_activation_functions_list",
        "u_nn_repeat_activation_functions",
        "q_nn_activation_functions_list",
        "q_nn_repeat_activation_functions",
    ]

    def equation_rhs_function(t, x, y, u, heat_coef):
        return 0

    def initial_condition_function(x, y, u, heat_coef):
        return 0

    def Zernike_polynomial(radial_distance, polar_angle):
        return radial_distance**2 * tf.math.sin(2*polar_angle)

    data = {
        "heat_coef": 1.0,
        "alpha":     0.4,
        "t_start":   0.0,
        "t_stop":    1.0,
        "circle_radius":            1.0,
        "circle_center_in_xy":      [0.0, 0.0],
        "num_of_time_plots":        10,
        "num_of_x_plot_points":     200,
        "num_of_y_plot_points":     200,
        "equation_rhs_function": equation_rhs_function,
        "initial_condition_function": initial_condition_function,
        "desired_function_at_final_time": Zernike_polynomial,
    }

    call_main_training_script_for_circle_domain(
        data,
        training_configuration,
        parameter_names_for_custom_conversion,
        function_to_convert_custom_parameters,
    )
