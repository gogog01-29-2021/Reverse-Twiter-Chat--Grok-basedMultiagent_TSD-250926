import tensorflow as tf
from ...training.PINN import *
import os
import tensorflow as tf

__all__ = [
    "sample_run_with_eagerly_run_functions_to_check_tensor_shapes_are_correct"
]

def equation_rhs_function(t, x, y, u, heat_coef):
    return 0.0


def initial_condition_function(x, y, u, heat_coef):
    return 0.0


def desired_polynomial(x, y):
    return x*y


def Zernike_polynomial(radial_distance, polar_angle):
    return radial_distance**2 * tf.math.sin(2*polar_angle)


def sample_run_with_eagerly_run_functions_to_check_tensor_shapes_are_correct():
    tf.config.run_functions_eagerly(True)
    training_configuration = {
        "main_save_dir":                            os.path.join(".", "TEST runs", "TEST with eagerly run functions to check shape equality during calculations"),
        "override_main_csv_file":                   True,
        "save_weights":                             False,
        "save_plots":                               False,
        "num_of_training_runs_for_one_setting":     1,
        "shuffle_training_points_each_epoch":       True,
        "u_nn_kernel_initializer":                  tf.keras.initializers.glorot_uniform,
        "q_nn_kernel_initializer":                  tf.keras.initializers.glorot_uniform,
        "loss_fn":                                  L2_norm,
        "optimizer":                                tf.keras.optimizers.Adam,
        "learning_rate":                            (0.001, 0.001, 0.001),
        "initial_epoch":                            0,
        "num_of_epochs":                            (100, 100, 100),
        "write_loss_values_every_x_epochs":         10,
        "iterable_with_cost_functional_loss_coefs": [(0.05, 1.0, 15.0)],
        "init_cond_loss_coef":                      (1.0, 1.0, 1.0),
        "boundary_loss_coef":                       (1.0, 1.0, 1.0),
        "num_of_t_training_points":                 (20, 21, 20),
        "num_of_grid_x_training_points":            (20, 21, 20),
        "num_of_grid_y_training_points":            (20, 21, 20),
        "num_xy_training_points_on_boundary":       (20, 31, 20),
        "u_nn_num_of_layers":                       (1, 9+1, 3),
        "u_nn_num_of_neurons_in_layer":             (20, 30+1, 10),
        "q_nn_num_of_layers":                       (1, 9+1, 3),
        "q_nn_num_of_neurons_in_layer":             (20, 30+1, 10),
        "num_of_batches":                           (4, 2, 2),
        "u_nn_activation_functions_list":           ["relu"],
        "u_nn_repeat_activation_functions":         True,
        "q_nn_activation_functions_list":           ["relu"],
        "q_nn_repeat_activation_functions":         True,
        "q_nn_final_activation_function":           ["relu"]
    }
    parameter_names_for_custom_conversion = [
        "u_nn_activation_functions_list",
        "u_nn_repeat_activation_functions",
        "q_nn_activation_functions_list",
        "q_nn_repeat_activation_functions",
    ]


    data = {
        "heat_coef": 1.0,
        "alpha":     0.4,
        "t_start":   0.0,
        "t_stop":    1.0,
        "circle_radius":            1.0,
        "circle_center_in_xy":      [1.0, 1.0],
        "num_of_time_plots":        4,
        "num_of_x_plot_points":     200,
        "num_of_y_plot_points":     200,
        "equation_rhs_function": equation_rhs_function,
        "initial_condition_function": initial_condition_function,
        "desired_function_at_final_time": desired_polynomial,
    }
    # In data, anything you need can be stored.
    # For example: constants and information for solved equations and problem, points for evaluating the network, results of evaluation,
    #   training times, references to the model/networks, information for plotting, etc.
    # Basically anything extra from training parameters.
    # Data dictionary gets updated during runtime in different functions.

    call_main_training_script_for_circle_domain(
        data,
        training_configuration,
        parameter_names_for_custom_conversion,
        function_to_convert_custom_parameters,
    )
    tf.config.run_functions_eagerly(False)


def sample_run_with_small_setting_to_test_functionality(dir_name: str):
    tf.config.run_functions_eagerly(False)
    training_configuration = {
        "main_save_dir":                            os.path.join(".", "TEST runs", dir_name),
        "override_main_csv_file":                   True,
        "save_weights":                             True,
        "save_plots":                               True,
        "num_of_training_runs_for_one_setting":     1,
        "shuffle_training_points_each_epoch":       True,
        "u_nn_kernel_initializer":                  tf.keras.initializers.glorot_uniform,
        "q_nn_kernel_initializer":                  tf.keras.initializers.glorot_uniform,
        "loss_fn":                                  L2_norm,
        "optimizer":                                tf.keras.optimizers.Adam,
        "learning_rate":                            (0.001, 0.001, 0.001),
        "initial_epoch":                            0,
        "num_of_epochs":                            (100, 100, 100),
        "write_loss_values_every_x_epochs":         10,
        "iterable_with_cost_functional_loss_coefs": [(0.05, 1.0, 15.0)],
        "init_cond_loss_coef":                      (1.0, 1.0, 1.0),
        "boundary_loss_coef":                       (1.0, 1.0, 1.0),
        "num_of_t_training_points":                 (20, 21, 20),
        "num_of_grid_x_training_points":            (20, 21, 20),
        "num_of_grid_y_training_points":            (20, 21, 20),
        "num_xy_training_points_on_boundary":       (20, 31, 20),
        "u_nn_num_of_layers":                       (2, 3+1, 2),
        "u_nn_num_of_neurons_in_layer":             (20, 30+1, 10),
        "q_nn_num_of_layers":                       (2, 3+1, 2),
        "q_nn_num_of_neurons_in_layer":             (20, 30+1, 10),
        "num_of_batches":                           (2, 2, 2),
        "u_nn_activation_functions_list":           ["tanh", "relu"],
        "u_nn_repeat_activation_functions":         False,
        "q_nn_activation_functions_list":           ["tanh", "relu"],
        "q_nn_repeat_activation_functions":         False,
        "q_nn_final_activation_function":           ["relu"]
    }
    parameter_names_for_custom_conversion = [
        "u_nn_activation_functions_list",
        "u_nn_repeat_activation_functions",
        "q_nn_activation_functions_list",
        "q_nn_repeat_activation_functions",
    ]


    data = {
        "heat_coef": 1.0,
        "alpha":     0.4,
        "t_start":   0.0,
        "t_stop":    1.0,
        "circle_radius":            1.0,
        "circle_center_in_xy":      [1.0, 1.0],
        "num_of_time_plots":        10,
        "num_of_x_plot_points":     200,
        "num_of_y_plot_points":     200,
        "equation_rhs_function": equation_rhs_function,
        "initial_condition_function": initial_condition_function,
        "desired_function_at_final_time": desired_polynomial,
    }
    # In data, anything you need can be stored.
    # For example: constants and information for solved equations and problem, points for evaluating the network, results of evaluation,
    #   training times, references to the model/networks, information for plotting, etc.
    # Basically anything extra from training parameters.
    # Data dictionary gets updated during runtime in different functions.

    call_main_training_script_for_circle_domain(
        data,
        training_configuration,
        parameter_names_for_custom_conversion,
        function_to_convert_custom_parameters,
    )


def run_with_eager_functions_and_short_training_times(dir_name: str):
    tf.config.run_functions_eagerly(True)
    training_configuration = {
        "main_save_dir":                            os.path.join(".", "TEST runs", dir_name),
        "override_main_csv_file":                   True,
        "save_weights":                             True,
        "save_plots":                               True,
        "num_of_training_runs_for_one_setting":     1,
        "shuffle_training_points_each_epoch":       True,
        "u_nn_kernel_initializer":                  tf.keras.initializers.glorot_uniform,
        "q_nn_kernel_initializer":                  tf.keras.initializers.glorot_uniform,
        "loss_fn":                                  L2_norm,
        "optimizer":                                tf.keras.optimizers.Adam,
        "learning_rate":                            (0.001, 0.001, 0.001),
        "initial_epoch":                            0,
        "num_of_epochs":                            (10, 10, 10),
        "write_loss_values_every_x_epochs":         1,
        "iterable_with_cost_functional_loss_coefs": [(0.05, 1.0, 15.0)],
        "init_cond_loss_coef":                      (1.0, 1.0, 1.0),
        "boundary_loss_coef":                       (1.0, 1.0, 1.0),
        "num_of_t_training_points":                 (20, 21, 20),
        "num_of_grid_x_training_points":            (20, 21, 20),
        "num_of_grid_y_training_points":            (20, 21, 20),
        "num_xy_training_points_on_boundary":       (20, 31, 20),
        "u_nn_num_of_layers":                       (2, 3+1, 2),
        "u_nn_num_of_neurons_in_layer":             (20, 30+1, 10),
        "q_nn_num_of_layers":                       (2, 3+1, 2),
        "q_nn_num_of_neurons_in_layer":             (20, 30+1, 10),
        "num_of_batches":                           (2, 2, 2),
        "u_nn_activation_functions_list":           ["tanh", "relu"],
        "u_nn_repeat_activation_functions":         False,
        "q_nn_activation_functions_list":           ["tanh", "relu"],
        "q_nn_repeat_activation_functions":         False,
        "q_nn_final_activation_function":           ["relu"]
    }
    parameter_names_for_custom_conversion = [
        "u_nn_activation_functions_list",
        "u_nn_repeat_activation_functions",
        "q_nn_activation_functions_list",
        "q_nn_repeat_activation_functions",
    ]


    data = {
        "heat_coef": 1.0,
        "alpha":     0.4,
        "t_start":   0.0,
        "t_stop":    1.0,
        "circle_radius":            2.0,
        "circle_center_in_xy":      [2.0, 1.5],
        "num_of_time_plots":        10,
        "num_of_x_plot_points":     200,
        "num_of_y_plot_points":     200,
        "equation_rhs_function": equation_rhs_function,
        "initial_condition_function": initial_condition_function,
        "desired_function_at_final_time": Zernike_polynomial,
    }
    # In data, anything you need can be stored.
    # For example: constants and information for solved equations and problem, points for evaluating the network, results of evaluation,
    #   training times, references to the model/networks, information for plotting, etc.
    # Basically anything extra from training parameters.
    # Data dictionary gets updated during runtime in different functions.

    call_main_training_script_for_circle_domain(
        data,
        training_configuration,
        parameter_names_for_custom_conversion,
        function_to_convert_custom_parameters,
    )


def get_nn_with_weights_initialized_to_zero():
    nn = tf.keras.Sequential()
    nn.add(tf.keras.Input(shape=(3,)))
    nn.add(tf.keras.layers.Dense(200, trainable=False))
    nn.add(tf.keras.layers.Dense(200, trainable=False))
    nn.add(tf.keras.layers.Dense(200, trainable=False))
    nn.add(tf.keras.layers.Dense(200, trainable=False))
    nn.add(tf.keras.layers.Dense(200, trainable=False))
    nn.add(tf.keras.layers.Dense(200, trainable=False))
    nn.add(tf.keras.layers.Dense(200, trainable=False))
    nn.add(tf.keras.layers.Dense(200, trainable=False))
    nn.add(tf.keras.layers.Dense(200, trainable=False))
    nn.add(tf.keras.layers.Dense(200, trainable=False))
    nn.add(tf.keras.layers.Dense(200, trainable=False))
    nn.add(tf.keras.layers.Dense(200, trainable=False))
    nn.add(tf.keras.layers.Dense(200, trainable=False))
    nn.add(tf.keras.layers.Dense(200, trainable=False))
    nn.add(tf.keras.layers.Dense(200, trainable=False))
    nn.add(tf.keras.layers.Dense(3, trainable=False))
    output_layer = tf.keras.layers.Dense(1, kernel_initializer="zeros", trainable=False)
    nn.add(output_layer)
    return nn


def get_nn_that_returns_x():
    nn = get_nn_with_weights_initialized_to_zero()
    last_index = len(nn.layers)-1
    output_layer = nn.get_layer(index=last_index)
    output_layer_kernel_weights = output_layer.weights[0]
    output_layer_kernel_weights.assign([[0], [1], [0]])
    return nn


def debugging_mayvi_figures_crash(number_of_figure_generation_repeats, save_dir_base):
    x = tf.linspace(-1.0, 1.0, 200)
    y = tf.linspace(-1.0, 1.0, 200)
    t_stop = 1.0
    t = tf.linspace(0.0, t_stop, 10)

    X_plot, Y_plot, T_plot = tf.meshgrid(x, y, t, indexing="ij")
    
    u_nn = get_nn_that_returns_x()
    q_nn = get_nn_that_returns_x()

    Zernike_polynomial = lambda r, phi: r**2 * tf.sin(phi)

    circle_radius = 1.0
    circle_center = [0.0, 0.0]

    data = {
        "t_stop": t_stop,
        "X_plot": X_plot,
        "Y_plot": Y_plot,
        "T_plot": T_plot,
        "u_nn": u_nn,
        "q_nn": q_nn,
        "desired_function_at_final_time": Zernike_polynomial,
        "circle_radius": circle_radius,
        "circle_center_in_xy": circle_center,
    }

    training_setting_dictionary = {}

    for i in range(number_of_figure_generation_repeats):
        save_dir = os.path.join(save_dir_base, "figures", f"{i}")
        os.makedirs(save_dir, exist_ok=True)

        training_setting_dictionary["save_dir_for_current_specific_training_setting"] = save_dir

        plot_and_save_fig(training_setting_dictionary, data, plot_relative_nn_error_from_fitted_function)
        plot_and_save_fig(training_setting_dictionary, data, plot_absolute_nn_error_from_fitted_function)
        plot_and_save_fig(training_setting_dictionary, data, plot_nn_result_and_fitted_function)
        plot_and_save_fig(training_setting_dictionary, data, plot_q_nn)
