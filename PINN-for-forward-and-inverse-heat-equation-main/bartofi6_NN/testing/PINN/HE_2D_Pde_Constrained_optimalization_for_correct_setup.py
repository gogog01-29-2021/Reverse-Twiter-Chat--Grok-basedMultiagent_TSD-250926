import tensorflow as tf
import os
import math
from ...training.PINN.general import L2_norm
from ...training.PINN.HE_2D_Pde_Constrained_optimalization_correct_setup import \
    call_main_training_script_for_correct_setup, \
    function_to_convert_activation_functions_in_custom_way


def equation_rhs_function(t, x, y, u, heat_coef):
    return 0.0


def initial_condition_function(x, y, u, heat_coef):
    return 0.0


def desired_polynomial(x, y):
    return x*y


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
        "dtype":                                    tf.float64,
        "optimizer":                                tf.keras.optimizers.Adam,
        "learning_rate":                            (0.001, 0.001, 0.001),
        "initial_epoch":                            0,
        "num_of_epochs":                            (100, 100, 100),
        "write_loss_values_every_x_epochs":         10,
        "iterable_with_cost_functional_loss_coefs": [(0.05, 1.0, 15.0)],
        "init_cond_loss_coef":                      (1.0, 1.0, 1.0),
        "boundary_loss_coef":                       (1.0, 1.0, 1.0),
        "q_nn_cooling_penalty_weight":              (1.0, 1.0, 1.0),
        "num_of_t_training_points":                 (20, 21, 20),
        "num_of_grid_x_training_points":            (20, 21, 20),
        "num_of_grid_y_training_points":            (20, 21, 20),
        "num_of_x_points_for_integral_evaluation":  100,
        "num_of_y_points_for_integral_evaluation":  100,
        "u_nn_num_of_layers":                       (2, 3+1, 2),
        "u_nn_num_of_neurons_in_layer":             (20, 30+1, 10),
        "q_nn_num_of_layers":                       (2, 3+1, 2),
        "q_nn_num_of_neurons_in_layer":             (20, 30+1, 10),
        "num_of_batches":                           (2, 2, 2),
        "u_nn_activation_functions_list":           ["tanh", "relu"],
        "u_nn_repeat_activation_functions":         False,
        "q_nn_activation_functions_list":           ["tanh", "relu"],
        "q_nn_repeat_activation_functions":         False,
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
        "dtype":                                    tf.float64,
        "optimizer":                                tf.keras.optimizers.Adam,
        "learning_rate":                            (0.001, 0.001, 0.001),
        "initial_epoch":                            0,
        "num_of_epochs":                            (10, 10, 10),
        "write_loss_values_every_x_epochs":         1,
        "iterable_with_cost_functional_loss_coefs": [(0.05, 1.0, 15.0)],
        "init_cond_loss_coef":                      (1.0, 1.0, 1.0),
        "boundary_loss_coef":                       (1.0, 1.0, 1.0),
        "q_nn_cooling_penalty_weight":              (1.0, 1.0, 1.0),
        "num_of_t_training_points":                 (20, 21, 20),
        "num_of_grid_x_training_points":            (20, 21, 20),
        "num_of_grid_y_training_points":            (20, 21, 20),
        "num_of_x_points_for_integral_evaluation":  100,
        "num_of_y_points_for_integral_evaluation":  100,
        "u_nn_num_of_layers":                       (2, 3+1, 2),
        "u_nn_num_of_neurons_in_layer":             (20, 30+1, 10),
        "q_nn_num_of_layers":                       (2, 3+1, 2),
        "q_nn_num_of_neurons_in_layer":             (20, 30+1, 10),
        "num_of_batches":                           (2, 2, 2),
        "u_nn_activation_functions_list":           ["tanh", "relu"],
        "u_nn_repeat_activation_functions":         False,
        "q_nn_activation_functions_list":           ["tanh", "relu"],
        "q_nn_repeat_activation_functions":         False,
        "q_nn_final_activation_function":           None
    }
    parameter_names_for_custom_conversion = [
        "u_nn_activation_functions_list",
        "u_nn_repeat_activation_functions",
        "q_nn_activation_functions_list",
        "q_nn_repeat_activation_functions",
    ]


    data["circle_radisu"] = 2.0
    data["circle_center_in_xy"] = [2.0, 1.5]

    call_main_training_script_for_correct_setup(
        data,
        training_configuration,
        parameter_names_for_custom_conversion,
        function_to_convert_activation_functions_in_custom_way,
    )
