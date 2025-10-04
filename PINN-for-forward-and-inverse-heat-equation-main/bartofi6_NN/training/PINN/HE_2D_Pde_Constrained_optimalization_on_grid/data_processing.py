import itertools
import tensorflow as tf
from ....PINN.HE_2D_Pde_Constrained_optimalization_on_grid import *


def setup_data(data, config):
    """
    Implements additional data processing desired or needed for given problem.
    .
    """
    x_start = data["x_start"]
    x_stop = data["x_stop"]
    y_start = data["y_start"]
    y_stop = data["y_stop"]
    t_start = data["t_start"]
    t_stop = data["t_stop"]
    num_of_time_eval_points = data["num_of_time_eval_points"]
    num_of_x_eval_points = data["num_of_x_eval_points"]
    num_of_y_eval_points = data["num_of_y_eval_points"]
    num_of_time_plots = data["num_of_time_plots"]

    x = tf.linspace(x_start, x_stop, num=num_of_x_eval_points)
    y = tf.linspace(y_start, y_stop, num=num_of_y_eval_points)
    t = tf.linspace(t_start, t_stop, num=num_of_time_eval_points)

    X_eval, Y_eval, T_eval = tf.meshgrid(x, y, t, indexing='ij')
    # Indexování 'ij' zařídí, že bod na indexu [i,j,k] odpovídá bodu s hodnotami x[i], y[j], t[k].
    # Tedy prostě standardní kartészké indexování: z hodnot x, y a t se vygeneruje 3D mřížka a X, Y a T určují
    # hodnoty x, y a t v daných bodech. Např. hodnota x v bodě s indexy i,j,k je X[i,j,k], atd.

    X_plot, Y_plot = tf.meshgrid(x, y, indexing='ij')

    data["X_eval"] = X_eval
    data["Y_eval"] = Y_eval
    data["T_eval"] = T_eval
    data["X_plot"] = X_plot
    data["Y_plot"] = Y_plot

    # Delete unneeded tensors to free up memory.
    del x, y, t

    # x_eq = tf.reshape(X[1:-1, 1:-1, 1:], [-1])
    # y_eq = tf.reshape(Y[1:-1, 1:-1, 1:], [-1])
    # t_eq = tf.reshape(T[1:-1, 1:-1, 1:], [-1])


    # x_boundary_for_fixed_y0 = tf.reshape(X[:, 0, :], [-1])
    # y_boundary_for_fixed_y0 = tf.reshape(Y[:, 0, :], [-1])
    # t_boundary_for_fixed_y0 = tf.reshape(T[:, 0, :], [-1])

    # x_boundary_for_fixed_y_last = tf.reshape(X[:, -1, :], [-1])
    # y_boundary_for_fixed_y_last = tf.reshape(Y[:, -1, :], [-1])
    # t_boundary_for_fixed_y_last = tf.reshape(T[:, -1, :], [-1])

    # # Zde je y složka bodů od indexu 1 do předposledního (včetně), jelikož ty body s okrajovými hodnotami
    # # jsme již započetli výše.
    # x_boundary_for_fixed_x0 = tf.reshape(X[0, 1:-1, :], [-1])
    # y_boundary_for_fixed_x0 = tf.reshape(Y[0, 1:-1, :], [-1])
    # t_boundary_for_fixed_x0 = tf.reshape(T[0, 1:-1, :], [-1])

    # x_boundary_for_fixed_x_last = tf.reshape(X[-1, 1:-1, :], [-1])
    # y_boundary_for_fixed_x_last = tf.reshape(Y[-1, 1:-1, :], [-1])
    # t_boundary_for_fixed_x_last = tf.reshape(T[-1, 1:-1, :], [-1])

    # x_boundary = tf.concat([x_boundary_for_fixed_y0, x_boundary_for_fixed_y_last, x_boundary_for_fixed_x0, x_boundary_for_fixed_x_last], 0)
    # y_boundary = tf.concat([y_boundary_for_fixed_y0, y_boundary_for_fixed_y_last, y_boundary_for_fixed_x0, y_boundary_for_fixed_x_last], 0)
    # t_boundary = tf.concat([t_boundary_for_fixed_y0, t_boundary_for_fixed_y_last, t_boundary_for_fixed_x0, t_boundary_for_fixed_x_last], 0)

    # # Delete unneeded tensors to free up memory.
    # del x_boundary_for_fixed_y0, y_boundary_for_fixed_y0, t_boundary_for_fixed_y0,
    # del x_boundary_for_fixed_y_last, y_boundary_for_fixed_y_last, t_boundary_for_fixed_y_last,
    # del x_boundary_for_fixed_x0, y_boundary_for_fixed_x0, t_boundary_for_fixed_x0,
    # del x_boundary_for_fixed_x_last, y_boundary_for_fixed_x_last, t_boundary_for_fixed_x_last


    # x_ini = tf.reshape(X[:, :, 0], [-1])
    # y_ini = tf.reshape(Y[:, :, 0], [-1])
    # t_ini = tf.reshape(T[:, :, 0], [-1])

    # # del X, Y, T
    
    # data["x_eq"] = x_eq
    # data["y_eq"] = y_eq
    # data["t_eq"] = t_eq
    # data["x_boundary"] = x_boundary
    # data["y_boundary"] = y_boundary
    # data["t_boundary"] = t_boundary
    # data["x_ini"] = x_ini
    # data["y_ini"] = y_ini
    # data["t_ini"] = t_ini
    # data["X"] = X
    # data["Y"] = Y
    # data["T"] = T

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


def process_results(training_setting_dictionary: dict, data: dict):
    save_weighted_norms_calculated_after_training(training_setting_dictionary, data)


def delete_model(training_setting_dictionary: dict, data: dict):
    del data["u_nn"]
    del data["q_nn"]
