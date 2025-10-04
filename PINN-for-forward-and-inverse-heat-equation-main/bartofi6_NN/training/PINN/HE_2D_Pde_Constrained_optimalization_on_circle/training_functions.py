import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
from ....PINN.HE_2D_Pde_Constrained_optimalization_on_circle import *

import os


def create_u_nn(training_setting_dictionary: dict, data: dict):
    if "u_nn" in data.keys() and data["u_nn"]:
        del data["u_nn"]

    dtype = training_setting_dictionary["dtype"]

    u_nn = Sequential()
    u_nn.add(Input(shape=(3,), dtype=dtype))

    for layer_index in range(training_setting_dictionary["u_nn_num_of_layers"]):
        list_with_activation_functions_for_current_layer_for_u_nn = training_setting_dictionary["list_with_current_activation_functions_for_u_nn"]

        u_nn_layer = Dense(
            units               = training_setting_dictionary["u_nn_num_of_neurons_in_layer"],
            activation          = list_with_activation_functions_for_current_layer_for_u_nn[layer_index],
            kernel_initializer  = training_setting_dictionary["u_nn_kernel_initializer"],
            dtype               = dtype,
        )
        u_nn.add(u_nn_layer)
    u_nn.add(tf.keras.layers.Dense(1, kernel_initializer=training_setting_dictionary["u_nn_kernel_initializer"], dtype=dtype))

    data["u_nn"] = u_nn


def create_q_nn(training_setting_dictionary: dict, data: dict):
    dtype = training_setting_dictionary["dtype"]

    q_nn = Sequential()
    q_nn.add(Input(shape=(3,), dtype=dtype))

    for layer_index in range(training_setting_dictionary["q_nn_num_of_layers"]):
        list_with_activation_functions_for_current_layer_for_q_nn = training_setting_dictionary["list_with_current_activation_functions_for_q_nn"]

        q_nn_layer = Dense(
            units               = training_setting_dictionary["q_nn_num_of_neurons_in_layer"],
            activation          = list_with_activation_functions_for_current_layer_for_q_nn[layer_index],
            kernel_initializer  = training_setting_dictionary["q_nn_kernel_initializer"],
            dtype               = dtype,
        )
        q_nn.add(q_nn_layer)
    q_nn.add(tf.keras.layers.Dense(1,
                                   kernel_initializer=training_setting_dictionary["q_nn_kernel_initializer"],
                                   activation=training_setting_dictionary["q_nn_final_activation_function"],
                                   dtype=dtype))

    data["q_nn"] = q_nn


def _call_train_u_nn_and_q_nn_function_with_one_parameter_difference_for_circle_domain(training_setting_dictionary: dict, data: dict, train_only_u_nn: bool):
    return train_u_nn_and_q_nn(u_nn=                               data["u_nn"],
                               q_nn=                               data["q_nn"],
                               equation_rhs_function=              data["equation_rhs_function"],
                               initial_condition_function=         data["initial_condition_function"],
                               desired_function_at_final_time=     data["desired_function_at_final_time"],
                               heat_coef=                          data["heat_coef"],
                               alpha=                              data["alpha"],
                               circle_center_in_xy=                data["circle_center_in_xy"],
                               circle_radius=                      data["circle_radius"],
                               t_start=                            data["t_start"],
                               t_stop=                             data["t_stop"],
                               num_t_training_points=              training_setting_dictionary["num_of_t_training_points"],
                               num_of_grid_x_training_points=      training_setting_dictionary["num_of_grid_x_training_points"],
                               num_of_grid_y_training_points=      training_setting_dictionary["num_of_grid_y_training_points"],
                               num_xy_training_points_on_boundary= training_setting_dictionary["num_xy_training_points_on_boundary"],
                               num_batches=                        training_setting_dictionary["num_of_batches"],
                               initial_epoch=                      training_setting_dictionary["initial_epoch"],
                               num_epochs=                         training_setting_dictionary["num_of_epochs"],
                               write_loss_values_every_x_epochs=   training_setting_dictionary["write_loss_values_every_x_epochs"],
                               boundary_condition_weight=          training_setting_dictionary["boundary_loss_coef"],
                               initial_condition_weight=           training_setting_dictionary["init_cond_loss_coef"],
                               cost_functional_weight=             training_setting_dictionary["cost_functional_loss_coef"],
                               optimizer=                          training_setting_dictionary["optimizer"],
                               loss_fn=                            training_setting_dictionary["loss_fn"],
                               callbacks=                          None,
                               shuffle_each_epoch=                 training_setting_dictionary["shuffle_training_points_each_epoch"],
                               train_only_u_nn=                    train_only_u_nn,)


def _compile_u_nn(training_setting_dictionary: dict, data: dict):
    optimizer = training_setting_dictionary["optimizer"]
    learning_rate = training_setting_dictionary["learning_rate"]
    loss_fn = training_setting_dictionary["loss_fn"]

    u_nn = data["u_nn"]
    u_nn.compile(optimizer=optimizer(learning_rate), loss=loss_fn)


def _compile_q_nn(training_setting_dictionary: dict, data: dict):
    optimizer = training_setting_dictionary["optimizer"]
    learning_rate = training_setting_dictionary["learning_rate"]
    loss_fn = training_setting_dictionary["loss_fn"]

    q_nn = data["q_nn"]
    q_nn.compile(optimizer=optimizer(learning_rate), loss=loss_fn)


def train_model_on_circle_domain(training_setting_dictionary: dict, data: dict):
    create_u_nn(training_setting_dictionary, data)
    _compile_u_nn(training_setting_dictionary, data)

    create_q_nn(training_setting_dictionary, data)
    _compile_q_nn(training_setting_dictionary, data)

    print("---------------------------------------------------------------------------")
    print("---------------------------------------------------------------------------")
    for key, value in training_setting_dictionary.items():
        print(f"{key}: {value}")
    print()

    print("Training u_nn and q_nn")
    _, _, training_time_of_u_nn_and_q_nn = \
        _call_train_u_nn_and_q_nn_function_with_one_parameter_difference_for_circle_domain(training_setting_dictionary,
                                                                                           data,
                                                                                           train_only_u_nn=False)
    print()
    data["training_time_of_u_nn_and_q_nn"] = training_time_of_u_nn_and_q_nn


    create_u_nn(training_setting_dictionary, data)
    _compile_u_nn(training_setting_dictionary, data)

    print("Training u_nn")
    loss_history_dict, dictionary_with_norms_after_training, training_time_of_u_nn = \
        _call_train_u_nn_and_q_nn_function_with_one_parameter_difference_for_circle_domain(training_setting_dictionary,
                                                                                           data,
                                                                                           train_only_u_nn=True)

    data["loss_history_dict"] = loss_history_dict
    data["dictionary_with_norms_after_training"] = dictionary_with_norms_after_training
    data["training_time_of_u_nn"] = training_time_of_u_nn

    cost_functional_loss_coef = training_setting_dictionary["cost_functional_loss_coef"]
    dictionary_with_norms_after_training_for_different_cost_functional_coeficients =\
        data["dictionary_with_norms_after_training_for_different_cost_functional_coeficients"]
    dictionary_with_norms_after_training_for_different_cost_functional_coeficients[cost_functional_loss_coef] =\
        dictionary_with_norms_after_training
    return


def save_model(training_setting_dictionary: dict, data: dict):
    save_dir_for_current_specific_training_setting = training_setting_dictionary["save_dir_for_current_specific_training_setting"]

    u_nn_save_file = os.path.join(save_dir_for_current_specific_training_setting, "u_nn.weights.h5")
    data["u_nn"].save_weights(u_nn_save_file)

    q_nn_save_file = os.path.join(save_dir_for_current_specific_training_setting, "q_nn.weights.h5")
    data["q_nn"].save_weights(q_nn_save_file)
