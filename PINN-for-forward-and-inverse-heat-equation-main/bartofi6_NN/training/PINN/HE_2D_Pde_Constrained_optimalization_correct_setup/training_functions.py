from ....PINN.HE_2D_Pde_Constrained_optimalization_correct_setup import train_u_nn_and_q_nn
from ..HE_2D_Pde_Constrained_optimalization_on_circle.training_functions import create_q_nn
from ..HE_2D_Pde_Constrained_optimalization_on_grid.training_functions import \
    create_u_nn, \
    _compile_u_nn, \
    _compile_q_nn, \
    save_model


def _call_train_u_nn_and_q_nn_function_with_one_parameter_difference_for_circle_domain(training_setting_dictionary: dict, data: dict, train_only_u_nn: bool):
    return train_u_nn_and_q_nn(u_nn=                                    data["u_nn"],
                               q_nn=                                    data["q_nn"],
                               equation_rhs_function=                   data["equation_rhs_function"],
                               initial_condition_function=              data["initial_condition_function"],
                               desired_function_at_final_time=          data["desired_function_at_final_time"],
                               heat_coef=                               data["heat_coef"],
                               alpha=                                   data["alpha"],
                               circle_center_in_xy=                     data["circle_center_in_xy"],
                               circle_radius=                           data["circle_radius"],
                               t_start=                                 data["t_start"],
                               t_stop=                                  data["t_stop"],
                               num_t_training_points=                   training_setting_dictionary["num_of_t_training_points"],
                               x_start=                                 data["x_start"],
                               x_stop=                                  data["x_stop"],
                               num_of_grid_x_training_points=           training_setting_dictionary["num_of_grid_x_training_points"],
                               num_of_x_points_for_integral_evaluation= training_setting_dictionary["num_of_x_points_for_integral_evaluation"],
                               y_start=                                 data["y_start"],
                               y_stop=                                  data["y_stop"],
                               num_of_grid_y_training_points=           training_setting_dictionary["num_of_grid_y_training_points"],
                               num_of_y_points_for_integral_evaluation= training_setting_dictionary["num_of_y_points_for_integral_evaluation"],
                               num_batches=                             training_setting_dictionary["num_of_batches"],
                               initial_epoch=                           training_setting_dictionary["initial_epoch"],
                               num_epochs=                              training_setting_dictionary["num_of_epochs"],
                               write_loss_values_every_x_epochs=        training_setting_dictionary["write_loss_values_every_x_epochs"],
                               boundary_condition_weight=               training_setting_dictionary["boundary_loss_coef"],
                               initial_condition_weight=                training_setting_dictionary["init_cond_loss_coef"],
                               cost_functional_weight=                  training_setting_dictionary["cost_functional_loss_coef"],
                               q_nn_cooling_penalty_weight=             training_setting_dictionary["q_nn_cooling_penalty_weight"],
                               optimizer=                               training_setting_dictionary["optimizer"],
                               loss_fn=                                 training_setting_dictionary["loss_fn"],
                               dtype=                                   training_setting_dictionary["dtype"],
                               callbacks=                               None,
                               shuffle_each_epoch=                      training_setting_dictionary["shuffle_training_points_each_epoch"],
                               train_only_u_nn=                         train_only_u_nn,)


def train_model_for_correct_setup(training_setting_dictionary: dict, data: dict):
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
