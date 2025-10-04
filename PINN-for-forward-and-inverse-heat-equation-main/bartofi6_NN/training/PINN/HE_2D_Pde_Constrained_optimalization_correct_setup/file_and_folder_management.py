import tensorflow as tf
import os
import csv


def write_main_csv_file_header_row(config: dict, data: dict):
    parameter_names_to_write = [
        "training_number",
        "training_time_of_u_nn_and_q_nn",
        "training_time_of_u_nn",
        "L1_equation_norm",
        "L2_equation_norm",
        "max_equation_norm",
        "L1_boundary_cond_norm",
        "L2_boundary_cond_norm",
        "max_boundary_cond_norm",
        "L1_init_cond_norm",
        "L2_init_cond_norm",
        "max_init_cond_norm",
        "weighted_L1_equation_norm",
        "weighted_L2_equation_norm",
        "weighted_max_equation_norm",
        "weighted_L1_boundary_cond_norm",
        "weighted_L2_boundary_cond_norm",
        "weighted_max_boundary_cond_norm",
        "weighted_L1_init_cond_norm",
        "weighted_L2_init_cond_norm",
        "weighted_max_init_cond_norm",
        "cost_functional_loss",
        "weighted_cost_functional_loss",
        "num_of_t_training_points",
        "num_of_grid_x_training_points",
        "num_of_grid_y_training_points",
        "list_with_current_activation_functions_for_u_nn",
        "list_with_current_activation_functions_for_q_nn",
        "u_nn_num_of_layers",
        "u_nn_num_of_neurons_in_layer",
        "q_nn_num_of_layers",
        "q_nn_num_of_neurons_in_layer",
        "cost_functional_loss_coef",
        "init_cond_loss_coef",
        "boundary_loss_coef",
        "num_of_epochs",
        "num_of_batches",
        "learning_rate",
        "shuffle_training_points_each_epoch",
        "loss_fn",
        "optimizer",
        "u_nn_kernel_initializer",
        "q_nn_kernel_initializer",
        "num_of_training_runs_for_one_setting",
    ]

    data["parameters_names_to_write_to_main_csv_file"] = parameter_names_to_write

    main_csv_file = config["main_csv_file"]
    with open(main_csv_file, "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(parameter_names_to_write)

    return


def generate_letter_from_activation_function(activation_function):
    if activation_function == "relu" or activation_function == tf.keras.activations.relu:
        return "R"
    elif activation_function == "sigmoid" or activation_function == tf.keras.activations.sigmoid:
        return "S"
    elif activation_function == "tanh" or activation_function == tf.keras.activations.tanh:
        return "T"
    else:
        return "I"  # identity


def _generate_save_dir_name_for_current_training(training_setting_dictionary: dict, data: dict):
    list_with_current_activation_functions_for_u_nn = training_setting_dictionary["list_with_current_activation_functions_for_u_nn"]
    string_representing_activation_functions_for_u_nn = "".join(
        generate_letter_from_activation_function(activation_function) for activation_function in list_with_current_activation_functions_for_u_nn)

    list_with_current_activation_functions_for_q_nn = training_setting_dictionary["list_with_current_activation_functions_for_q_nn"]
    string_representing_activation_functions_for_q_nn = "".join(
        generate_letter_from_activation_function(activation_function) for activation_function in list_with_current_activation_functions_for_q_nn)

    return (
        f"q_nn: {training_setting_dictionary['q_nn_num_of_layers']}l "
        f"{training_setting_dictionary['q_nn_num_of_neurons_in_layer']}n "
        f"{string_representing_activation_functions_for_q_nn} | "
        f"{training_setting_dictionary['num_of_t_training_points']} t pts | "
        f"{training_setting_dictionary['num_of_grid_x_training_points']} x pts | "
        f"{training_setting_dictionary['num_of_grid_y_training_points']} y pts | "
        f"{training_setting_dictionary['num_of_epochs']} ep | "
        f"{training_setting_dictionary['num_of_batches']} b | "
        f"b_coef: {training_setting_dictionary['boundary_loss_coef']} | "
        f"ini_coef: {training_setting_dictionary['init_cond_loss_coef']} | "
        f"u_nn: {training_setting_dictionary['u_nn_num_of_layers']}l "
        f"{training_setting_dictionary['u_nn_num_of_neurons_in_layer']}n "
        f"{string_representing_activation_functions_for_u_nn}"
    )


def create_save_dir_and_csv_file_for_current_general_training_setting(training_setting_dictionary: dict, data: dict):
    save_dir_name_for_current_training = _generate_save_dir_name_for_current_training(training_setting_dictionary, data)
    training_setting_dictionary["save_dir_for_current_general_training_setting"] = os.path.join(training_setting_dictionary["main_save_dir"],
                                                                                                save_dir_name_for_current_training)
    if not os.path.exists(training_setting_dictionary["save_dir_for_current_general_training_setting"]):
        os.mkdir(training_setting_dictionary["save_dir_for_current_general_training_setting"])


    name_of_csv_file_for_current_general_training_setting = "results for current training setting.csv"
    training_setting_dictionary["csv_file_for_current_general_training_setting"] = \
        os.path.join(training_setting_dictionary["save_dir_for_current_general_training_setting"],
                     name_of_csv_file_for_current_general_training_setting)

    with open(training_setting_dictionary["csv_file_for_current_general_training_setting"], "w", newline="") as file:
        writer = csv.writer(file)
        header_names = [
            "training number",
            "training_time_of_u_nn_and_q_nn",
            "training_time_of_u_nn",
            "L1_equation_norm",
            "L2_equation_norm",
            "max_equation_norm",
            "L1_init_cond_norm",
            "L2_init_cond_norm",
            "max_init_cond_norm",
            "L1_boundary_cond_norm",
            "L2_boundary_cond_norm",
            "max_boundary_cond_norm",
            "cost_functional_loss",
            "cost_functional_loss_coef",
        ]
        writer.writerow(header_names)

        data["parameters_names_to_write_into_csv_file_for_current_setting"] = header_names


def create_txt_file_with_parameters_for_current_training_setting(training_setting_dictionary: dict, data: dict):
    file_with_training_setting = os.path.join(training_setting_dictionary["save_dir_for_current_general_training_setting"],
                                              "training setting.txt")

    with open(file_with_training_setting, "w") as file:
        for parameter_name in training_setting_dictionary.keys():
            file.write(f"{parameter_name}: {training_setting_dictionary[parameter_name]}\n")
        for parameter_name in data.keys():
            file.write(f"{parameter_name}: {data[parameter_name]}\n")


def create_save_dir_for_current_specific_training_setting(training_setting_dictionary: dict, data: dict):
    save_dir_for_current_general_training_setting = training_setting_dictionary["save_dir_for_current_general_training_setting"]

    training_number = training_setting_dictionary["training_number"]
    cost_functional_loss_coef = training_setting_dictionary["cost_functional_loss_coef"]

    name_of_save_dir_for_current_specific_training_setting = (f"training number {training_number} | "
                                                              f"cost functional loss coef: {cost_functional_loss_coef}")

    save_dir_for_current_specific_training_setting = os.path.join(save_dir_for_current_general_training_setting,
                                                                  name_of_save_dir_for_current_specific_training_setting)

    if not os.path.exists(save_dir_for_current_specific_training_setting):
        os.mkdir(save_dir_for_current_specific_training_setting)

    training_setting_dictionary["save_dir_for_current_specific_training_setting"] = save_dir_for_current_specific_training_setting


def append_this_trainings_results_into_csv_file_with_results_for_single_training_setting(training_setting_dictionary: dict, data: dict):
    dictionary_with_norms_after_training = data["dictionary_with_norms_after_training"]

    values_of_results_to_write = []

    parameter_names_to_write = data["parameter_names_to_write_into_file_with_training_setting"]
    for parameter_name in parameter_names_to_write:
        if parameter_name in training_setting_dictionary.keys():
            values_of_results_to_write.append(training_setting_dictionary[parameter_name])

        elif parameter_name in data.keys():
            values_of_results_to_write.append(data[parameter_name])

        elif parameter_name in dictionary_with_norms_after_training.keys():
            values_of_results_to_write.append(dictionary_with_norms_after_training[parameter_name])

        else:
            raise (KeyError((f"Couldn't find key {parameter_name} in neither training_setting_dictionary, data nor "
                             f"dictionary_with_norms_after_training when trying to append results into csv file "
                             f"for current training setting.")))

    for parameter_index, parameter_value in enumerate(values_of_results_to_write):
        if isinstance(parameter_value, tf.Tensor):
            values_of_results_to_write[parameter_index] = parameter_value.numpy()

    csv_file_for_current_general_training_setting = training_setting_dictionary["csv_file_for_current_general_training_setting"]
    with open(csv_file_for_current_general_training_setting, "a") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(values_of_results_to_write)

    return


def append_this_trainings_results_into_main_csv_file(training_setting_dictionary: dict, data: dict):
    dictionary_with_norms_after_training = data["dictionary_with_norms_after_training"]

    values_of_results_to_write = []

    parameter_names_to_write = data["parameters_names_to_write_to_main_csv_file"]
    for parameter_name in parameter_names_to_write:
        if parameter_name in training_setting_dictionary.keys():
            values_of_results_to_write.append(training_setting_dictionary[parameter_name])

        elif parameter_name in data.keys():
            values_of_results_to_write.append(data[parameter_name])

        elif parameter_name in dictionary_with_norms_after_training.keys():
            values_of_results_to_write.append(dictionary_with_norms_after_training[parameter_name])

        else:
            raise (KeyError((f"Couldn't find key {parameter_name} in neither training_setting_dictionary, data nor"
                             f"dictionary_with_norms_after_training when trying to append results into main csv file.")))

    for parameter_index, parameter_value in enumerate(values_of_results_to_write):
        if isinstance(parameter_value, tf.Tensor):
            values_of_results_to_write[parameter_index] = parameter_value.numpy()

    main_csv_file = training_setting_dictionary["main_csv_file"]
    with open(main_csv_file, "a") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(values_of_results_to_write)

    return
