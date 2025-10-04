import tensorflow as tf

import os

from mayavi import mlab
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import math


def plot_nn_result_and_fitted_function(
    training_setting_dictionary,
    data,
    X_plot,
    Y_plot,
    u_pred_plot,
    q_values_plot,
    fitted_function_plot,
    mask
):
    u_nn_surf = mlab.surf(X_plot, Y_plot, u_pred_plot, colormap="gnuplot", mask=mask)
    u_f_surf = mlab.surf(X_plot, Y_plot, fitted_function_plot, colormap="ocean", mask=mask)

    space_from_the_right = 0.01
    width = 0.1

    mlab.colorbar(object=u_nn_surf, title="u_nn", nb_labels=5, orientation="vertical")
    u_nn_surf.module_manager.scalar_lut_manager.scalar_bar_representation.position = [1.0 - 2*width - space_from_the_right, 0.15]
    u_nn_surf.module_manager.scalar_lut_manager.scalar_bar_representation.position2 = [width, 0.7]
    u_nn_surf.module_manager.scalar_lut_manager.label_text_property.font_size = 10

    mlab.colorbar(object=u_f_surf, title="p", nb_labels=5, orientation="vertical")
    u_f_surf.module_manager.scalar_lut_manager.scalar_bar_representation.position = [1.0 - width - space_from_the_right, 0.15]
    u_f_surf.module_manager.scalar_lut_manager.scalar_bar_representation.position2 = [width, 0.7]
    u_f_surf.module_manager.scalar_lut_manager.label_text_property.font_size = 10

    mlab.axes(xlabel="x", ylabel="y", zlabel="u(T, x, y)")
    mlab.move(-1.0)  # Move the camera a bit so everything fits within the png.

    save_dir = training_setting_dictionary["save_dir_for_current_specific_training_setting"]
    save_file = os.path.join(save_dir, "NN and fitted function values at final time" + ".png")
    mlab.savefig(save_file)

    del u_nn_surf, u_f_surf


def plot_relative_nn_error_from_fitted_function(
    training_setting_dictionary,
    data,
    X_plot,
    Y_plot,
    u_pred_plot,
    q_values_plot,
    fitted_function_plot,
    mask
):
    keep_points_inside_circle = np.invert(mask)
    relative_nn_error = (u_pred_plot - fitted_function_plot) / np.max(fitted_function_plot[keep_points_inside_circle])
    rel_error_surf = mlab.surf(X_plot, Y_plot, relative_nn_error, colormap="gnuplot", mask=mask)

    space_from_the_right = 0.01
    width = 0.1

    mlab.colorbar(object=rel_error_surf, title="relativní chyba:\nu_nn - p\n------------\n  max(p)", nb_labels=5, orientation="vertical")
    rel_error_surf.module_manager.scalar_lut_manager.scalar_bar_representation.position = [1.0 - width - space_from_the_right, 0.15]
    rel_error_surf.module_manager.scalar_lut_manager.scalar_bar_representation.position2 = [width, 0.7]

    mlab.axes(xlabel="x", ylabel="y", zlabel="relativní chyba:\n(u_nn - p) / max(p)")
    mlab.move(-1.5)  # Move the camera a bit so everything fits within the png.

    save_dir = training_setting_dictionary["save_dir_for_current_specific_training_setting"]
    save_file = os.path.join(save_dir, "relative error: (u_nn - p) ÷ max(p)" + ".png")
    mlab.savefig(save_file)

    del rel_error_surf, relative_nn_error


def plot_absolute_nn_error_from_fitted_function(
    training_setting_dictionary,
    data,
    X_plot,
    Y_plot,
    u_pred_plot,
    q_values_plot,
    fitted_function_plot,
    mask
):
    absolute_nn_error = u_pred_plot - fitted_function_plot
    abs_error_surf = mlab.surf(X_plot, Y_plot, absolute_nn_error, colormap="gnuplot", mask=mask)

    space_from_the_right = 0.01
    width = 0.1

    mlab.colorbar(object=abs_error_surf, title="absolutní chyba\nu_nn - p", nb_labels=5, orientation="vertical")
    abs_error_surf.module_manager.scalar_lut_manager.scalar_bar_representation.position = [1.0 - width - space_from_the_right, 0.15]
    abs_error_surf.module_manager.scalar_lut_manager.scalar_bar_representation.position2 = [width, 0.7]

    mlab.axes(xlabel="x", ylabel="y", zlabel="absolutní chyba:\nu_nn - p")
    mlab.move(-1.0)  # Move the camera a bit so everything fits within the png.

    save_dir = training_setting_dictionary["save_dir_for_current_specific_training_setting"]
    save_file = os.path.join(save_dir, "absolute error: u_nn - p" + ".png")
    mlab.savefig(save_file)

    del abs_error_surf, absolute_nn_error


def plot_q_nn(
    training_setting_dictionary,
    data,
    X_plot,
    Y_plot,
    u_pred_plot,
    q_values_plot,
    fitted_function_plot,
    mask
):
    q_nn_surf = mlab.surf(X_plot, Y_plot, q_values_plot, colormap="gnuplot", mask=mask)

    mlab.colorbar(object=q_nn_surf, title="q_nn", nb_labels=5)
    q_nn_surf.module_manager.scalar_lut_manager.scalar_bar_representation.position = [0.15, 0.85]
    q_nn_surf.module_manager.scalar_lut_manager.scalar_bar_representation.position2 = [0.7, 0.15]

    mlab.axes(xlabel="x", ylabel="y", zlabel="q(T, x, y)")
    mlab.move(-1.5)  # Move the camera a bit so everything fits within the png.

    save_dir = training_setting_dictionary["save_dir_for_current_specific_training_setting"]
    save_file = os.path.join(save_dir, "q_nn function values at final time" + ".png")
    mlab.savefig(save_file)


def plot_and_save_fig(training_setting_dictionary, data, function_with_plotting_procedure):
    X_plot = data["X_plot"]
    Y_plot = data["Y_plot"]
    u_nn = data["u_nn"]
    q_nn = data["q_nn"]
    desired_function_at_final_time = data["desired_function_at_final_time"]
    circle_radius = data["circle_radius"]
    circle_center_in_xy = data["circle_center_in_xy"]

    mlab.figure(size=(800, 600), bgcolor=(0.3, 0.3, 0.3))

    x_eval = tf.reshape(X_plot, shape=-1)
    y_eval = tf.reshape(Y_plot, shape=-1)
    t_eval = tf.constant(data["t_stop"], shape=x_eval.shape)

    u_nn_pred = u_nn(tf.stack([t_eval, x_eval, y_eval], axis=-1), training=False)
    u_nn_plot_matrix = tf.reshape(u_nn_pred, shape=X_plot.shape)

    q_nn_values = q_nn(tf.stack([t_eval, x_eval, y_eval], axis=-1), training=False)
    q_nn_plot_matrix = tf.reshape(q_nn_values, shape=X_plot.shape)

    x_eval_on_unit_circle = (x_eval - circle_center_in_xy[0]) / circle_radius
    y_eval_on_unit_circle = (y_eval - circle_center_in_xy[1]) / circle_radius

    radial_distance = tf.math.sqrt(x_eval_on_unit_circle**2 + y_eval_on_unit_circle**2)
    polar_angle = tf.math.atan2(y_eval_on_unit_circle, x_eval_on_unit_circle)

    desired_function_at_final_time_values = desired_function_at_final_time(radial_distance, polar_angle)
    desired_function_at_final_time_plot_matrix = tf.reshape(desired_function_at_final_time_values, shape=X_plot.shape)

    # TODO: Add plotting for different times, right now only the last time gets plotted.
    X_plot = X_plot.numpy()[:, :, -1]
    Y_plot = Y_plot.numpy()[:, :, -1]
    u_nn_plot_matrix = u_nn_plot_matrix.numpy()[:, :, -1]
    q_nn_plot_matrix = q_nn_plot_matrix.numpy()[:, :, -1]
    desired_function_at_final_time_plot_matrix = desired_function_at_final_time_plot_matrix.numpy()[:, :, -1]

    mask_points_outside_circle_domain = (X_plot - circle_center_in_xy[0])**2 + (Y_plot - circle_center_in_xy[1])**2 > circle_radius**2

    # print(f"X_plot shape: {X_plot.shape}")
    # print(f"Y_plot shape: {Y_plot.shape}")
    # print(f"u_nn_plot_matrix shape: {u_nn_plot_matrix.shape}")
    # print(f"q_nn_plot_matrix shape: {q_nn_plot_matrix.shape}")
    # print(f"desired_function_at_final_time_plot_matrix shape: {desired_function_at_final_time_plot_matrix.shape}")
    # print(f"X_plot type: {type(X_plot)}")
    # print(f"Y_plot type: {type(Y_plot)}")
    # print(f"u_nn_plot_matrix type: {type(u_nn_plot_matrix)}")
    # print(f"q_nn_plot_matrix type: {type(q_nn_plot_matrix)}")
    # print(f"desired_function_at_final_time_plot_matrix type: {type(desired_function_at_final_time_plot_matrix)}")
    # TODO: Zkontrolovat velikosti, zda je mam dobře.
    function_with_plotting_procedure(training_setting_dictionary,
                                     data,
                                     X_plot,
                                     Y_plot,
                                     u_nn_plot_matrix,
                                     q_nn_plot_matrix,
                                     desired_function_at_final_time_plot_matrix,
                                     mask_points_outside_circle_domain)

    # time_text = f"t = {1000 * t_eval_points_array[time_index]:4.0f}ms"
    # mlab.text3d(0, 0, 0, time_text, scale=(0.05, 0.05, 0.05), color=(0, 0, 0))

    mlab.close(all=True)
    # input()
    del t_eval, x_eval, y_eval, u_nn_pred, u_nn_plot_matrix, desired_function_at_final_time_plot_matrix


def _set_right_axis_limits_when_using_normal_scale(axis, loss_history):
    min_loss_value = min(loss_history)
    max_loss_value = max(loss_history)
    height_of_the_graph = max_loss_value - min_loss_value

    multiplier_for_the_lowest_loss_history_value_to_display = 0.1
    lowest_loss_history_value_to_display = min_loss_value - multiplier_for_the_lowest_loss_history_value_to_display * height_of_the_graph

    # print()
    # print(f"loss_name: {loss_name}")
    # print(f"multiplier_for_the_lowest_loss_history_value_to_display: {multiplier_for_the_lowest_loss_history_value_to_display}")
    # print(f"min_loss_value: {min_loss_value}")
    # print(f"lowest_loss_history_value_to_display: {lowest_loss_history_value_to_display}")

    multiplier_for_the_highest_loss_history_value_to_display = 0.1
    highest_loss_history_value_to_display = max_loss_value + multiplier_for_the_highest_loss_history_value_to_display * height_of_the_graph

    axis.set_ylim(lowest_loss_history_value_to_display,
                  highest_loss_history_value_to_display)


def _set_right_axis_limits_when_using_log_scale(axis, loss_history):
    #   Sometimes the plot of loss history showed some extremely small values that weren't in the actual values
    # which cause the whole graph of the values to be at the top of the figure and huge part of the figure was empty.
    # This change will ensure that the top value on y-axis is "c" times higher than the log of the highest value
    # in the loss history (the log is there because log scale is used), so "c" times higher than the highest value
    # that is actually plotted.
    # The "c" refers to the multipliers above.
    # Similar scaling is done to the lowest value.
    # Because of the large empty area, the top and bottom y values need to be limited. But just limiting them to max and min
    # of the plotted values would make the graph too "squeezed" in the figure, so we scale the values first so that there is
    # a little bit of room above and below the actual graph to make it look better.

    min_loss_value_on_logged_scale = math.log10(min(loss_history))
    max_loss_value_on_logged_scale = math.log10(max(loss_history))
    height_of_the_graph_on_logged_scale = max_loss_value_on_logged_scale - min_loss_value_on_logged_scale

    multiplier_for_the_lowest_loss_history_value_to_display = 0.1
    lowest_loss_history_value_to_be_displayed_evaluated_on_logged_scale = \
        min_loss_value_on_logged_scale - multiplier_for_the_lowest_loss_history_value_to_display * height_of_the_graph_on_logged_scale
    lowest_loss_history_value_to_display = 10 ** lowest_loss_history_value_to_be_displayed_evaluated_on_logged_scale

    # print()
    # print(f"loss_name: {loss_name}")
    # print(f"multiplier_for_the_lowest_loss_history_value_to_display: {multiplier_for_the_lowest_loss_history_value_to_display}")
    # print(f"min_loss_value: {min_loss_value_on_logged_scale}")
    # print(f"lowest_loss_history_value_to_be_displayed_evaluated_on_logged_scale: {lowest_loss_history_value_to_be_displayed_evaluated_on_logged_scale}")
    # print(f"lowest_loss_history_value_to_display: {lowest_loss_history_value_to_display}")

    multiplier_for_the_highest_loss_history_value_to_display = 0.1
    highest_loss_history_value_to_be_displayed_evaluated_on_logged_scale = \
        max_loss_value_on_logged_scale + multiplier_for_the_highest_loss_history_value_to_display * height_of_the_graph_on_logged_scale
    highest_loss_history_value_to_display = 10 ** highest_loss_history_value_to_be_displayed_evaluated_on_logged_scale

    axis.set_ylim(lowest_loss_history_value_to_display,
                  highest_loss_history_value_to_display)


def plot_and_save_loss_histories(training_setting_dictionary, data):
    dict_with_loss_histories = data["loss_history_dict"]
    for loss_name, loss_history in dict_with_loss_histories.items():
        print(f"Saving plot of history of {loss_name}.")

        batch_indicies = range(1, len(loss_history) + 1)

        plt.figure(loss_name)

        plt.xlabel("číslo dávky")
        plt.axhline(xmin=0.01, xmax=0.99, color="k", linestyle="-")

        plt.plot(batch_indicies, loss_history, "-b", label=f"{loss_name} values")

        plt.ylabel(f"{loss_name} hodnota")
        if "q_nn cooling soft penalty" in loss_name:
            _set_right_axis_limits_when_using_normal_scale(plt.gca(), loss_history)
        else:
            plt.yscale("log")
            _set_right_axis_limits_when_using_log_scale(plt.gca(), loss_history)

        plt.legend()

        save_dir = training_setting_dictionary["save_dir_for_current_specific_training_setting"]
        save_file = os.path.join(save_dir, f"{loss_name} history" + ".pdf")
        plt.savefig(save_file, bbox_inches="tight")

    plt.close("all")


def _norm_name_is_of_the_desired_norm_type(norm_name, norm_type):
    either_only_norm_name_or_norm_type_contains_the_word_weighted = ("weighted" in norm_name and "weighted" not in norm_type) or \
        ("weighted" not in norm_name and "weighted" in norm_type)

    if (norm_type in norm_name or "cost_functional" in norm_name) and not either_only_norm_name_or_norm_type_contains_the_word_weighted:
        return True
    else:
        return False

    raise ValueError(f"Couldn't determine, whether norm name {norm_name} is of desired norm type ({norm_type} or cost functional "
                     f"and both have to be either weighted or not weighted).")


def plot_and_save_norm_values_after_training_for_single_norm_type(training_setting_dictionary, data, norm_type: str):
    """
    ARGS:
        norm_type: "L1", "L2", "max", "weighted_L1", "weighted_L2" or "weighted_max"
    .
    """
    allowed_norm_type_values = ["L1", "L2", "max", "weighted_L1", "weighted_L2", "weighted_max"]
    if norm_type not in allowed_norm_type_values:
        raise ValueError(f'Norm type {norm_type} is not allowed. Allowed values are "L1", "L2", "max", "weighted_L1", "weighted_L2" or "weighted_max".')

    fig, ax = plt.subplots()

    dictionary_with_norms_after_training = data["dictionary_with_norms_after_training"]

    dictionary_with_desired_norms = {
        norm_name: norm_value
        for norm_name, norm_value in dictionary_with_norms_after_training.items()
        if _norm_name_is_of_the_desired_norm_type(norm_name, norm_type)
    }

    for norm_name, norm_value in dictionary_with_desired_norms.items():
        if isinstance(norm_value, tf.Tensor):
            dictionary_with_desired_norms[norm_name] = norm_value.numpy()

        if "equation" in norm_name:
            ax.scatter(0.0, norm_value, color="blue", marker="*", label=norm_name)

        if "boundary" in norm_name:
            ax.scatter(0.0, norm_value, color="orange", marker="s", label=norm_name)

        if "init_cond" in norm_name:
            ax.scatter(0.0, norm_value, color="black", marker=".", label=norm_name)

        if "cost_functional" in norm_name:
            ax.scatter(0.0, norm_value, color="red", marker="D", label=norm_name)
    desired_norm_values = [norm_value for norm_value in dictionary_with_desired_norms.values()]
    desired_norm_values_formatted_as_string = [f"{norm_value:.3e}" for norm_value in desired_norm_values]

    ax.legend()

    ax.set_xticks([])
    ax.set_xlim(-1.0, 2.0)

    ax.set_yscale("log")
    ax.get_yaxis().set_major_locator(matplotlib.ticker.FixedLocator(desired_norm_values))
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FixedFormatter(desired_norm_values_formatted_as_string))
    ax.get_yaxis().set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylabel("Hodnota normy")

    ax.axhline(xmin=0.01, xmax=0.99, color="k", linestyle="-")

    save_dir = training_setting_dictionary["save_dir_for_current_specific_training_setting"]
    save_file = os.path.join(save_dir, f"{norm_type}_norm and cost functional values" + ".pdf")
    plt.savefig(save_file, bbox_inches="tight")

    return


def plot_and_save_norm_values_after_training(training_setting_dictionary, data):
    plot_and_save_norm_values_after_training_for_single_norm_type(training_setting_dictionary, data, "L1")
    plot_and_save_norm_values_after_training_for_single_norm_type(training_setting_dictionary, data, "L2")
    plot_and_save_norm_values_after_training_for_single_norm_type(training_setting_dictionary, data, "max")
    plot_and_save_norm_values_after_training_for_single_norm_type(training_setting_dictionary, data, "weighted_L1")
    plot_and_save_norm_values_after_training_for_single_norm_type(training_setting_dictionary, data, "weighted_L2")
    plot_and_save_norm_values_after_training_for_single_norm_type(training_setting_dictionary, data, "weighted_max")
    plt.close("all")


def plot_and_save_figs(training_setting_dictionary, data):
    plot_and_save_fig(training_setting_dictionary, data, plot_relative_nn_error_from_fitted_function)
    plot_and_save_fig(training_setting_dictionary, data, plot_absolute_nn_error_from_fitted_function)
    plot_and_save_fig(training_setting_dictionary, data, plot_nn_result_and_fitted_function)
    plot_and_save_fig(training_setting_dictionary, data, plot_q_nn)
    plot_and_save_loss_histories(training_setting_dictionary, data)
    plot_and_save_norm_values_after_training(training_setting_dictionary, data)
    plt.close("all")


def plot_and_save_norm_values_after_all_cost_functional_coeficients_for_single_norm_type(
    training_setting_dictionary,
    data,
    norm_type: str
):
    """
    ARGS:
        norm_type: "L1", "L2", "max", "weighted_L1", "weighted_L2" or "weighted_max"
    .
    """
    allowed_norm_type_values = ["L1", "L2", "max", "weighted_L1", "weighted_L2", "weighted_max"]
    if norm_type not in allowed_norm_type_values:
        raise ValueError(f'Norm type {norm_type} is not allowed. Allowed values are "L1", "L2", "max", "weighted_L1", "weighted_L2" or "weighted_max".')

    fig, ax = plt.subplots()

    dictionary_with_lists_of_cost_functional_values_for_given_norms = {}
    dictionary_with_norms_after_training_for_different_cost_functional_coeficients =\
        data["dictionary_with_norms_after_training_for_different_cost_functional_coeficients"]
    for dictionary_with_norms_after_training_for_given_cost_functional_coeficient \
            in dictionary_with_norms_after_training_for_different_cost_functional_coeficients.values():

        dictionary_with_desired_norms_for_given_cost_functional_coeficient = {
            norm_name: norm_value
            for norm_name, norm_value in dictionary_with_norms_after_training_for_given_cost_functional_coeficient.items()
            if _norm_name_is_of_the_desired_norm_type(norm_name, norm_type)
        }

        for norm_name, norm_value in dictionary_with_desired_norms_for_given_cost_functional_coeficient.items():
            if isinstance(norm_value, tf.Tensor):
                norm_value = norm_value.numpy()
            if norm_name not in dictionary_with_lists_of_cost_functional_values_for_given_norms.keys():
                dictionary_with_lists_of_cost_functional_values_for_given_norms[norm_name] = []
            dictionary_with_lists_of_cost_functional_values_for_given_norms[norm_name].append(norm_value)

    for norm_name, list_with_norm_values in dictionary_with_lists_of_cost_functional_values_for_given_norms.items():
        tick_placement_values_for_cost_functional_coeficient_values = list(range(len(list_with_norm_values)))
        if "equation" in norm_name:
            ax.scatter(tick_placement_values_for_cost_functional_coeficient_values, list_with_norm_values, color="blue", marker="*", label=norm_name)

        if "boundary" in norm_name:
            ax.scatter(tick_placement_values_for_cost_functional_coeficient_values, list_with_norm_values, color="orange", marker="s", label=norm_name)

        if "init_cond" in norm_name:
            ax.scatter(tick_placement_values_for_cost_functional_coeficient_values, list_with_norm_values, color="black", marker=".", label=norm_name)

        if "cost_functional" in norm_name:
            ax.scatter(tick_placement_values_for_cost_functional_coeficient_values, list_with_norm_values, color="red", marker="D", label=norm_name)

    ax.legend()

    cost_functional_coeficient_values = list(dictionary_with_norms_after_training_for_different_cost_functional_coeficients.keys())
    cost_functional_coeficient_values_formatted_as_string = [f"{coef_value}" for coef_value in cost_functional_coeficient_values]
    tick_placement_values_for_cost_functional_coeficient_values = list(range(len(cost_functional_coeficient_values)))
    ax.get_xaxis().set_minor_locator(matplotlib.ticker.NullLocator())
    ax.get_xaxis().set_major_locator(matplotlib.ticker.FixedLocator(tick_placement_values_for_cost_functional_coeficient_values))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FixedFormatter(cost_functional_coeficient_values_formatted_as_string))
    ax.set_xlabel("Koeficient členu ztrátového funkcionálu")

    ax.axhline(xmin=0.01, xmax=0.99, color="k", linestyle="-")

    ax.set_yscale("log")

    ax.set_ylabel("Norm value")

    save_dir = training_setting_dictionary["save_dir_for_current_general_training_setting"]
    save_file = os.path.join(save_dir, f"{norm_type}_norm and cost functional values for different cost functional coeficients" + ".pdf")
    plt.savefig(save_file, bbox_inches="tight")

    return


def plot_and_save_norm_values_after_all_cost_functional_coeficients(training_setting_dictionary, data):
    plot_and_save_norm_values_after_all_cost_functional_coeficients_for_single_norm_type(training_setting_dictionary, data, "L1")
    plot_and_save_norm_values_after_all_cost_functional_coeficients_for_single_norm_type(training_setting_dictionary, data, "L2")
    plot_and_save_norm_values_after_all_cost_functional_coeficients_for_single_norm_type(training_setting_dictionary, data, "max")
    plot_and_save_norm_values_after_all_cost_functional_coeficients_for_single_norm_type(training_setting_dictionary, data, "weighted_L1")
    plot_and_save_norm_values_after_all_cost_functional_coeficients_for_single_norm_type(training_setting_dictionary, data, "weighted_L2")
    plot_and_save_norm_values_after_all_cost_functional_coeficients_for_single_norm_type(training_setting_dictionary, data, "weighted_max")
    plt.close("all")
