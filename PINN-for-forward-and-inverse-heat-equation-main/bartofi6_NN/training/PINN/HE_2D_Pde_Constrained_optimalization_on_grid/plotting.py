import tensorflow as tf
from ....PINN.HE_2D_Pde_Constrained_optimalization_on_grid import *

import os

from mayavi import mlab
import matplotlib
from matplotlib import pyplot as plt


def plot_nn_result_and_fitted_function(
training_setting_dictionary,
data,
X_plot,
Y_plot,
u_pred_plot,
q_values_plot,
fitted_function_plot
):
    mlab.surf(X_plot, Y_plot, u_pred_plot, colormap="ocean")
    mlab.surf(X_plot, Y_plot, fitted_function_plot, colormap="gnuplot")
    
    mlab.axes(xlabel="x", ylabel="y", zlabel="u(T, x, y)")
    mlab.move(-1.0) # Move the camera a bit so everything fits within the png.

    save_dir = training_setting_dictionary["save_dir_for_current_specific_training_setting"]
    save_file = os.path.join(save_dir, "NN and fitted function values at final time" + ".png")
    mlab.savefig(save_file)


# Zde se zřejmě někdy dělí malými čísly (fitted_function_plot je občas okolo nuly) a to pak způsobí obrovský
#   error. Zkusit to nějak spravit, aby ty obrázky dávaly smysl třeba alespoň pro nenulové hodnoty fitted_function_plot.
def plot_relative_nn_error_from_fitted_function(
training_setting_dictionary,
data,
X_plot,
Y_plot,
u_pred_plot,
q_values_plot,
fitted_function_plot
):
    relative_nn_error = (u_pred_plot - fitted_function_plot) / fitted_function_plot
    mlab.surf(X_plot, Y_plot, relative_nn_error, colormap="gnuplot")
    
    mlab.axes(xlabel="x", ylabel="y", zlabel="relative error:\n(u_nn - u_f) / u_f")
    mlab.move(-1.0) # Move the camera a bit so everything fits within the png.

    save_dir = training_setting_dictionary["save_dir_for_current_specific_training_setting"]
    save_file = os.path.join(save_dir, "relative error: (u_nn - u_f) : u_f" + ".png")
    mlab.savefig(save_file)


def plot_absolute_nn_error_from_fitted_function(
training_setting_dictionary,
data,
X_plot,
Y_plot,
u_pred_plot,
q_values_plot,
fitted_function_plot
):
    absolute_nn_error = u_pred_plot - fitted_function_plot
    test_plot = tf.zeros_like(u_pred_plot)
    mlab.surf(X_plot, Y_plot, absolute_nn_error, colormap="gnuplot")
    
    mlab.axes(xlabel="x", ylabel="y", zlabel="absolute error:\nu_nn - u_f")
    mlab.move(-1.0) # Move the camera a bit so everything fits within the png.

    save_dir = training_setting_dictionary["save_dir_for_current_specific_training_setting"]
    save_file = os.path.join(save_dir, "absolute error:u_nn - u_f" + ".png")
    mlab.savefig(save_file)


def plot_q_nn(
training_setting_dictionary,
data,
X_plot,
Y_plot,
u_pred_plot,
q_values_plot,
fitted_function_plot
):
    mlab.surf(X_plot, Y_plot, q_values_plot, colormap="gnuplot")
    
    mlab.axes(xlabel="x", ylabel="y", zlabel="q(T, x, y)")
    mlab.move(-1.0) # Move the camera a bit so everything fits within the png.

    save_dir = training_setting_dictionary["save_dir_for_current_specific_training_setting"]
    save_file = os.path.join(save_dir, "q_nn function values at final time" + ".png")
    mlab.savefig(save_file)


def plot_and_save_fig(training_setting_dictionary, data, function_with_plotting_procedure):
    X_plot = data["X_plot"]
    Y_plot = data["Y_plot"]
    u_nn = data["u_nn"]
    q_nn = data["q_nn"]
    desired_function_at_final_time = data["desired_function_at_final_time"]

    mlab.figure(size=(800, 600), bgcolor=(0.3, 0.3, 0.3))

    x_eval = tf.reshape(X_plot, shape=-1)
    y_eval = tf.reshape(Y_plot, shape=-1)
    t_eval = tf.constant(data["t_stop"], shape=x_eval.shape)

    u_nn_pred = u_nn(tf.stack([t_eval, x_eval, y_eval], axis=-1), training=False)
    u_nn_plot_matrix = tf.reshape(u_nn_pred, shape=X_plot.shape)
    q_nn_values = q_nn(tf.stack([t_eval, x_eval, y_eval], axis=-1), training=False)
    q_nn_plot_matrix = tf.reshape(q_nn_values, shape=X_plot.shape)

    desired_function_at_final_time_plot_matrix = desired_function_at_final_time(X_plot, Y_plot)

    # print(f"X_plot shape: {X_plot.shape}")
    # print(f"Y_plot shape: {Y_plot.shape}")
    # print(f"u_nn_plot_matrix shape: {u_nn_plot_matrix.shape}")
    # print(f"desired_function_at_final_time_plot_matrix shape: {desired_function_at_final_time_plot_matrix.shape}")
    # TODO: Zkontrolovat velikosti, zda je mam dobře.
    function_with_plotting_procedure(training_setting_dictionary,
                                     data,
                                     X_plot,
                                     Y_plot,
                                     u_nn_plot_matrix,
                                     q_nn_plot_matrix,
                                     desired_function_at_final_time_plot_matrix)
    # Without transpose, one must use mlab.mesh() to generate 3D plot, surf and contour_surf only works with transposed matricies for some reason.

    # time_text = f"t = {1000 * t_eval_points_array[time_index]:4.0f}ms"
    # mlab.text3d(0, 0, 0, time_text, scale=(0.05, 0.05, 0.05), color=(0, 0, 0))

    mlab.close(all=True)
    # input()
    del t_eval, x_eval, y_eval, u_nn_pred, u_nn_plot_matrix, desired_function_at_final_time_plot_matrix


def plot_and_save_loss_histories(training_setting_dictionary, data):
    dict_with_loss_histories = data["loss_history_dict"]
    for loss_name, loss_history in dict_with_loss_histories.items():
        batch_indicies = range(1, len(loss_history) + 1)
        
        plt.figure(loss_name)
        plt.plot(batch_indicies, loss_history, "-b", label=f"{loss_name} values")
        
        plt.xlabel("batch")
        plt.axhline(xmin=0.01, xmax=0.99, color="k", linestyle="-")
        
        plt.ylabel(f"{loss_name} value")
        plt.yscale("log")

        plt.legend()
        
        save_dir = training_setting_dictionary["save_dir_for_current_specific_training_setting"]
        save_file = os.path.join(save_dir, f"{loss_name} history" + ".pdf")
        plt.savefig(save_file, bbox_inches="tight")
    
    plt.close("all")


def plot_and_save_norm_values_after_training_for_single_norm_type(training_setting_dictionary, data, norm_type: str):
    """
    ARGS:
        norm_type: "L1", "L2" or "max"
    .
    """
    allowed_norm_type_values = ["L1", "L2", "max"]
    if not norm_type in allowed_norm_type_values:
        raise(ValueError(f'Norm type {norm_type} is not allowed. Allowed values are "L1", "L2" or "max".'))

    fig, ax = plt.subplots()

    dictionary_with_norms_after_training = data["dictionary_with_norms_after_training"]

    dictionary_with_desired_weighted_norms = {
        norm_name: norm_value for norm_name, norm_value in dictionary_with_norms_after_training.items()
        if norm_type in norm_name or "cost_functional" in norm_name
    }
        

    for norm_name, norm_value in dictionary_with_desired_weighted_norms.items():
        if isinstance(norm_value, tf.Tensor):
            dictionary_with_desired_weighted_norms[norm_name] = norm_value.numpy()

        if "equation" in norm_name:
            ax.scatter(0.0, norm_value, color="blue", marker="*", label=norm_name)

        if "boundary" in norm_name:
            ax.scatter(0.0, norm_value, color="orange", marker="s", label=norm_name)

        if "init_cond" in norm_name:
            ax.scatter(0.0, norm_value, color="black", marker=".", label=norm_name)

        if "cost_functional" in norm_name:
            ax.scatter(0.0, norm_value, color="red", marker="D", label=norm_name)
    desired_norm_values = [norm_value for norm_value in dictionary_with_desired_weighted_norms.values()]
    desired_norm_values_formatted_as_string = [f"{norm_value:.3e}" for norm_value in desired_norm_values]

    ax.legend()

    ax.set_xticks([])
    ax.set_xlim(-1.0, 2.0)

    ax.set_yscale("log")
    ax.get_yaxis().set_major_locator(matplotlib.ticker.FixedLocator(desired_norm_values))
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FixedFormatter(desired_norm_values_formatted_as_string))
    ax.get_yaxis().set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylabel("Norm value")
    
    ax.axhline(xmin=0.01, xmax=0.99, color="k", linestyle="-")

    save_dir = training_setting_dictionary["save_dir_for_current_specific_training_setting"]
    save_file = os.path.join(save_dir, f"weighted {norm_type} norm and cost functional values" + ".pdf")
    plt.savefig(save_file, bbox_inches="tight")

    return


def plot_and_save_norm_values_after_training(training_setting_dictionary, data):
    plot_and_save_norm_values_after_training_for_single_norm_type(training_setting_dictionary, data, "L1")
    plot_and_save_norm_values_after_training_for_single_norm_type(training_setting_dictionary, data, "L2")
    plot_and_save_norm_values_after_training_for_single_norm_type(training_setting_dictionary, data, "max")
    plt.close("all")



def plot_and_save_figs(training_setting_dictionary, data):
    plot_and_save_fig(training_setting_dictionary, data, plot_relative_nn_error_from_fitted_function)
    plot_and_save_fig(training_setting_dictionary, data, plot_absolute_nn_error_from_fitted_function)
    plot_and_save_fig(training_setting_dictionary, data, plot_nn_result_and_fitted_function)
    plot_and_save_fig(training_setting_dictionary, data, plot_q_nn)
    plot_and_save_loss_histories(training_setting_dictionary, data)
    plot_and_save_norm_values_after_training(training_setting_dictionary, data)

def plot_and_save_norm_values_after_all_cost_functional_coeficients_for_single_norm_type(
training_setting_dictionary,
data,
norm_type: str
):
    """
    ARGS:
        norm_type: "L1", "L2" or "max"
    .
    """
    allowed_norm_type_values = ["L1", "L2", "max"]
    if not norm_type in allowed_norm_type_values:
        raise(ValueError(f'Norm type {norm_type} is not allowed. Allowed values are "L1", "L2" or "max".'))

    fig, ax = plt.subplots()

    dictionary_with_lists_of_cost_functional_values_for_given_norms = {}
    dictionary_with_norms_after_training_for_different_cost_functional_coeficients =\
        data["dictionary_with_norms_after_training_for_different_cost_functional_coeficients"]
    for dictionary_with_norms_after_training_for_given_cost_functional_coeficient \
    in dictionary_with_norms_after_training_for_different_cost_functional_coeficients.values():

        dictionary_with_desired_weighted_norms_for_given_cost_functional_coeficient = {
            norm_name: norm_value
            for norm_name, norm_value in dictionary_with_norms_after_training_for_given_cost_functional_coeficient.items()
            if norm_type in norm_name or "cost_functional" in norm_name
        }

        for norm_name, norm_value in dictionary_with_desired_weighted_norms_for_given_cost_functional_coeficient.items():
            if isinstance(norm_value, tf.Tensor):
                norm_value = norm_value.numpy()
            if norm_name not in dictionary_with_lists_of_cost_functional_values_for_given_norms.keys():
                dictionary_with_lists_of_cost_functional_values_for_given_norms[norm_name] = []
            dictionary_with_lists_of_cost_functional_values_for_given_norms[norm_name].append(norm_value)

    for norm_name, list_with_norm_values in dictionary_with_lists_of_cost_functional_values_for_given_norms.items():
        placement_values_for_cost_functional_coeficient_values = list(range(len(list_with_norm_values)))
        if "equation" in norm_name:
            ax.scatter(placement_values_for_cost_functional_coeficient_values, list_with_norm_values, color="blue", marker="*", label=norm_name)

        if "boundary" in norm_name:
            ax.scatter(placement_values_for_cost_functional_coeficient_values, list_with_norm_values, color="orange", marker="s", label=norm_name)

        if "init_cond" in norm_name:
            ax.scatter(placement_values_for_cost_functional_coeficient_values, list_with_norm_values, color="black", marker=".", label=norm_name)

        if "cost_functional" in norm_name:
            ax.scatter(placement_values_for_cost_functional_coeficient_values, list_with_norm_values, color="red", marker="D", label=norm_name)

    ax.legend()

    cost_functional_coeficient_values = list(dictionary_with_norms_after_training_for_different_cost_functional_coeficients.keys())
    cost_functional_coeficient_values_formatted_as_string = [f"{coef_value}" for coef_value in cost_functional_coeficient_values]
    tick_placement_values_for_cost_functional_coeficient_values = list(range(len(cost_functional_coeficient_values)))
    ax.get_xaxis().set_minor_locator(matplotlib.ticker.NullLocator())
    ax.get_xaxis().set_major_locator(matplotlib.ticker.FixedLocator(tick_placement_values_for_cost_functional_coeficient_values))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FixedFormatter(cost_functional_coeficient_values_formatted_as_string))
    ax.set_xlabel("Cost functional loss coeficient")
    
    ax.axhline(xmin=0.01, xmax=0.99, color="k", linestyle="-")

    ax.set_yscale("log")

    ax.set_ylabel("Norm value")

    save_dir = training_setting_dictionary["save_dir_for_current_general_training_setting"]
    save_file = os.path.join(save_dir, f"weighted {norm_type} norm and cost functional values for different cost functional coeficients" + ".pdf")
    plt.savefig(save_file, bbox_inches="tight")

    return

def plot_and_save_norm_values_after_all_cost_functional_coeficients(training_setting_dictionary, data):
    plot_and_save_norm_values_after_all_cost_functional_coeficients_for_single_norm_type(training_setting_dictionary, data, "L1")
    plot_and_save_norm_values_after_all_cost_functional_coeficients_for_single_norm_type(training_setting_dictionary, data, "L2")
    plot_and_save_norm_values_after_all_cost_functional_coeficients_for_single_norm_type(training_setting_dictionary, data, "max")
