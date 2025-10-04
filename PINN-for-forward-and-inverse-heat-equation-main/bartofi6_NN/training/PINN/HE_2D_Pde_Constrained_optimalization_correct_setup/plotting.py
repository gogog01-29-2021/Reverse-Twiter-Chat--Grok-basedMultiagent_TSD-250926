import tensorflow as tf
from mayavi import mlab
import os
from ..HE_2D_Pde_Constrained_optimalization_on_circle import plot_nn_result_and_fitted_function, \
                                                             plot_absolute_nn_error_from_fitted_function, \
                                                             plot_relative_nn_error_from_fitted_function, \
                                                             plot_and_save_loss_histories, \
                                                             plot_and_save_norm_values_after_training, \
                                                             plot_and_save_norm_values_after_all_cost_functional_coeficients


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
    q_nn_surf = mlab.surf(X_plot, Y_plot, q_values_plot, colormap="gnuplot")

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
    Zernike_polynomial = data["desired_function_at_final_time"]
    circle_radius = data["circle_radius"]
    circle_center_in_xy = data["circle_center_in_xy"]

    mlab.figure(size=(800, 600), bgcolor=(0.3, 0.3, 0.3))

    x_eval = tf.reshape(X_plot, shape=-1)
    y_eval = tf.reshape(Y_plot, shape=-1)
    t_eval = tf.constant(data["t_stop"], shape=x_eval.shape, dtype=training_setting_dictionary["dtype"])

    u_nn_pred = u_nn(tf.stack([t_eval, x_eval, y_eval], axis=-1), training=False)
    u_nn_plot_matrix = tf.reshape(u_nn_pred, shape=X_plot.shape)

    q_nn_values = q_nn(tf.stack([t_eval, x_eval, y_eval], axis=-1), training=False)
    q_nn_plot_matrix = tf.reshape(q_nn_values, shape=X_plot.shape)

    x_eval_on_unit_circle = (x_eval - circle_center_in_xy[0]) / circle_radius
    y_eval_on_unit_circle = (y_eval - circle_center_in_xy[1]) / circle_radius

    radial_distance = tf.math.sqrt(x_eval_on_unit_circle**2 + y_eval_on_unit_circle**2)
    polar_angle = tf.math.atan2(y_eval_on_unit_circle, x_eval_on_unit_circle)

    Zernike_polynomial_values = Zernike_polynomial(radial_distance, polar_angle)
    Zernike_polynomial_plot_matrix = tf.reshape(Zernike_polynomial_values, shape=X_plot.shape)

    # TODO: Add plotting for different times, right now only the last time gets plotted.
    X_plot = X_plot.numpy()[:, :, -1]  # data["spacial_scale_coef"] * X_plot.numpy()[:, :, -1]
    Y_plot = Y_plot.numpy()[:, :, -1]  # data["spacial_scale_coef"] * Y_plot.numpy()[:, :, -1]
    # circle_radius = data["spacial_scale_coef"] * circle_radius
    # circle_center_in_xy[0] = data["spacial_scale_coef"] * circle_center_in_xy[0]
    # circle_center_in_xy[1] = data["spacial_scale_coef"] * circle_center_in_xy[1]
    u_nn_plot_matrix = u_nn_plot_matrix.numpy()[:, :, -1]  # data["u_scale_coef"] * u_nn_plot_matrix.numpy()[:, :, -1]
    q_nn_plot_matrix = q_nn_plot_matrix.numpy()[:, :, -1]  # data["q_scale_coef"] * q_nn_plot_matrix.numpy()[:, :, -1]
    Zernike_polynomial_plot_matrix = Zernike_polynomial_plot_matrix.numpy()[:, :, -1]  # data["polynomial_scale_coef"] * Zernike_polynomial_plot_matrix.numpy()[:, :, -1]

    mask_points_outside_circle_domain = (X_plot - circle_center_in_xy[0])**2 + (Y_plot - circle_center_in_xy[1])**2 > circle_radius**2

    # print(f"X_plot shape: {X_plot.shape}")
    # print(f"Y_plot shape: {Y_plot.shape}")
    # print(f"u_nn_plot_matrix shape: {u_nn_plot_matrix.shape}")
    # print(f"q_nn_plot_matrix shape: {q_nn_plot_matrix.shape}")
    # print(f"Zernike_polynomial_plot_matrix shape: {Zernike_polynomial_plot_matrix.shape}")
    # print(f"X_plot type: {type(X_plot)}")
    # print(f"Y_plot type: {type(Y_plot)}")
    # print(f"u_nn_plot_matrix type: {type(u_nn_plot_matrix)}")
    # print(f"q_nn_plot_matrix type: {type(q_nn_plot_matrix)}")
    # print(f"Zernike_polynomial_plot_matrix type: {type(Zernike_polynomial_plot_matrix)}")
    # TODO: Zkontrolovat velikosti, zda je mam dobře.
    function_with_plotting_procedure(training_setting_dictionary,
                                     data,
                                     X_plot,
                                     Y_plot,
                                     u_nn_plot_matrix,
                                     q_nn_plot_matrix,
                                     Zernike_polynomial_plot_matrix,
                                     mask_points_outside_circle_domain)

    # time_text = f"t = {1000 * t_eval_points_array[time_index]:4.0f}ms"
    # mlab.text3d(0, 0, 0, time_text, scale=(0.05, 0.05, 0.05), color=(0, 0, 0))

    mlab.close(all=True)
    del t_eval, x_eval, y_eval, u_nn_pred, u_nn_plot_matrix, Zernike_polynomial_plot_matrix


def plot_and_save_figs(training_setting_dictionary, data):
    plot_and_save_fig(training_setting_dictionary, data, plot_relative_nn_error_from_fitted_function)
    plot_and_save_fig(training_setting_dictionary, data, plot_absolute_nn_error_from_fitted_function)
    plot_and_save_fig(training_setting_dictionary, data, plot_nn_result_and_fitted_function)
    plot_and_save_fig(training_setting_dictionary, data, plot_q_nn)
    plot_and_save_loss_histories(training_setting_dictionary, data)
    plot_and_save_norm_values_after_training(training_setting_dictionary, data)
