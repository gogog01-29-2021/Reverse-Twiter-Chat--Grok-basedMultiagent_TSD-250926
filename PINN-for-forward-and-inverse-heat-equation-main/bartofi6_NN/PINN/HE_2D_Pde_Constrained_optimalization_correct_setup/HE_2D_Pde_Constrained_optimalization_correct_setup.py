import tensorflow as tf
import time
from ..HE_2D_Pde_Constrained_optimalization_on_grid.HE_2D_Pde_Constrained_optimalization_on_grid import \
    generate_training_points, generate_eq_boundary_and_intial_points_tuples, _calculate_equation_residual, \
    _calculate_initial_condition_residual, _calculate_boundary_residual, shuffle_points, generate_training_points_slices, \
    print_epoch_results
from ..HE_2D_Pde_Constrained_optimalization_on_circle import calculate_functional_loss
from ...training import tfprint_tensor_shape_and_type, tfprint_tensor_values
from ...training.PINN.general import L1_norm, L2_norm, max_norm

tfprint_tensor_shapes_and_types_for_debug = False
tfprint_tensor_values_for_debug = False


tfprint_tensor_shapes_and_types_for_function_generate_points_for_evaluation_of_function_fit_integral = False
tfprint_tensor_values_for_function_generate_points_for_evaluation_of_function_fit_integral = False


def generate_points_for_evaluation_of_function_fit_integral(
    t_final,
    x_start,
    x_stop,
    num_of_x_points,
    y_start,
    y_stop,
    num_of_y_points,
    circle_radius,
    circle_center_in_xy,
    dtype,
):
    x = tf.linspace(tf.constant(x_start, dtype), x_stop, num_of_x_points)
    y = tf.linspace(tf.constant(y_start, dtype), y_stop, num_of_y_points)

    X, Y = tf.meshgrid(x, y, indexing="ij")

    x_evaluation_points = tf.reshape(X, [-1])
    y_evaluation_points = tf.reshape(Y, [-1])
    t_evaluation_points = tf.constant(t_final, dtype, tf.shape(x_evaluation_points))

    boolean_mask_to_keep_points_in_circle = (x_evaluation_points - circle_center_in_xy[0])**2 + (y_evaluation_points - circle_center_in_xy[1])**2 \
                                             <= circle_radius**2

    if tfprint_tensor_shapes_and_types_for_debug and \
    tfprint_tensor_shapes_and_types_for_function_generate_points_for_evaluation_of_function_fit_integral:
        tfprint_tensor_shape_and_type("X", X)
        tfprint_tensor_shape_and_type("Y", Y)

        tfprint_tensor_shape_and_type("x_evaluation_points before boolean mask", x_evaluation_points)
        tfprint_tensor_shape_and_type("y_evaluation_points before boolean mask", y_evaluation_points)
        tfprint_tensor_shape_and_type("t_evaluation_points before boolean mask", t_evaluation_points)

        tfprint_tensor_shape_and_type("boolean mask to keep points in circle", boolean_mask_to_keep_points_in_circle)

    if tfprint_tensor_values_for_debug and \
    tfprint_tensor_values_for_function_generate_points_for_evaluation_of_function_fit_integral:
        tfprint_tensor_values("X", X)
        tfprint_tensor_values("Y", Y)

        tfprint_tensor_values("x_evaluation_points before boolean mask", x_evaluation_points)
        tfprint_tensor_values("y_evaluation_points before boolean mask", y_evaluation_points)
        tfprint_tensor_values("t_evaluation_points before boolean mask", t_evaluation_points)

        tfprint_tensor_values("boolean mask to keep points in circle", boolean_mask_to_keep_points_in_circle)

    x_evaluation_points = x_evaluation_points[boolean_mask_to_keep_points_in_circle]
    y_evaluation_points = y_evaluation_points[boolean_mask_to_keep_points_in_circle]
    t_evaluation_points = t_evaluation_points[boolean_mask_to_keep_points_in_circle]

    txy_evaluation_points = tf.stack([t_evaluation_points, x_evaluation_points, y_evaluation_points,], axis=1)


    x_evaluation_points_on_unit_circle = (x_evaluation_points - circle_center_in_xy[0]) / circle_radius
    y_evaluation_points_on_unit_circle = (y_evaluation_points - circle_center_in_xy[1]) / circle_radius

    radial_distance = tf.math.sqrt(x_evaluation_points_on_unit_circle**2 + y_evaluation_points_on_unit_circle**2)
    polar_angle = tf.math.atan2(y_evaluation_points_on_unit_circle, x_evaluation_points_on_unit_circle)


    x_step = X[1, 0] - X[0, 0]
    y_step = Y[0, 1] - Y[0, 0]
    xy_grid_square_area = x_step * y_step

    if tfprint_tensor_shapes_and_types_for_debug and \
    tfprint_tensor_shapes_and_types_for_function_generate_points_for_evaluation_of_function_fit_integral:
        tfprint_tensor_shape_and_type("x_evaluation_points", x_evaluation_points)
        tfprint_tensor_shape_and_type("y_evaluation_points", y_evaluation_points)
        tfprint_tensor_shape_and_type("t_evaluation_points", t_evaluation_points)
        tfprint_tensor_shape_and_type("txy_evaluation_points", txy_evaluation_points)
        tfprint_tensor_shape_and_type("x_evaluation_points_on_unit_circle", x_evaluation_points_on_unit_circle)
        tfprint_tensor_shape_and_type("y_evaluation_points_on_unit_circle", y_evaluation_points_on_unit_circle)
        tfprint_tensor_shape_and_type("radial_distance", radial_distance)
        tfprint_tensor_shape_and_type("polar_angle", polar_angle)
        tfprint_tensor_shape_and_type("x_step", x_step)
        tfprint_tensor_shape_and_type("y_step", y_step)
        tfprint_tensor_shape_and_type("xy_grid_square_area", xy_grid_square_area)

    if tfprint_tensor_values_for_debug and \
    tfprint_tensor_values_for_function_generate_points_for_evaluation_of_function_fit_integral:
        tfprint_tensor_values("x_evaluation_points", x_evaluation_points)
        tfprint_tensor_values("y_evaluation_points", y_evaluation_points)
        tfprint_tensor_values("t_evaluation_points", t_evaluation_points)
        tfprint_tensor_values("txy_evaluation_points", txy_evaluation_points)
        tfprint_tensor_values("x_evaluation_points_on_unit_circle", x_evaluation_points_on_unit_circle)
        tfprint_tensor_values("y_evaluation_points_on_unit_circle", y_evaluation_points_on_unit_circle)
        tfprint_tensor_values("radial_distance", radial_distance)
        tfprint_tensor_values("polar_angle", polar_angle)
        tfprint_tensor_values("x_step", x_step)
        tfprint_tensor_values("y_step", y_step)
        tfprint_tensor_values("xy_grid_square_area", xy_grid_square_area)

    return txy_evaluation_points, radial_distance, polar_angle, xy_grid_square_area


tfprint_tensor_shapes_and_types_for_function_calculate_approximation_of_function_fit_integral = False
tfprint_tensor_values_for_function_calculate_approximation_of_function_fit_integral = False


def calculate_approximation_of_function_fit_integral(
    u_nn,
    desired_Zernike_polynomial_at_final_time,
    txy_evaluation_points_for_neural_network,
    radial_distance,
    polar_angle,
    xy_grid_square_area,
):
    u_Txy = u_nn(txy_evaluation_points_for_neural_network, training=True)[:, 0]
    # Make it into rank 1 tensor so that shape is same as desired_function_at_final_time_at_xy.

    desired_values_at_final_time_at_xy = desired_Zernike_polynomial_at_final_time(radial_distance, polar_angle)
    if isinstance(desired_values_at_final_time_at_xy, (float, int)):
        desired_values_at_final_time_at_xy = tf.fill(u_Txy.shape, tf.cast(desired_values_at_final_time_at_xy, u_Txy.dtype))

    approximation_of_function_fit_integral = tf.math.reduce_sum(tf.math.squared_difference(u_Txy, desired_values_at_final_time_at_xy))
    approximation_of_function_fit_integral = xy_grid_square_area * approximation_of_function_fit_integral

    if tfprint_tensor_shapes_and_types_for_debug and \
    tfprint_tensor_shapes_and_types_for_function_calculate_approximation_of_function_fit_integral:
        tfprint_tensor_shape_and_type("u_Txy", u_Txy)
        tfprint_tensor_shape_and_type("desired_values_at_final_time_at_xy", desired_values_at_final_time_at_xy)
        tfprint_tensor_shape_and_type("approximation_of_function_fit_integral", approximation_of_function_fit_integral)

    if tfprint_tensor_values_for_debug and \
    tfprint_tensor_values_for_function_calculate_approximation_of_function_fit_integral:
        tfprint_tensor_values("u_Txy", u_Txy)
        tfprint_tensor_values("desired_values_at_final_time_at_xy", desired_values_at_final_time_at_xy)
        tfprint_tensor_values("approximation_of_function_fit_integral", approximation_of_function_fit_integral)

    return approximation_of_function_fit_integral


def shuffle_tensors_with_same_length(
    list_with_tensors_of_same_length
):
    first_tensor = list_with_tensors_of_same_length[0]
    random_indicies = tf.random.shuffle(tf.range(tf.size(first_tensor)))
    list_with_shuffled_tensors = []

    for tensor in list_with_tensors_of_same_length:
        list_with_shuffled_tensors.append(tf.gather(tensor, random_indicies))

    return list_with_shuffled_tensors


@tf.function(jit_compile=True)
def shuffle_points_for_integral_evaluation(
    txy_points_for_integral_evaluation,
    radial_distance_for_integral_evaluation,
    polar_angle_for_integral_evaluation,
):
    print("Tracing shuffle_points_for_integral_evaluation")
    list_with_tensors_for_integral_evaluation = [
        txy_points_for_integral_evaluation[:, 0],
        txy_points_for_integral_evaluation[:, 1],
        txy_points_for_integral_evaluation[:, 2],
        radial_distance_for_integral_evaluation,
        polar_angle_for_integral_evaluation,
    ]
    list_with_shuffled_tensors_for_integral_evaluation = shuffle_tensors_with_same_length(list_with_tensors_for_integral_evaluation)
    shuffled_txy_points_for_integral_evaluation = tf.stack([list_with_shuffled_tensors_for_integral_evaluation[0],
                                                            list_with_shuffled_tensors_for_integral_evaluation[1],
                                                            list_with_shuffled_tensors_for_integral_evaluation[2]],
                                                           axis=-1)
    shuffled_radial_distance_for_integral_evaluation = list_with_shuffled_tensors_for_integral_evaluation[3]
    shuffled_polar_angle_for_integral_evaluation = list_with_shuffled_tensors_for_integral_evaluation[4]

    return shuffled_txy_points_for_integral_evaluation, shuffled_radial_distance_for_integral_evaluation, shuffled_polar_angle_for_integral_evaluation


def _generate_slice_of_tensor(
    tensor,
    batch_index,
    num_of_batches,
):
    last_batch_index = num_of_batches - 1
    num_of_points_in_slice = tf.cast(tf.math.floor(tf.size(tensor) / num_of_batches), dtype=tf.int32)

    if batch_index != last_batch_index:
        tensor_slice = tf.slice(tensor, begin=[batch_index * num_of_points_in_slice], size=[num_of_points_in_slice])
    else:
        tensor_slice = tensor[last_batch_index * num_of_points_in_slice : ]

    return tensor_slice


def slice_tensors(
    list_with_tensors,
    batch_index,
    num_of_batches,
):
    list_with_sliced_tensors = [
        _generate_slice_of_tensor(tensor, batch_index, num_of_batches)
        for tensor in list_with_tensors
    ]

    return list_with_sliced_tensors


@tf.function(jit_compile=True)
def slice_points_for_integral_evaluation(
    txy_points_for_integral_evaluation,
    radial_distance_for_integral_evaluation,
    polar_angle_for_integral_evaluation,
    batch_index,
    num_of_batches
):
    print("Tracing slice_points_for_integral_evaluation")
    list_with_tensors_for_integral_evaluation = [
        txy_points_for_integral_evaluation[:, 0],
        txy_points_for_integral_evaluation[:, 1],
        txy_points_for_integral_evaluation[:, 2],
        radial_distance_for_integral_evaluation,
        polar_angle_for_integral_evaluation,
    ]
    list_with_sliced_tensors_for_integral_evaluation = slice_tensors(list_with_tensors_for_integral_evaluation, batch_index, num_of_batches)
    sliced_txy_points_for_integral_evaluation = tf.stack([list_with_sliced_tensors_for_integral_evaluation[0],
                                                          list_with_sliced_tensors_for_integral_evaluation[1],
                                                          list_with_sliced_tensors_for_integral_evaluation[2]],
                                                         axis=-1)
    sliced_radial_distance_for_integral_evaluation = list_with_sliced_tensors_for_integral_evaluation[3]
    sliced_polar_angle_for_integral_evaluation = list_with_sliced_tensors_for_integral_evaluation[4]

    return sliced_txy_points_for_integral_evaluation, sliced_radial_distance_for_integral_evaluation, sliced_polar_angle_for_integral_evaluation


def _calculate_q_nn_cooling_soft_penalty(
    q_nn,
    boundary_points_tuple,
):
    tuple_with_bottom_boundary_points, tuple_with_top_boundary_points, tuple_with_left_boundary_points, tuple_with_right_boundary_points = \
        boundary_points_tuple

    t, x, y = tuple_with_bottom_boundary_points

    remaining_tuples_with_boundary_points = [tuple_with_top_boundary_points, tuple_with_left_boundary_points, tuple_with_right_boundary_points]
    for tuple_with_boundary_points in remaining_tuples_with_boundary_points:
        t_b, x_b, y_b = tuple_with_boundary_points
        t = tf.concat([t, t_b], axis=0)
        x = tf.concat([x, x_b], axis=0)
        y = tf.concat([y, y_b], axis=0)

    q_nn_values = q_nn(tf.stack([t, x, y], axis=1), training=True)
    zero = tf.zeros_like(q_nn_values)
    only_negative_q_nn_values_or_zero = tf.math.minimum(q_nn_values, zero)
    return L1_norm(only_negative_q_nn_values_or_zero)


tfprint_tensor_shapes_and_types_for_function__calculate_q_nn_values_for_cooling_soft_penalty = False
tfprint_tensor_values_for_function__calculate_q_nn_values_for_cooling_soft_penalty = False


def _calculate_q_nn_values_for_cooling_soft_penalty(
    q_nn,
    boundary_points_tuple,
):
    tuple_with_bottom_boundary_points, tuple_with_top_boundary_points, tuple_with_left_boundary_points, tuple_with_right_boundary_points = \
        boundary_points_tuple

    t, x, y = tuple_with_bottom_boundary_points

    remaining_tuples_with_boundary_points = [tuple_with_top_boundary_points, tuple_with_left_boundary_points, tuple_with_right_boundary_points]
    for tuple_with_boundary_points in remaining_tuples_with_boundary_points:
        t_b, x_b, y_b = tuple_with_boundary_points
        t = tf.concat([t, t_b], axis=0)
        x = tf.concat([x, x_b], axis=0)
        y = tf.concat([y, y_b], axis=0)

    q_nn_values = q_nn(tf.stack([t, x, y], axis=1), training=True)
    q_nn_values = tf.squeeze(q_nn_values)
    zero = tf.zeros_like(q_nn_values)
    only_negative_q_nn_values_or_zero = tf.math.minimum(q_nn_values, zero)

    if tfprint_tensor_shapes_and_types_for_debug and tfprint_tensor_shapes_and_types_for_function__calculate_q_nn_values_for_cooling_soft_penalty:
        tfprint_tensor_shape_and_type("q_nn_values", q_nn_values)
        tfprint_tensor_shape_and_type("only_negative_q_nn_values_or_zero", only_negative_q_nn_values_or_zero)

    if tfprint_tensor_values_for_debug and tfprint_tensor_values_for_function__calculate_q_nn_values_for_cooling_soft_penalty:
        tfprint_tensor_values("q_nn_values", q_nn_values, -1)
        tfprint_tensor_values("only_negative_q_nn_values_or_zero", only_negative_q_nn_values_or_zero, -1)

    return only_negative_q_nn_values_or_zero


# @tf.function(jit_compile=True,reduce_retracing=True)
def train_step(
    u_nn,
    q_nn,
    equation_rhs_function,
    initial_condition_function,
    desired_function_at_final_time,
    heat_coef,
    alpha,
    eq_points_tuple,
    boundary_points_tuple,
    initial_points_tuple,
    txy_points_for_integral_evaluation,
    radial_distance_for_integral_evaluation,
    polar_angle_for_integral_evaluation,
    xy_grid_cell_area_for_integral_evaluation,
    boundary_condition_weight,
    initial_condition_weight,
    cost_functional_weight,
    q_nn_cooling_penalty_weight,
    optimizer,
    loss_fn,
    train_only_u_nn=False,
):
    print("Tracing train step.")
    # This is here just to catch if the function was retracing in every batch call.

    used_dtype = u_nn.layers[0].dtype

    with tf.GradientTape(persistent=True) as weights_update_tape:
        equation_residual, u_pred = _calculate_equation_residual(eq_points_tuple, u_nn, equation_rhs_function, heat_coef)
        equation_loss = loss_fn(equation_residual)

        initial_condition_residual = _calculate_initial_condition_residual(initial_points_tuple, u_nn, u_pred, initial_condition_function, heat_coef)
        init_cond_loss = loss_fn(initial_condition_residual)
        weighted_init_cond_loss = tf.cast(initial_condition_weight, used_dtype) * init_cond_loss

        boundary_residual = _calculate_boundary_residual(boundary_points_tuple, u_nn, q_nn)
        boundary_cond_loss = loss_fn(boundary_residual)
        weighted_boundary_cond_loss = tf.cast(boundary_condition_weight, used_dtype) * boundary_cond_loss

        cost_functional_loss = calculate_approximation_of_function_fit_integral(u_nn,
                                                                                desired_function_at_final_time,
                                                                                txy_points_for_integral_evaluation,
                                                                                radial_distance_for_integral_evaluation,
                                                                                polar_angle_for_integral_evaluation,
                                                                                xy_grid_cell_area_for_integral_evaluation)
        weighted_cost_functional_loss = tf.cast(cost_functional_weight, used_dtype) * cost_functional_loss

        q_nn_values_for_cooling_soft_penalty = _calculate_q_nn_values_for_cooling_soft_penalty(q_nn, boundary_points_tuple)
        q_nn_cooling_soft_penalty = L1_norm(q_nn_values_for_cooling_soft_penalty)
        weighted_q_nn_cooling_soft_penalty = tf.cast(q_nn_cooling_penalty_weight, used_dtype) * q_nn_cooling_soft_penalty

        weighted_loss = equation_loss + \
                        weighted_init_cond_loss + \
                        weighted_boundary_cond_loss + \
                        weighted_cost_functional_loss + \
                        weighted_q_nn_cooling_soft_penalty

    grads_u_nn = weights_update_tape.gradient(weighted_loss, u_nn.trainable_variables)
    u_nn.optimizer.apply_gradients(zip(grads_u_nn, u_nn.trainable_variables))

    if not train_only_u_nn:
        grads_q_nn = weights_update_tape.gradient(weighted_loss, q_nn.trainable_variables)
        q_nn.optimizer.apply_gradients(zip(grads_q_nn, q_nn.trainable_variables))


    loss_dict = {
        "equation loss": equation_loss,
        "boundary condition loss": boundary_cond_loss,
        "initial condition loss": init_cond_loss,
        "cost functional loss": cost_functional_loss,
        "q_nn cooling soft penalty": q_nn_cooling_soft_penalty,
        "weighted equation loss": equation_loss,
        #   Weight for equation loss is always 1.0,
        # but for easier visual comparing during training,
        # I include weighted equation loss in the dictionary as well.
        "weighted boundary condition loss": weighted_boundary_cond_loss,
        "weighted initial condition loss": weighted_init_cond_loss,
        "weighted cost functional loss": weighted_cost_functional_loss,
        "weighted q_nn cooling soft penalty": weighted_q_nn_cooling_soft_penalty,
    }

    return loss_dict


@tf.function(jit_compile=True, reduce_retracing=True)
def evaluate_norms_after_training(
    u_nn,
    q_nn,
    equation_rhs_function,
    initial_condition_function,
    desired_function_at_final_time,
    heat_coef,
    alpha,
    X,
    Y,
    T,
    txy_points_for_integral_evaluation,
    radial_distance_for_integral_evaluation,
    polar_angle_for_integral_evaluation,
    xy_grid_cell_area_for_integral_evaluation,
    boundary_condition_weight,
    initial_condition_weight,
    cost_functional_weight,
):
    print("Tracing evaluate_norms_after_training.")

    used_dtype = u_nn.layers[0].dtype

    eq_points_tuple, boundary_points_tuple, initial_points_tuple = generate_eq_boundary_and_intial_points_tuples(X, Y, T)

    equation_residual, u_pred = _calculate_equation_residual(eq_points_tuple, u_nn, equation_rhs_function, heat_coef)
    L1_equation_norm = L1_norm(equation_residual)
    L2_equation_norm = L2_norm(equation_residual)
    max_equation_norm = max_norm(equation_residual)

    initial_condition_residual = _calculate_initial_condition_residual(initial_points_tuple, u_nn, u_pred, initial_condition_function, heat_coef)
    L1_init_cond_norm = L1_norm(initial_condition_residual)
    L2_init_cond_norm = L2_norm(initial_condition_residual)
    max_init_cond_norm = max_norm(initial_condition_residual)
    weighted_L1_init_cond_norm = tf.cast(initial_condition_weight, used_dtype) * L1_init_cond_norm
    weighted_L2_init_cond_norm = tf.cast(initial_condition_weight, used_dtype) * L2_init_cond_norm
    weighted_max_init_cond_norm = tf.cast(initial_condition_weight, used_dtype) * max_init_cond_norm

    boundary_residual = _calculate_boundary_residual(boundary_points_tuple, u_nn, q_nn)
    L1_boundary_cond_norm = L1_norm(boundary_residual)
    L2_boundary_cond_norm = L2_norm(boundary_residual)
    max_boundary_cond_norm = max_norm(boundary_residual)
    weighted_L1_boundary_cond_norm = tf.cast(boundary_condition_weight, used_dtype) * L1_boundary_cond_norm
    weighted_L2_boundary_cond_norm = tf.cast(boundary_condition_weight, used_dtype) * L2_boundary_cond_norm
    weighted_max_boundary_cond_norm = tf.cast(boundary_condition_weight, used_dtype) * max_boundary_cond_norm

    cost_functional_loss = calculate_approximation_of_function_fit_integral(u_nn,
                                                                            desired_function_at_final_time,
                                                                            txy_points_for_integral_evaluation,
                                                                            radial_distance_for_integral_evaluation,
                                                                            polar_angle_for_integral_evaluation,
                                                                            xy_grid_cell_area_for_integral_evaluation)
    weighted_cost_functional_loss = tf.cast(cost_functional_weight, used_dtype) * cost_functional_loss

    # Sorted in this order to prevent certain plotting symbols, during plotting final norm values after training,
    # from being hidden by other symbols because of overlaping.
    weighted_norm_dict = {
        "cost_functional_loss": cost_functional_loss,
        "weighted_cost_functional_loss": weighted_cost_functional_loss,
        "L1_boundary_cond_norm": L1_boundary_cond_norm,
        "L2_boundary_cond_norm": L2_boundary_cond_norm,
        "max_boundary_cond_norm": max_boundary_cond_norm,
        "weighted_L1_boundary_cond_norm": weighted_L1_boundary_cond_norm,
        "weighted_L2_boundary_cond_norm": weighted_L2_boundary_cond_norm,
        "weighted_max_boundary_cond_norm": weighted_max_boundary_cond_norm,
        "L1_equation_norm": L1_equation_norm,
        "L2_equation_norm": L2_equation_norm,
        "max_equation_norm": max_equation_norm,
        #   Even though weighted equation norms are same as non-weighted (weight is always 1.0),
        # I need to have weighted variants of the keys in the dict for easier plotting.
        "weighted_L1_equation_norm": L1_equation_norm,
        "weighted_L2_equation_norm": L2_equation_norm,
        "weighted_max_equation_norm": max_equation_norm,
        "L1_init_cond_norm": L1_init_cond_norm,
        "L2_init_cond_norm": L2_init_cond_norm,
        "max_init_cond_norm": max_init_cond_norm,
        "weighted_L1_init_cond_norm": weighted_L1_init_cond_norm,
        "weighted_L2_init_cond_norm": weighted_L2_init_cond_norm,
        "weighted_max_init_cond_norm": weighted_max_init_cond_norm,
    }

    return weighted_norm_dict


def train_u_nn_and_q_nn(
    u_nn,
    q_nn,
    equation_rhs_function,
    initial_condition_function,
    desired_function_at_final_time,
    heat_coef,
    alpha,
    circle_center_in_xy,
    circle_radius,
    t_start,
    t_stop,
    num_t_training_points,
    x_start,
    x_stop,
    num_of_grid_x_training_points,
    num_of_x_points_for_integral_evaluation,
    y_start,
    y_stop,
    num_of_grid_y_training_points,
    num_of_y_points_for_integral_evaluation,
    num_batches,
    initial_epoch,
    num_epochs,
    write_loss_values_every_x_epochs,
    boundary_condition_weight,
    initial_condition_weight,
    cost_functional_weight,
    q_nn_cooling_penalty_weight,
    optimizer,
    loss_fn,
    dtype,
    callbacks=None,
    shuffle_each_epoch=True,
    train_only_u_nn=False,
):
    """
    equation_rhs_function has arguments (t, x, y, u_prediction, heat_coeficient)
    initial_condition_function has arguments (x, y, u_prediction, heat_coeficient)
    .
    """
    eq_points_tuple, boundary_points_tuple, initial_points_tuple, X, Y, T = generate_training_points(t_start,
                                                                                                     t_stop,
                                                                                                     num_t_training_points,
                                                                                                     x_start,
                                                                                                     x_stop,
                                                                                                     num_of_grid_x_training_points,
                                                                                                     y_start,
                                                                                                     y_stop,
                                                                                                     num_of_grid_y_training_points,
                                                                                                     dtype)

    txy_points_for_integral_evaluation, \
    radial_distance_for_integral_evaluation, \
    polar_angle_for_integral_evaluation, \
    xy_grid_cell_area_for_integral_evaluation = generate_points_for_evaluation_of_function_fit_integral(t_stop,
                                                                                                        x_start,
                                                                                                        x_stop,
                                                                                                        num_of_x_points_for_integral_evaluation,
                                                                                                        y_start,
                                                                                                        y_stop,
                                                                                                        num_of_y_points_for_integral_evaluation,
                                                                                                        circle_radius,
                                                                                                        circle_center_in_xy,
                                                                                                        dtype)

    loss_history_dict = dict()  # Loss history contains list with loss values for specific loss names.

    # Because of problem of creating tf.variables in optimizer.apply_gradients in train_step,
    #   it is needed to retrace the train_step in every call of train_u_nn_and_q_nn.
    # Should this problem be fixed in the future, this explicit tf.function call could be replaced
    #   with a tf.function decorator on train_step function.
    _train_step = tf.function(train_step, jit_compile=False, reduce_retracing=True)
    # TODO: Něco nefunguje, když dám jit_compile (možná if podmínka uvnitř train_step). Zkusit to spravit a podívat se, jestli
    #   se nezrychlí výpočet při použití jit_compile.

    if callbacks is not None:
        for callback in callbacks:
            callback.on_train_begin()

    train_start_time = time.time()
    for epoch_index in range(initial_epoch, num_epochs):

        if callbacks is not None:
            for callback in callbacks:
                callback.on_epoch_begin()

        if shuffle_each_epoch:
            with tf.profiler.experimental.Trace("shuffle_in_epoch", step_num=epoch_index, _r=1):  # This enables the use of profiling and Tensorboard.
                eq_points_tuple, boundary_points_tuple, initial_points_tuple = shuffle_points(eq_points_tuple,
                                                                                              boundary_points_tuple,
                                                                                              initial_points_tuple)
                txy_points_for_integral_evaluation, \
                radial_distance_for_integral_evaluation, \
                polar_angle_for_integral_evaluation = shuffle_points_for_integral_evaluation(txy_points_for_integral_evaluation,
                                                                                             radial_distance_for_integral_evaluation,
                                                                                             polar_angle_for_integral_evaluation)

        for batch_index in range(num_batches):
            # tf.print(f"batch_index: {batch_index}")

            if callbacks is not None:
                for callback in callbacks:
                    callback.on_train_batch_begin(batch_index)

            if num_batches > 1:
                with tf.profiler.experimental.Trace("slicing_in_train_step", step_num=batch_index, _r=1):
                    # TODO: Check for possible memory leak with the slices: make sure the slice tensors don't stay in memory after train step.
                    eq_points_sliced_tuple, boundary_points_sliced_tuple, initial_points_sliced_tuple = \
                        generate_training_points_slices(eq_points_tuple,
                                                        boundary_points_tuple,
                                                        initial_points_tuple,
                                                        tf.constant(batch_index),
                                                        # batch_index is passed as tensor to stop retracing for different values of it.
                                                        num_batches)
                slice_of_txy_points_for_integral_evaluation, \
                slice_of_radial_distance_for_integral_evaluation, \
                slice_of_polar_angle_for_integral_evaluation = slice_points_for_integral_evaluation(txy_points_for_integral_evaluation,
                                                                                                    radial_distance_for_integral_evaluation,
                                                                                                    polar_angle_for_integral_evaluation,
                                                                                                    tf.constant(batch_index),
                                                                                                    num_batches)
            else:
                eq_points_sliced_tuple = eq_points_tuple
                boundary_points_sliced_tuple = boundary_points_tuple
                initial_points_sliced_tuple = initial_points_tuple
                slice_of_txy_points_for_integral_evaluation = txy_points_for_integral_evaluation
                slice_of_radial_distance_for_integral_evaluation = radial_distance_for_integral_evaluation
                slice_of_polar_angle_for_integral_evaluation = polar_angle_for_integral_evaluation


            with tf.profiler.experimental.Trace("train_step", step_num=batch_index, _r=1):
                loss_dict = _train_step(u_nn,
                                        q_nn,
                                        equation_rhs_function,
                                        initial_condition_function,
                                        desired_function_at_final_time,
                                        heat_coef,
                                        tf.constant(alpha),
                                        eq_points_sliced_tuple,
                                        boundary_points_sliced_tuple,
                                        initial_points_sliced_tuple,
                                        slice_of_txy_points_for_integral_evaluation,
                                        slice_of_radial_distance_for_integral_evaluation,
                                        slice_of_polar_angle_for_integral_evaluation,
                                        xy_grid_cell_area_for_integral_evaluation,
                                        boundary_condition_weight,
                                        initial_condition_weight,
                                        cost_functional_weight,
                                        q_nn_cooling_penalty_weight,
                                        optimizer,
                                        loss_fn,
                                        train_only_u_nn)

            for loss_name, loss_value in loss_dict.items():
                if loss_name not in loss_history_dict:
                    loss_history_dict[loss_name] = []
                loss_history_dict[loss_name].append(loss_value)

            if callbacks is not None:
                for callback in callbacks:
                    callback.on_train_batch_end(batch_index, loss_dict)

        if callbacks is not None:
            for callback in callbacks:
                callback.on_epoch_end(epoch_index, loss_dict)

        if epoch_index == 0:
            print_epoch_results(0, loss_dict)

        elif epoch_index == num_epochs - 1:
            print_epoch_results(num_epochs, loss_dict)

        elif epoch_index % write_loss_values_every_x_epochs == 0:
            print_epoch_results(epoch_index, loss_dict)

    train_end_time = time.time()
    training_time = train_end_time - train_start_time
    tf.print(f"\nTraining took {training_time}s.")

    # TODO: Retracing happens, fix it.
    weighted_norm_dict = evaluate_norms_after_training(u_nn,
                                                       q_nn,
                                                       equation_rhs_function,
                                                       initial_condition_function,
                                                       desired_function_at_final_time,
                                                       heat_coef,
                                                       tf.constant(alpha),
                                                       X,
                                                       Y,
                                                       T,
                                                       slice_of_txy_points_for_integral_evaluation,
                                                       slice_of_radial_distance_for_integral_evaluation,
                                                       slice_of_polar_angle_for_integral_evaluation,
                                                       xy_grid_cell_area_for_integral_evaluation,
                                                       tf.constant(boundary_condition_weight),
                                                       tf.constant(initial_condition_weight),
                                                       tf.constant(cost_functional_weight))

    if callbacks is not None:
        for callback in callbacks:
            callback.on_train_end()

    return loss_history_dict, weighted_norm_dict, training_time
