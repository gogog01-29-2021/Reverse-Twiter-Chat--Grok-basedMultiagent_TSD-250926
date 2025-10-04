import keras
import tensorflow as tf
import numpy as np
import math
import time
import sys
from ...training import raise_error, tfprint_tensor_shape_and_type, print_tensor_shape_and_type, tfprint_tensor_values
from ...training.PINN import L1_norm, L2_norm, max_norm, assert_equal_shapes, assert_finite_values

debug_assert_finite_values = True
print_tensor_shapes_and_types_for_debug = False
tfprint_tensor_shapes_and_types_for_debug = True
tfprint_tensor_values_for_debug = True

# TODO!!!!!!!!: PŘIDAT VŠUDE VYPISOVÁNÍ VELIKOSTÍ A TYPU A ZKONTROLOVAT HLAVNĚ VELIKOSTI, JESTLI SEDÍ.
#   TAKY NAPSAT NĚJAKÉ TESTY PRO JEDNOTLIVÉ FUNKCE, ABYCH OVĚŘIL, ŽE POČÍTAJÍ SPRÁVNĚ. V NĚJAKÝCH JSEM MĚL NĚKOLIK CHYB.

# __all__ = [
#     "print_epoch_results",
#     "generate_training_points",
#     "shuffle_points",
#     "generate_training_points_slices",
#     "train_step",
#     "train_u_nn_and_q_nn",
# ]

def print_epoch_results(epoch_index, logs):
    tf.print(f"Epoch {epoch_index}:")
    for key, value in logs.items():
        tf.print(f"{key}: {value}", end=", ")
    tf.print("")


tfprint_tensor_shapes_and_types_for_function_generate_training_points_in_circle_in_xy_plane_from_grid = False
tfprint_tensor_values_for_function_generate_training_points_in_circle_in_xy_plane_from_grid = False
@tf.function(reduce_retracing=True)
def generate_training_points_in_circle_in_xy_plane_from_grid(
    circle_center_in_xy,
    circle_radius,
    t_start,
    t_stop,
    num_t_training_points,
    num_of_grid_x_training_points,
    num_of_grid_y_training_points,
    num_xy_training_points_on_boundary,
):
    # Points for training equation are gained by making a grid and keeping only the points in the required circle.
    # Points for training boundary condition are made separately from their count.
    print("Tracing generate_training_points_in_circle_in_xy_plane_from_grid.")
    x = tf.linspace(circle_center_in_xy[0] - circle_radius, circle_center_in_xy[0] + circle_radius, num_of_grid_x_training_points)
    y = tf.linspace(circle_center_in_xy[1] - circle_radius, circle_center_in_xy[1] + circle_radius, num_of_grid_y_training_points)
    if tfprint_tensor_shapes_and_types_for_debug and tfprint_tensor_shapes_and_types_for_function_generate_training_points_in_circle_in_xy_plane_from_grid:
        tfprint_tensor_shape_and_type("x", x)
        tfprint_tensor_shape_and_type("y", y)
    x = x[1:-1]
    y = y[1:-1]
        #   The condition for circle doesn't include points from the edges of the grid.
        # If points in the circle are taken from the whole grid, the resulting ragged tensor will contain useless empty parts []
        # corresponding to the points on the edges of the grid, as none of the points from any edge were included in the final circle.
        # Thus we throw the edges of the grid out before we make the circle in order to not have empty parts (empty rows/columns)
        # in the final Ragged tensor.
    t = tf.linspace(t_start, t_stop, num_t_training_points)
    if tfprint_tensor_shapes_and_types_for_debug and tfprint_tensor_shapes_and_types_for_function_generate_training_points_in_circle_in_xy_plane_from_grid:
        tfprint_tensor_shape_and_type("t", t)

    T_grid, X_grid, Y_grid = tf.meshgrid(t, x, y, indexing="ij")
    # Indexování 'ij' zařídí, že bod na indexu [i,j,k] odpovídá bodu s hodnotami t[i], x[j], y[k].
    # Tedy prostě standardní kartészké indexování: z hodnot t, x a y se vygeneruje 3D mřížka a T, X a Y určují
    # hodnoty t, x a y v daných bodech. Např. hodnota x v bodě s indexy i,j,k je X[i,j,k], atd.

    if tfprint_tensor_shapes_and_types_for_debug and tfprint_tensor_shapes_and_types_for_function_generate_training_points_in_circle_in_xy_plane_from_grid:
        tfprint_tensor_shape_and_type("T_grid", T_grid)
        tfprint_tensor_shape_and_type("X_grid", X_grid)
        tfprint_tensor_shape_and_type("Y_grid", Y_grid)

    if tfprint_tensor_values_for_debug and tfprint_tensor_values_for_function_generate_training_points_in_circle_in_xy_plane_from_grid:
        tfprint_tensor_values("T_grid", T_grid, -1)
        tfprint_tensor_values("X_grid", X_grid, -1)
        tfprint_tensor_values("Y_grid", Y_grid, -1)

    boolean_mask_to_keep_points_in_circle = (X_grid - circle_center_in_xy[0])**2 + (Y_grid - circle_center_in_xy[1])**2 <= circle_radius**2
    T = tf.ragged.boolean_mask(T_grid, boolean_mask_to_keep_points_in_circle)
    X = tf.ragged.boolean_mask(X_grid, boolean_mask_to_keep_points_in_circle)
    Y = tf.ragged.boolean_mask(Y_grid, boolean_mask_to_keep_points_in_circle)

    if tfprint_tensor_shapes_and_types_for_debug and tfprint_tensor_shapes_and_types_for_function_generate_training_points_in_circle_in_xy_plane_from_grid:
        tfprint_tensor_shape_and_type("T", T)
        tfprint_tensor_shape_and_type("X", X)
        tfprint_tensor_shape_and_type("Y", Y)

    if tfprint_tensor_values_for_debug and tfprint_tensor_values_for_function_generate_training_points_in_circle_in_xy_plane_from_grid:
        tfprint_tensor_values("T", T, -1)
        tfprint_tensor_values("X", X, -1)
        tfprint_tensor_values("Y", Y, -1)

    # Delete unneeded tensors to free up memory.
    del x, y

    x_eq = tf.reshape(X[1:, :, :], [-1])
    y_eq = tf.reshape(Y[1:, :, :], [-1])
    t_eq = tf.reshape(T[1:, :, :], [-1])

    if tfprint_tensor_shapes_and_types_for_debug and tfprint_tensor_shapes_and_types_for_function_generate_training_points_in_circle_in_xy_plane_from_grid:
        tfprint_tensor_shape_and_type("t_eq", t_eq)
        tfprint_tensor_shape_and_type("x_eq", x_eq)
        tfprint_tensor_shape_and_type("y_eq", y_eq)

    if tfprint_tensor_values_for_debug and tfprint_tensor_values_for_function_generate_training_points_in_circle_in_xy_plane_from_grid:
        tfprint_tensor_values("t_eq", t_eq, -1)
        tfprint_tensor_values("x_eq", x_eq, -1)
        tfprint_tensor_values("y_eq", y_eq, -1)

    eq_points_tuple = (t_eq, x_eq, y_eq)


    polar_coordinates_angles_for_points = tf.linspace(0.0, 2*np.pi, num_xy_training_points_on_boundary + 1)[:-1]
    # Angle of 2pi gives the same point as angle of 0, so we generate generate one extra angle and throw out the angle of 2pi.
    # The one extra angle generated is to keep the desired number of boundary training points.
    x_b = circle_radius * tf.math.cos(polar_coordinates_angles_for_points)
    y_b = circle_radius * tf.math.sin(polar_coordinates_angles_for_points)

    T_b, X_b = tf.meshgrid(t, x_b, indexing="ij")
    _, Y_b = tf.meshgrid(t, y_b, indexing="ij")

    t_boundary = tf.reshape(T_b, [-1])
    x_boundary = tf.reshape(X_b, [-1])
    y_boundary = tf.reshape(Y_b, [-1])

    if tfprint_tensor_shapes_and_types_for_debug and tfprint_tensor_shapes_and_types_for_function_generate_training_points_in_circle_in_xy_plane_from_grid:
        tfprint_tensor_shape_and_type("t_boundary", t_boundary)
        tfprint_tensor_shape_and_type("x_boundary", x_boundary)
        tfprint_tensor_shape_and_type("y_boundary", y_boundary)

    if tfprint_tensor_values_for_debug and tfprint_tensor_values_for_function_generate_training_points_in_circle_in_xy_plane_from_grid:
        tfprint_tensor_values("t_boundary", t_boundary, -1)
        tfprint_tensor_values("x_boundary", x_boundary, -1)
        tfprint_tensor_values("y_boundary", y_boundary, -1)

    boundary_points_tuple = (t_boundary, x_boundary, y_boundary)


    x_ini_from_equation_points = tf.reshape(X[0, :, :], [-1])
    y_ini_from_equation_points = tf.reshape(Y[0, :, :], [-1])
    t_ini_from_equation_points = tf.reshape(T[0, :, :], [-1])

    # x_ini = tf.concat([x_ini_from_equation_points, x_boundary], 0)
    # y_ini = tf.concat([y_ini_from_equation_points, y_boundary], 0)
    # t_ini = tf.concat([t_ini_from_equation_points, t_boundary], 0)

    if tfprint_tensor_shapes_and_types_for_debug and tfprint_tensor_shapes_and_types_for_function_generate_training_points_in_circle_in_xy_plane_from_grid:
        tfprint_tensor_shape_and_type("t_ini_from_equation_points", t_ini_from_equation_points)
        tfprint_tensor_shape_and_type("x_ini_from_equation_points", x_ini_from_equation_points)
        tfprint_tensor_shape_and_type("y_ini_from_equation_points", y_ini_from_equation_points)

    if tfprint_tensor_values_for_debug and tfprint_tensor_values_for_function_generate_training_points_in_circle_in_xy_plane_from_grid:
        tfprint_tensor_values("t_ini_from_equation_points", t_ini_from_equation_points, -1)
        tfprint_tensor_values("x_ini_from_equation_points", x_ini_from_equation_points, -1)
        tfprint_tensor_values("y_ini_from_equation_points", y_ini_from_equation_points, -1)

    # initial_points_tuple = (t_ini, x_ini, y_ini)
    initial_points_tuple = (t_ini_from_equation_points, x_ini_from_equation_points, y_ini_from_equation_points)


    return eq_points_tuple, boundary_points_tuple, initial_points_tuple, X_grid, Y_grid, T_grid
    # TODO: Put X_grid, Y_grid and T_grid for calculating cost functional loss as separate values into the passed dictionaries,
    # then delete X_grid, Y_grid and T_grid here.


tfprint_tensor_shapes_and_types_for_function_shuffle_points = False
tfprint_tensor_values_for_function_shuffle_points = False
@tf.function(jit_compile=True,reduce_retracing=True)
def shuffle_points(
    eq_points_tuple,
    boundary_points_tuple,
    initial_points_tuple,
):
    print("Tracing shuffle_points.")
    t_eq, x_eq, y_eq = eq_points_tuple
    eq_random_indicies = tf.random.shuffle(tf.range(tf.size(t_eq)))
    t_eq = tf.gather(t_eq, eq_random_indicies)
    x_eq = tf.gather(x_eq, eq_random_indicies)
    y_eq = tf.gather(y_eq, eq_random_indicies)
    eq_points_tuple = (t_eq, x_eq, y_eq)

    if tfprint_tensor_shapes_and_types_for_debug and tfprint_tensor_shapes_and_types_for_function_shuffle_points:
        tfprint_tensor_shape_and_type("t_eq", t_eq)
        tfprint_tensor_shape_and_type("x_eq", x_eq)
        tfprint_tensor_shape_and_type("y_eq", y_eq)

    if tfprint_tensor_values_for_debug and tfprint_tensor_values_for_function_shuffle_points:
        tfprint_tensor_values("t_eq", t_eq, -1)
        tfprint_tensor_values("x_eq", x_eq, -1)
        tfprint_tensor_values("y_eq", y_eq, -1)


    t_ini, x_ini, y_ini = initial_points_tuple
    initial_random_indicies = tf.random.shuffle(tf.range(tf.size(t_ini)))
    t_ini = tf.gather(t_ini, initial_random_indicies)
    x_ini = tf.gather(x_ini, initial_random_indicies)
    y_ini = tf.gather(y_ini, initial_random_indicies)
    initial_points_tuple = (t_ini, x_ini, y_ini)

    if tfprint_tensor_shapes_and_types_for_debug and tfprint_tensor_shapes_and_types_for_function_shuffle_points:
        tfprint_tensor_shape_and_type("t_ini", t_ini)
        tfprint_tensor_shape_and_type("x_ini", x_ini)
        tfprint_tensor_shape_and_type("y_ini", y_ini)

    if tfprint_tensor_values_for_debug and tfprint_tensor_values_for_function_shuffle_points:
        tfprint_tensor_values("t_ini", t_ini, -1)
        tfprint_tensor_values("x_ini", x_ini, -1)
        tfprint_tensor_values("y_ini", y_ini, -1)

    t_boundary, x_boundary, y_boundary = boundary_points_tuple
    boundary_random_indicies = tf.random.shuffle(tf.range(tf.size(t_boundary)))
    t_boundary = tf.gather(t_boundary, boundary_random_indicies)
    x_boundary = tf.gather(x_boundary, boundary_random_indicies)
    y_boundary = tf.gather(y_boundary, boundary_random_indicies)
    boundary_points_tuple = (t_boundary, x_boundary, y_boundary)

    if tfprint_tensor_shapes_and_types_for_debug and tfprint_tensor_shapes_and_types_for_function_shuffle_points:
        tfprint_tensor_shape_and_type("t_boundary", t_boundary)
        tfprint_tensor_shape_and_type("x_boundary", x_boundary)
        tfprint_tensor_shape_and_type("y_boundary", y_boundary)

    if tfprint_tensor_values_for_debug and tfprint_tensor_values_for_function_shuffle_points:
        tfprint_tensor_values("t_boundary", t_boundary, -1)
        tfprint_tensor_values("x_boundary", x_boundary, -1)
        tfprint_tensor_values("y_boundary", y_boundary, -1)

    return eq_points_tuple, boundary_points_tuple, initial_points_tuple


tfprint_tensor_shapes_and_types_for_function__generate_training_points_slice = False
tfprint_tensor_values_for_function__generate_training_points_slice = False
def _generate_training_points_slice(
    tuple_with_points_in_txy_order,
    batch_index,
    num_of_batches,
):
    print("Tracing _generate_training_points_slice.")
    t, x, y = tuple_with_points_in_txy_order

    # POROVNÁVÁNÍ VELIKOSTÍ A CELKOVĚ TENZORŮ A HÁZENÍ VÝJIMEK JE TROCHU SLOŽITĚJŠÍ, VIZ POZNÁMKA VE SLOŽCE U MĚ NA PLOŠE.

    # if tf.size(t) != tf.size(x):
    #     tf.print("sizes don't match")
    #     # raise_error(ValueError, f"Size of t ({tf.size(t)}) doesn't match size of x ({tf.size(x)}).")
    #     raise ValueError("Size of t doesn't match size of x.")

    # TODO: Assert funguje, ale tf.cond ne, podívat se na to.
    # tf.debugging.assert_equal(tf.size(t), tf.size(x), message="Error: Sizes do not match.")
    # tf.cond(tf.equal(tf.size(t), tf.size(x)),
    #     true_fn=lambda: tf.print(),
    #     false_fn=lambda: raise_error(ValueError, f"Size of t ({tf.size(t)}) doesn't match size of x ({tf.size(x)})."))
    # tf.cond(tf.equal(tf.size(x), tf.size(y)),
    #     true_fn=lambda: tf.print(),
    #     false_fn=lambda: raise_error(ValueError, f"Size of x ({tf.size(x)}) doesn't match size of y ({tf.size(y)})."))
    # tf.cond(tf.equal(tf.size(t), tf.size(y)),
    #     true_fn=lambda: tf.print(),
    #     false_fn=lambda: raise_error(ValueError, f"Size of t ({tf.size(t)}) doesn't match size of y ({tf.size(y)})."))

    last_batch_index = num_of_batches - 1
    num_of_points_in_slice = tf.cast(tf.math.floor(tf.size(t) / num_of_batches), dtype=tf.int32)
    
    if batch_index != last_batch_index:
        t_slice = tf.slice(t, begin=[batch_index * num_of_points_in_slice], size=[num_of_points_in_slice])
        x_slice = tf.slice(x, begin=[batch_index * num_of_points_in_slice], size=[num_of_points_in_slice])
        y_slice = tf.slice(y, begin=[batch_index * num_of_points_in_slice], size=[num_of_points_in_slice])
    else:
        t_slice = t[last_batch_index * num_of_points_in_slice : ]
        x_slice = x[last_batch_index * num_of_points_in_slice : ]
        y_slice = y[last_batch_index * num_of_points_in_slice : ]

    if tfprint_tensor_shapes_and_types_for_debug and tfprint_tensor_shapes_and_types_for_function__generate_training_points_slice:
        tfprint_tensor_shape_and_type("t_slice", t_slice)
        tfprint_tensor_shape_and_type("x_slice", x_slice)
        tfprint_tensor_shape_and_type("y_slice", y_slice)

    if tfprint_tensor_values_for_debug and tfprint_tensor_values_for_function__generate_training_points_slice:
        tfprint_tensor_values("t_slice", t_slice)
        tfprint_tensor_values("x_slice", x_slice)
        tfprint_tensor_values("y_slice", y_slice)

    tuple_with_slices = (t_slice, x_slice, y_slice)
    return tuple_with_slices


@tf.function(reduce_retracing=True)
def generate_training_points_slices(
    tuple_with_eq_points,
    tuple_with_boundary_points,
    tuple_with_initial_points,
    batch_index,
    num_of_batches,
):
    print(f"Tracing generate_training_points_slices with batch_index={batch_index}.")
    tuple_with_eq_points_slice = _generate_training_points_slice(tuple_with_eq_points, batch_index, num_of_batches)

    tuple_with_boundary_points_slice = _generate_training_points_slice(tuple_with_boundary_points, batch_index, num_of_batches)

    tuple_with_initial_points_slice = _generate_training_points_slice(tuple_with_initial_points, batch_index, num_of_batches)

    return (tuple_with_eq_points_slice, tuple_with_boundary_points_slice, tuple_with_initial_points_slice)


tfprint_tensor_shapes_and_types_for_function__calculate_approximation_of_integral_with_final_time_fit = False
tfprint_tensor_values_for_function__calculate_approximation_of_integral_with_final_time_fit = False
def _calculate_approximation_of_integral_with_final_time_fit(
    u_nn,
    desired_function_at_final_time,
    X,
    Y,
    T,
    circle_center_in_xy,
    circle_radius,
):
    """
    Argument desired_function_at_final_time should be a function representing Zernike polynomial.
    It's arguments are (radial_distance, polar_angle).
    """
    print("Tracing _calculate_approximation_of_integral_with_final_time_fit.")
    x_evaluation_points = ( X[-1, :-1, :-1] + X[-1, 1:, 1:] ) / 2
    y_evaluation_points = ( Y[-1, :-1, :-1] + Y[-1, 1:, 1:] ) / 2
    t_evaluation_points = ( T[-1, :-1, :-1] + T[-1, 1:, 1:] ) / 2

    boolean_mask_to_keep_points_in_circle = (x_evaluation_points - circle_center_in_xy[0])**2 + (y_evaluation_points - circle_center_in_xy[1])**2 <= circle_radius**2

    if tfprint_tensor_shapes_and_types_for_debug and tfprint_tensor_shapes_and_types_for_function__calculate_approximation_of_integral_with_final_time_fit:
        tfprint_tensor_shape_and_type("X", X)
        tfprint_tensor_shape_and_type("Y", Y)
        tfprint_tensor_shape_and_type("T", T)

        tfprint_tensor_shape_and_type("x_evaluation_points before boolean mask", x_evaluation_points)
        tfprint_tensor_shape_and_type("y_evaluation_points before boolean mask", y_evaluation_points)
        tfprint_tensor_shape_and_type("t_evaluation_points before boolean mask", t_evaluation_points)

        tfprint_tensor_shape_and_type("boolean mask to keep points in circle", boolean_mask_to_keep_points_in_circle)

    if tfprint_tensor_values_for_debug and tfprint_tensor_values_for_function__calculate_approximation_of_integral_with_final_time_fit:
        tfprint_tensor_values("X", X)
        tfprint_tensor_values("Y", Y)
        tfprint_tensor_values("T", T)

        tfprint_tensor_values("x_evaluation_points before boolean mask", x_evaluation_points)
        tfprint_tensor_values("y_evaluation_points before boolean mask", y_evaluation_points)
        tfprint_tensor_values("t_evaluation_points before boolean mask", t_evaluation_points)

        tfprint_tensor_values("boolean mask to keep points in circle", boolean_mask_to_keep_points_in_circle)

    x_evaluation_points = x_evaluation_points[boolean_mask_to_keep_points_in_circle]
    y_evaluation_points = y_evaluation_points[boolean_mask_to_keep_points_in_circle]
    t_evaluation_points = t_evaluation_points[boolean_mask_to_keep_points_in_circle]

    txy_evaluation_points = tf.stack([t_evaluation_points, x_evaluation_points, y_evaluation_points,], axis=1)

    u_Txy = u_nn(txy_evaluation_points, training=True)[:, 0]
    # Make it into rank 1 tensor so that shape is same as desired_function_at_final_time_at_xy.


    x_evaluation_points_on_unit_circle = (x_evaluation_points - circle_center_in_xy[0]) / circle_radius
    y_evaluation_points_on_unit_circle = (y_evaluation_points - circle_center_in_xy[1]) / circle_radius

    radial_distance = tf.math.sqrt(x_evaluation_points_on_unit_circle**2 + y_evaluation_points_on_unit_circle**2)
    polar_angle = tf.math.atan2(y_evaluation_points_on_unit_circle, x_evaluation_points_on_unit_circle)

    desired_function_at_final_time_at_xy = desired_function_at_final_time(radial_distance, polar_angle)
    if isinstance(desired_function_at_final_time_at_xy, (float, int)):
        desired_function_at_final_time_at_xy = tf.fill(u_Txy.shape, float(desired_function_at_final_time_at_xy))


    # Because X, Y and T are created from linspace, so equally spaced points,
    #   we can calculate the distance between neighbouring points on the grid (=step) as difference between any points,
    #   for example points with index 0 and 1.
    x_step = X[0, 1, 0] - X[0, 0, 0]
    y_step = Y[0, 0, 1] - Y[0, 0, 0]
    xy_grid_square_area = x_step * y_step

    if tfprint_tensor_shapes_and_types_for_debug and tfprint_tensor_shapes_and_types_for_function__calculate_approximation_of_integral_with_final_time_fit:
        tfprint_tensor_shape_and_type("x_evaluation_points after boolean mask", x_evaluation_points)
        tfprint_tensor_shape_and_type("y_evaluation_points after boolean mask", y_evaluation_points)
        tfprint_tensor_shape_and_type("t_evaluation_points after boolean mask", t_evaluation_points)
        tfprint_tensor_shape_and_type("txy_evaluation_points", txy_evaluation_points)

        tfprint_tensor_shape_and_type("x_evaluation_points_on_unit_circle", x_evaluation_points_on_unit_circle)
        tfprint_tensor_shape_and_type("y_evaluation_points_on_unit_circle", y_evaluation_points_on_unit_circle)
        tfprint_tensor_shape_and_type("radial_distance", radial_distance)
        tfprint_tensor_shape_and_type("polar_angle", polar_angle)

        tfprint_tensor_shape_and_type("x_step", x_step)
        tfprint_tensor_shape_and_type("y_step", y_step)
        tfprint_tensor_shape_and_type("xy_grid_square_area", xy_grid_square_area)

        tfprint_tensor_shape_and_type("u_Txy", u_Txy)
        tfprint_tensor_shape_and_type("desired_function_at_final_time_at_xy", desired_function_at_final_time_at_xy)
        print()

    if tfprint_tensor_values_for_debug and tfprint_tensor_values_for_function__calculate_approximation_of_integral_with_final_time_fit:
        tfprint_tensor_values("x_evaluation_points after boolean mask", x_evaluation_points)
        tfprint_tensor_values("y_evaluation_points after boolean mask", y_evaluation_points)
        tfprint_tensor_values("t_evaluation_points after boolean mask", t_evaluation_points)
        tfprint_tensor_values("txy_evaluation_points", txy_evaluation_points)

        tfprint_tensor_values("x_evaluation_points_on_unit_circle", x_evaluation_points_on_unit_circle)
        tfprint_tensor_values("y_evaluation_points_on_unit_circle", y_evaluation_points_on_unit_circle)
        tfprint_tensor_values("radial_distance", radial_distance)
        tfprint_tensor_values("polar_angle", polar_angle)

        tfprint_tensor_values("x_step", x_step)
        tfprint_tensor_values("y_step", y_step)
        tfprint_tensor_values("xy_grid_square_area", xy_grid_square_area)

        tfprint_tensor_values("u_Txy", u_Txy)
        tfprint_tensor_values("desired_function_at_final_time_at_xy", desired_function_at_final_time_at_xy)
        print()

    if print_tensor_shapes_and_types_for_debug:
        print_tensor_shape_and_type(f"{x_evaluation_points=}")
        print_tensor_shape_and_type(f"{y_evaluation_points=}")
        print_tensor_shape_and_type(f"{t_evaluation_points=}")
        print_tensor_shape_and_type(f"{txy_evaluation_points=}")

        print_tensor_shape_and_type(f"{x_step=}")
        print_tensor_shape_and_type(f"{y_step=}")
        print_tensor_shape_and_type(f"{xy_grid_square_area=}")

        print_tensor_shape_and_type(f"{u_Txy=}")
        print_tensor_shape_and_type(f"{desired_function_at_final_time_at_xy=}")
        print()

    # if tf.config.functions_run_eagerly(): # Running functions eagerly is used to check proper implementation and calculation.
                                          # Run in graph mode for actual calculations after checking implementation is correct.
    assert_equal_shapes(u_Txy, desired_function_at_final_time_at_xy)
    assert_equal_shapes(xy_grid_square_area, tf.constant(0.0)) # Check scalar shape.

    approximation_of_integral_with_final_time_fit = tf.math.reduce_sum(tf.math.squared_difference(u_Txy, desired_function_at_final_time_at_xy))
    approximation_of_integral_with_final_time_fit = xy_grid_square_area * approximation_of_integral_with_final_time_fit

    if tfprint_tensor_shapes_and_types_for_debug and tfprint_tensor_shapes_and_types_for_function__calculate_approximation_of_integral_with_final_time_fit:
        tfprint_tensor_shape_and_type("approximation_of_integral_with_final_time_fit", approximation_of_integral_with_final_time_fit)
        print()

    if print_tensor_shapes_and_types_for_debug:
        print_tensor_shape_and_type(f"{approximation_of_integral_with_final_time_fit=}")
        print()

    return approximation_of_integral_with_final_time_fit


@tf.function()
def calculate_functional_loss(
    u_nn,
    desired_function_at_final_time,
    alpha,
    X,
    Y,
    T,
    circle_center_in_xy,
    circle_radius,
):
    print("Tracing calculate_functional_loss.")
    approximation_of_integral_with_final_time_fit = _calculate_approximation_of_integral_with_final_time_fit(
        u_nn, desired_function_at_final_time, X, Y, T, circle_center_in_xy, circle_radius,)

    # TODO: Fitování neuronky na finální funkci bylo horší s tím okrajovým integrálem - zeptat se, zda je opravdu v tom cost funkcionálu potřeba.
    #   A provést více testů s timhle členem a bez něj, aby se zjistilo, jak moc má na fitování a jiný výsledky vliv.
    # approximation_of_boundary_integral_at_left_boundary = _calculate_approximation_of_boundary_integral_for_left_boundary_part(u_nn, X, Y, T)

    # TODO: Přepsat pro oblast tvaru kruhu.
    # approximation_of_boundary_integral_at_right_boundary = _calculate_approximation_of_boundary_integral_for_right_boundary_part(u_nn, X, Y, T)

    # approximation_of_boundary_integral_at_bottom_boundary = _calculate_approximation_of_boundary_integral_for_bottom_boundary_part(u_nn, X, Y, T)

    # approximation_of_boundary_integral_at_top_boundary = _calculate_approximation_of_boundary_integral_for_top_boundary_part(u_nn, X, Y, T)

    # approximation_of_boundary_integral = (approximation_of_boundary_integral_at_left_boundary
    #                                       + approximation_of_boundary_integral_at_right_boundary
    #                                       + approximation_of_boundary_integral_at_bottom_boundary
    #                                       + approximation_of_boundary_integral_at_top_boundary)
 
    cost_functional_loss_term_value = 1/2 * approximation_of_integral_with_final_time_fit #+ alpha/2 * approximation_of_boundary_integral

    return cost_functional_loss_term_value


tfprint_tensor_shapes_and_types_for_function__calculate_equation_residual = False
tfprint_tensor_values_for_function__calculate_equation_residual = False
def _calculate_equation_residual(
    eq_points_tuple,
    u_nn,
    equation_rhs_function,
    heat_coef,
):
    t_eq, x_eq, y_eq = eq_points_tuple

    with tf.GradientTape(watch_accessed_variables=False, persistent=True) as derivative_tape:
        derivative_tape.watch(t_eq)
        derivative_tape.watch(x_eq)
        derivative_tape.watch(y_eq)
        txy_eq = tf.stack([t_eq, x_eq, y_eq], axis=1)
        u_pred = u_nn(txy_eq, training=True)

        u_x = derivative_tape.gradient(u_pred, x_eq)
        u_y = derivative_tape.gradient(u_pred, y_eq)
    u_xx = derivative_tape.gradient(u_x, x_eq)
    u_yy = derivative_tape.gradient(u_y, y_eq)
    u_t = derivative_tape.gradient(u_pred, t_eq)

    if print_tensor_shapes_and_types_for_debug:
        print_tensor_shape_and_type(f"{t_eq=}")
        print_tensor_shape_and_type(f"{x_eq=}")
        print_tensor_shape_and_type(f"{y_eq=}")
        print_tensor_shape_and_type(f"{txy_eq=}")
        print_tensor_shape_and_type(f"{u_pred=}")
        print_tensor_shape_and_type(f"{u_x=}")
        print_tensor_shape_and_type(f"{u_y=}")
        print_tensor_shape_and_type(f"{u_xx=}")
        print_tensor_shape_and_type(f"{u_yy=}")
        print_tensor_shape_and_type(f"{u_t=}")

    if tfprint_tensor_shapes_and_types_for_debug and tfprint_tensor_shapes_and_types_for_function__calculate_equation_residual:
        tfprint_tensor_shape_and_type("t_eq", t_eq)
        tfprint_tensor_shape_and_type("x_eq", x_eq)
        tfprint_tensor_shape_and_type("y_eq", y_eq)
        tfprint_tensor_shape_and_type("txy_eq", txy_eq)
        tfprint_tensor_shape_and_type("u_pred", u_pred)
        tfprint_tensor_shape_and_type("u_x", u_x)
        tfprint_tensor_shape_and_type("u_y", u_y)
        tfprint_tensor_shape_and_type("u_xx", u_xx)
        tfprint_tensor_shape_and_type("u_yy", u_yy)
        tfprint_tensor_shape_and_type("u_t", u_t)

    if tfprint_tensor_values_for_debug and tfprint_tensor_values_for_function__calculate_equation_residual:
        tfprint_tensor_values("t_eq", t_eq)
        tfprint_tensor_values("x_eq", x_eq)
        tfprint_tensor_values("y_eq", y_eq)
        tfprint_tensor_values("txy_eq", txy_eq)
        tfprint_tensor_values("u_pred", u_pred)
        tfprint_tensor_values("u_x", u_x)
        tfprint_tensor_values("u_y", u_y)
        tfprint_tensor_values("u_xx", u_xx)
        tfprint_tensor_values("u_yy", u_yy)
        tfprint_tensor_values("u_t", u_t)

    if debug_assert_finite_values:
        assert_finite_values(t_eq)
        assert_finite_values(x_eq)
        assert_finite_values(y_eq)
        assert_finite_values(txy_eq)
        assert_finite_values(u_pred)
        assert_finite_values(u_x)
        assert_finite_values(u_y)
        assert_finite_values(u_xx)
        assert_finite_values(u_yy)
        assert_finite_values(u_t)

    del derivative_tape, txy_eq, u_x, u_y

    # tf.print("txy_eq.shape: ", end="")
    # tf.print(txy_eq.shape)
    # tf.print("u_pred.shape: ", end="")
    # tf.print(u_pred.shape)
    # tf.print("u_x.shape: ", end="")
    # tf.print(u_x.shape)
    # tf.print("u_xx.shape: ", end="")
    # tf.print(u_xx.shape)
    # tf.print("u_t.shape: ", end="")
    # tf.print(u_t.shape)

    # print_tensor(tf.reshape(u_x, shape=[-1]))
    # print_tensor(tf.reshape(u_xx, shape=[-1]))
    # print_tensor(tf.reshape(u_t, shape=[-1]))

    RHS = equation_rhs_function(t_eq, x_eq, y_eq, u_pred, heat_coef)
    if isinstance(RHS, (float, int)):
        RHS = tf.fill(u_t.shape, float(RHS))

    # if tf.config.functions_run_eagerly():
    assert_equal_shapes(u_t,              heat_coef * u_xx)
    assert_equal_shapes(heat_coef * u_xx, heat_coef * u_yy)
    assert_equal_shapes(heat_coef * u_yy, RHS)

    equation_residual = u_t - heat_coef * u_xx - heat_coef * u_yy - RHS

    if debug_assert_finite_values:
        assert_finite_values(RHS)
        assert_finite_values(equation_residual)

    return equation_residual, u_pred


def _calculate_initial_condition_residual(
    initial_points_tuple,
    u_nn,
    u_pred,
    initial_condition_function,
    heat_coef,
):
    t_ini, x_ini, y_ini = initial_points_tuple

    txy_ini = tf.stack([t_ini, x_ini, y_ini], axis=-1)
    u_ini_pred = u_nn(txy_ini, training=True)[:, 0]
    u_ini_target = initial_condition_function(x_ini, y_ini, u_pred, heat_coef)
    if isinstance(u_ini_target, (float, int)):
        u_ini_target = tf.fill(tf.shape(u_ini_pred), float(u_ini_target))

    # if tf.config.functions_run_eagerly():
    assert_equal_shapes(u_ini_pred, u_ini_target)

    initial_condition_residual = u_ini_pred - u_ini_target

    del txy_ini, u_ini_pred, u_ini_target

    return initial_condition_residual


tfprint_tensor_shapes_and_types_for_function__calculate_boundary_residual = False
tfprint_tensor_values_for_function__calculate_boundary_residual = False
def _calculate_boundary_residual(
    boundary_points_tuple,
    u_nn,
    q_nn,
    circle_center_in_xy,
):
    t_boundary, x_boundary, y_boundary = boundary_points_tuple

    with tf.GradientTape(watch_accessed_variables=False, persistent=True) as derivative_tape:
        derivative_tape.watch(t_boundary)
        derivative_tape.watch(x_boundary)
        derivative_tape.watch(y_boundary)
        txy_boundary = tf.stack([t_boundary, x_boundary, y_boundary], axis=1)
        u_pred = u_nn(txy_boundary, training=True)

    u_x = derivative_tape.gradient(u_pred, x_boundary)
    u_y = derivative_tape.gradient(u_pred, y_boundary)

    normal_to_circle_boundary_x = x_boundary - circle_center_in_xy[0]
    normal_to_circle_boundary_y = y_boundary - circle_center_in_xy[1]
    normal_to_circle_boundary = tf.stack([normal_to_circle_boundary_x, normal_to_circle_boundary_y], axis=1)
    unit_normal_to_circle_boundary = tf.linalg.normalize(normal_to_circle_boundary, axis=1)[0]
    # [0] extracts the tensor from normalize, as normalize returns tuple containing the normalized tensor and its norm.

    dot_product_of_gradient_of_u_and_circle_boundary_normal = u_x * unit_normal_to_circle_boundary[:, 0] + u_y * unit_normal_to_circle_boundary[:, 1]
    boundary_residual = dot_product_of_gradient_of_u_and_circle_boundary_normal - q_nn(txy_boundary, training=True)[:,0]
    # txy_boundary has shape [..., 1], therefor q_nn result has the same shape,
    # but the dot product only [...], so we need to take only the first dimension of q_nn.

    if tfprint_tensor_shapes_and_types_for_debug and tfprint_tensor_shapes_and_types_for_function__calculate_boundary_residual:
        tfprint_tensor_shape_and_type("t_boundary", t_boundary)
        tfprint_tensor_shape_and_type("x_boundary", x_boundary)
        tfprint_tensor_shape_and_type("y_boundary", y_boundary)
        tfprint_tensor_shape_and_type("txy_boundary", txy_boundary)
        tfprint_tensor_shape_and_type("u_pred", u_pred)
        tfprint_tensor_shape_and_type("u_x", u_x)
        tfprint_tensor_shape_and_type("u_y", u_y)
        tfprint_tensor_shape_and_type("normal_to_circle_boundary_x", normal_to_circle_boundary_x)
        tfprint_tensor_shape_and_type("normal_to_circle_boundary_y", normal_to_circle_boundary_y)
        tfprint_tensor_shape_and_type("normal_to_circle_boundary", normal_to_circle_boundary)
        tfprint_tensor_shape_and_type("unit_normal_to_circle_boundary", unit_normal_to_circle_boundary)
        tfprint_tensor_shape_and_type("dot_product_of_gradient_of_u_and_circle_boundary_normal", dot_product_of_gradient_of_u_and_circle_boundary_normal)
        tfprint_tensor_shape_and_type("boundary_residual", boundary_residual)

    if tfprint_tensor_values_for_debug and tfprint_tensor_values_for_function__calculate_boundary_residual:
        tfprint_tensor_values("t_boundary", t_boundary)
        tfprint_tensor_values("x_boundary", x_boundary)
        tfprint_tensor_values("y_boundary", y_boundary)
        tfprint_tensor_values("txy_boundary", txy_boundary)
        tfprint_tensor_values("u_pred", u_pred)
        tfprint_tensor_values("u_x", u_x)
        tfprint_tensor_values("u_y", u_y)
        tfprint_tensor_values("normal_to_circle_boundary_x", normal_to_circle_boundary_x)
        tfprint_tensor_values("normal_to_circle_boundary_y", normal_to_circle_boundary_y)
        tfprint_tensor_values("normal_to_circle_boundary", normal_to_circle_boundary)
        tfprint_tensor_values("unit_normal_to_circle_boundary", unit_normal_to_circle_boundary)
        tfprint_tensor_values("dot_product_of_gradient_of_u_and_circle_boundary_normal", dot_product_of_gradient_of_u_and_circle_boundary_normal)
        tfprint_tensor_values("boundary_residual", boundary_residual)

    if debug_assert_finite_values:
        assert_finite_values(t_boundary)
        assert_finite_values(x_boundary)
        assert_finite_values(y_boundary)
        assert_finite_values(txy_boundary)
        assert_finite_values(u_pred)
        assert_finite_values(u_x)
        assert_finite_values(u_y)
        assert_finite_values(normal_to_circle_boundary_x)
        assert_finite_values(normal_to_circle_boundary_y)
        assert_finite_values(normal_to_circle_boundary)
        assert_finite_values(unit_normal_to_circle_boundary)
        assert_finite_values(dot_product_of_gradient_of_u_and_circle_boundary_normal)
        assert_finite_values(boundary_residual)

    if tf.config.functions_run_eagerly():
        assert_equal_shapes(normal_to_circle_boundary_x, x_boundary)
        assert_equal_shapes(normal_to_circle_boundary_y, y_boundary)
        assert_equal_shapes(unit_normal_to_circle_boundary, normal_to_circle_boundary)
        assert_equal_shapes(u_x * unit_normal_to_circle_boundary[:, 0], u_y * unit_normal_to_circle_boundary[:, 1])
        assert_equal_shapes(dot_product_of_gradient_of_u_and_circle_boundary_normal, q_nn(txy_boundary, training=True)[:,0])

    return boundary_residual

# TODO: Continue to check the copied funtions from here.
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
    X,
    Y,
    T,
    circle_center_in_xy,
    circle_radius,
    boundary_condition_weight,
    initial_condition_weight,
    cost_functional_weight,
    optimizer,
    loss_fn,
    train_only_u_nn=False,
):
    print(f"Tracing train step.")
    # This is here just to catch if the function was retracing in every batch call.


    with tf.GradientTape(persistent=True) as weights_update_tape:
        equation_residual, u_pred = _calculate_equation_residual(eq_points_tuple, u_nn, equation_rhs_function, heat_coef)
        equation_loss = loss_fn(equation_residual)

        initial_condition_residual = _calculate_initial_condition_residual(initial_points_tuple, u_nn, u_pred, initial_condition_function, heat_coef)
        init_cond_loss = tf.cast(initial_condition_weight, tf.float32) * loss_fn(initial_condition_residual)

        boundary_residual = _calculate_boundary_residual(boundary_points_tuple, u_nn, q_nn, circle_center_in_xy)
        boundary_cond_loss = tf.cast(boundary_condition_weight, tf.float32) * loss_fn(boundary_residual)

        cost_functional_loss = tf.cast(cost_functional_weight, tf.float32) * calculate_functional_loss(u_nn,
                                                                                                       desired_function_at_final_time,
                                                                                                       alpha,
                                                                                                       X,
                                                                                                       Y,
                                                                                                       T,
                                                                                                       circle_center_in_xy,
                                                                                                       circle_radius)


        loss_dict = {
            "equation loss": equation_loss,
            "boundary condition loss": boundary_cond_loss,
            "initial condition loss": init_cond_loss,
            "cost functional loss": cost_functional_loss,
        }


    grads_u_nn = weights_update_tape.gradient(loss_dict, u_nn.trainable_variables)
    u_nn.optimizer.apply_gradients(zip(grads_u_nn, u_nn.trainable_variables))

    if not train_only_u_nn:
        grads_q_nn = weights_update_tape.gradient(loss_dict, q_nn.trainable_variables)
        q_nn.optimizer.apply_gradients(zip(grads_q_nn, q_nn.trainable_variables))

    return loss_dict


@tf.function(jit_compile=True,reduce_retracing=True)
def evaluate_norms_after_training(
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
    X,
    Y,
    T,
    circle_center_in_xy,
    circle_radius,
    boundary_condition_weight,
    initial_condition_weight,
    cost_functional_weight,
):
    print(f"Tracing evaluate_norms_after_training.")

    equation_residual, u_pred = _calculate_equation_residual(eq_points_tuple, u_nn, equation_rhs_function, heat_coef)
    L1_equation_norm = L1_norm(equation_residual)
    L2_equation_norm = L2_norm(equation_residual)
    max_equation_norm = max_norm(equation_residual)

    initial_condition_residual = _calculate_initial_condition_residual(initial_points_tuple, u_nn, u_pred, initial_condition_function, heat_coef)
    L1_init_cond_norm = tf.cast(initial_condition_weight, tf.float32) * L1_norm(initial_condition_residual)
    L2_init_cond_norm = tf.cast(initial_condition_weight, tf.float32) * L2_norm(initial_condition_residual)
    max_init_cond_norm = tf.cast(initial_condition_weight, tf.float32) * max_norm(initial_condition_residual)

    boundary_residual = _calculate_boundary_residual(boundary_points_tuple, u_nn, q_nn, circle_center_in_xy)
    L1_boundary_cond_norm = tf.cast(boundary_condition_weight, tf.float32) * L1_norm(boundary_residual)
    L2_boundary_cond_norm = tf.cast(boundary_condition_weight, tf.float32) * L2_norm(boundary_residual)
    max_boundary_cond_norm = tf.cast(boundary_condition_weight, tf.float32) * max_norm(boundary_residual)

    cost_functional_loss = tf.cast(cost_functional_weight, tf.float32) * calculate_functional_loss(u_nn,
                                                                                                   desired_function_at_final_time,
                                                                                                   alpha,
                                                                                                   X,
                                                                                                   Y,
                                                                                                   T,
                                                                                                   circle_center_in_xy,
                                                                                                   circle_radius)

    # Sorted in this order to prevent certain plotting symbols, during plotting final norm values after training,
    # from being hidden by other symbols because of overlaping.
    weighted_norm_dict = {
        "cost_functional_loss": cost_functional_loss,
        "L1_boundary_cond_norm": L1_boundary_cond_norm,
        "L2_boundary_cond_norm": L2_boundary_cond_norm,
        "max_boundary_cond_norm": max_boundary_cond_norm,
        "L1_equation_norm": L1_equation_norm,
        "L2_equation_norm": L2_equation_norm,
        "max_equation_norm": max_equation_norm,
        "L1_init_cond_norm": L1_init_cond_norm,
        "L2_init_cond_norm": L2_init_cond_norm,
        "max_init_cond_norm": max_init_cond_norm,
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
    num_of_grid_x_training_points,
    num_of_grid_y_training_points,
    num_xy_training_points_on_boundary,
    num_batches,
    initial_epoch,
    num_epochs,
    write_loss_values_every_x_epochs,
    boundary_condition_weight,
    initial_condition_weight,
    cost_functional_weight,
    optimizer,
    loss_fn,
    callbacks=None,
    shuffle_each_epoch=True,
    train_only_u_nn=False,
):
    """
    equation_rhs_function has arguments (t, x, y, u_prediction, heat_coeficient)
    initial_condition_function has arguments (x, y, u_prediction, heat_coeficient)
    .
    """
    eq_points_tuple, boundary_points_tuple, initial_points_tuple, X, Y, T = \
                                                    generate_training_points_in_circle_in_xy_plane_from_grid(circle_center_in_xy,
                                                                                                             circle_radius,
                                                                                                             t_start,
                                                                                                             t_stop,
                                                                                                             num_t_training_points,
                                                                                                             num_of_grid_x_training_points,
                                                                                                             num_of_grid_y_training_points,
                                                                                                             num_xy_training_points_on_boundary)

    loss_history_dict = dict() # Loss history contains list with loss values for specific loss names.

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
            with tf.profiler.experimental.Trace("shuffle_in_epoch", step_num=epoch_index, _r=1): # This enables the use of profiling and Tensorboard.
                eq_points_tuple, boundary_points_tuple, initial_points_tuple = shuffle_points(eq_points_tuple,
                                                                                              boundary_points_tuple,
                                                                                              initial_points_tuple)

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
                                                                                                            tf.constant(batch_index), # batch_index is passed as tensor to stop retracing for different values of it.
                                                                                                            num_batches)
            else:
                eq_points_sliced_tuple = eq_points_tuple
                boundary_points_sliced_tuple = boundary_points_tuple
                initial_points_sliced_tuple = initial_points_tuple


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
                                        # eq_points_tuple,
                                        # boundary_points_tuple,
                                        # initial_points_tuple,
                                        X,
                                        Y,
                                        T,
                                        circle_center_in_xy,
                                        circle_radius,
                                        boundary_condition_weight,
                                        initial_condition_weight,
                                        cost_functional_weight,
                                        optimizer,
                                        loss_fn,
                                        train_only_u_nn)

            # if batch_index == 0:
            #     concrete_train_step = train_step.get_concrete_function(u_nn,
            #                                                            q_nn,
            #                                                            equation_rhs_function,
            #                                                            initial_condition_function,
            #                                                            desired_function_at_final_time,
            #                                                            heat_coef,
            #                                                            tf.constant(alpha),
            #                                                            eq_points_sliced_tuple,
            #                                                            boundary_points_sliced_tuple,
            #                                                            initial_points_sliced_tuple,
            #                                                            X,
            #                                                            Y,
            #                                                            T,
            #                                                            boundary_condition_weight,
            #                                                            initial_condition_weight,
            #                                                            cost_functional_weight,
            #                                                            optimizer)
            #     print(concrete_train_step)

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

        elif epoch_index == num_epochs:
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
                                                       eq_points_tuple,
                                                       boundary_points_tuple,
                                                       initial_points_tuple,
                                                       X,
                                                       Y,
                                                       T,
                                                       circle_center_in_xy,
                                                       circle_radius,
                                                       tf.constant(boundary_condition_weight),
                                                       tf.constant(initial_condition_weight),
                                                       tf.constant(cost_functional_weight))

    if callbacks is not None:
        for callback in callbacks:
            callback.on_train_end()

    return loss_history_dict, weighted_norm_dict, training_time
