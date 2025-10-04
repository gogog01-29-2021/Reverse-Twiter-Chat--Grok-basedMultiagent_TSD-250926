import tensorflow as tf
import time
from ...training import raise_error, tfprint_tensor_shape_and_type, print_tensor_shape_and_type, tfprint_tensor_values
from ...training.PINN import L1_norm, L2_norm, max_norm

debug_assert_finite_values = True
print_tensor_shapes_and_types_for_debug = False
tfprint_tensor_shapes_and_types_for_debug = False
tfprint_tensor_values_for_debug = False


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
        tf.print(f"{key}: {value}")
    tf.print()


tfprint_tensor_shapes_and_types_for_function_generate_training_points = False
tfprint_tensor_values_for_function_generate_training_points = False
jit_compile_point_generation = True


@tf.function(jit_compile=jit_compile_point_generation)
def generate_eq_boundary_and_intial_points_tuples(
    X,
    Y,
    T,
):
    x_ini = tf.reshape(X[:, :, 0], [-1])
    y_ini = tf.reshape(Y[:, :, 0], [-1])
    t_ini = tf.reshape(T[:, :, 0], [-1])

    initial_points_tuple = (t_ini, x_ini, y_ini)

    if tfprint_tensor_shapes_and_types_for_debug and tfprint_tensor_shapes_and_types_for_function_generate_training_points:
        tfprint_tensor_shape_and_type("x_ini", x_ini)
        tfprint_tensor_shape_and_type("y_ini", y_ini)
        tfprint_tensor_shape_and_type("t_ini", t_ini)

    if tfprint_tensor_values_for_debug and tfprint_tensor_values_for_function_generate_training_points:
        tfprint_tensor_values("x_ini", x_ini, -1)
        tfprint_tensor_values("y_ini", y_ini, -1)
        tfprint_tensor_values("t_ini", t_ini, -1)

    x_eq = tf.reshape(X[1:-1, 1:-1, 1:], [-1])
    y_eq = tf.reshape(Y[1:-1, 1:-1, 1:], [-1])
    t_eq = tf.reshape(T[1:-1, 1:-1, 1:], [-1])

    eq_points_tuple = (t_eq, x_eq, y_eq)

    if tfprint_tensor_shapes_and_types_for_debug and tfprint_tensor_shapes_and_types_for_function_generate_training_points:
        tfprint_tensor_shape_and_type("x_eq", x_eq)
        tfprint_tensor_shape_and_type("y_eq", y_eq)
        tfprint_tensor_shape_and_type("t_eq", t_eq)

    if tfprint_tensor_values_for_debug and tfprint_tensor_values_for_function_generate_training_points:
        tfprint_tensor_values("x_eq", x_eq, -1)
        tfprint_tensor_values("y_eq", y_eq, -1)
        tfprint_tensor_values("t_eq", t_eq, -1)


    x_boundary_bottom = tf.reshape(X[:, 0, :], [-1])
    y_boundary_bottom = tf.reshape(Y[:, 0, :], [-1])
    t_boundary_bottom = tf.reshape(T[:, 0, :], [-1])
    boundary_points_bottom_tuple = (t_boundary_bottom, x_boundary_bottom, y_boundary_bottom)

    x_boundary_top = tf.reshape(X[:, -1, :], [-1])
    y_boundary_top = tf.reshape(Y[:, -1, :], [-1])
    t_boundary_top = tf.reshape(T[:, -1, :], [-1])
    boundary_points_top_tuple = (t_boundary_top, x_boundary_top, y_boundary_top)

    x_boundary_left = tf.reshape(X[0, :, :], [-1])
    y_boundary_left = tf.reshape(Y[0, :, :], [-1])
    t_boundary_left = tf.reshape(T[0, :, :], [-1])
    boundary_points_left_tuple = (t_boundary_left, x_boundary_left, y_boundary_left)

    x_boundary_right = tf.reshape(X[-1, :, :], [-1])
    y_boundary_right = tf.reshape(Y[-1, :, :], [-1])
    t_boundary_right = tf.reshape(T[-1, :, :], [-1])
    boundary_points_right_tuple = (t_boundary_right, x_boundary_right, y_boundary_right)

    boundary_points_tuple = (boundary_points_bottom_tuple, boundary_points_top_tuple, boundary_points_left_tuple, boundary_points_right_tuple)

    if tfprint_tensor_shapes_and_types_for_debug and tfprint_tensor_shapes_and_types_for_function_generate_training_points:
        tfprint_tensor_shape_and_type("x_boundary_bottom", x_boundary_bottom)
        tfprint_tensor_shape_and_type("y_boundary_bottom", y_boundary_bottom)
        tfprint_tensor_shape_and_type("t_boundary_bottom", t_boundary_bottom)
        tfprint_tensor_shape_and_type("x_boundary_top", x_boundary_top)
        tfprint_tensor_shape_and_type("y_boundary_top", y_boundary_top)
        tfprint_tensor_shape_and_type("t_boundary_top", t_boundary_top)
        tfprint_tensor_shape_and_type("x_boundary_left", x_boundary_left)
        tfprint_tensor_shape_and_type("y_boundary_left", y_boundary_left)
        tfprint_tensor_shape_and_type("t_boundary_left", t_boundary_left)
        tfprint_tensor_shape_and_type("x_boundary_right", x_boundary_right)
        tfprint_tensor_shape_and_type("y_boundary_right", y_boundary_right)
        tfprint_tensor_shape_and_type("t_boundary_right", t_boundary_right)

    if tfprint_tensor_values_for_debug and tfprint_tensor_values_for_function_generate_training_points:
        tfprint_tensor_values("x_boundary_bottom", x_boundary_bottom, -1)
        tfprint_tensor_values("y_boundary_bottom", y_boundary_bottom, -1)
        tfprint_tensor_values("t_boundary_bottom", t_boundary_bottom, -1)
        tfprint_tensor_values("x_boundary_top", x_boundary_top, -1)
        tfprint_tensor_values("y_boundary_top", y_boundary_top, -1)
        tfprint_tensor_values("t_boundary_top", t_boundary_top, -1)
        tfprint_tensor_values("x_boundary_left", x_boundary_left, -1)
        tfprint_tensor_values("y_boundary_left", y_boundary_left, -1)
        tfprint_tensor_values("t_boundary_left", t_boundary_left, -1)
        tfprint_tensor_values("x_boundary_right", x_boundary_right, -1)
        tfprint_tensor_values("y_boundary_right", y_boundary_right, -1)
        tfprint_tensor_values("t_boundary_right", t_boundary_right, -1)


    return eq_points_tuple, boundary_points_tuple, initial_points_tuple


@tf.function(jit_compile=jit_compile_point_generation)
def generate_training_points(
    t_start,
    t_stop,
    num_t_training_points,
    x_start,
    x_stop,
    num_x_training_points,
    y_start,
    y_stop,
    num_y_training_points,
    dtype,
):
    """
    Left and right boundary points are without corners, top and bottom points are with corners included.
    .
    """
    print("Tracing generate_training_points.")
    x = tf.linspace(tf.constant(x_start, dtype=dtype), x_stop, num_x_training_points)
    y = tf.linspace(tf.constant(y_start, dtype=dtype), y_stop, num_y_training_points)
    t = tf.linspace(tf.constant(t_start, dtype=dtype), t_stop, num_t_training_points)

    if tfprint_tensor_shapes_and_types_for_debug and tfprint_tensor_shapes_and_types_for_function_generate_training_points:
        tfprint_tensor_shape_and_type("x", x)
        tfprint_tensor_shape_and_type("y", y)
        tfprint_tensor_shape_and_type("t", t)

    if tfprint_tensor_values_for_debug and tfprint_tensor_values_for_function_generate_training_points:
        tfprint_tensor_values("x", x, -1)
        tfprint_tensor_values("y", y, -1)
        tfprint_tensor_values("t", t, -1)

    X, Y, T = tf.meshgrid(x, y, t, indexing='ij')
    # Indexování 'ij' zařídí, že bod na indexu [i,j,k] odpovídá bodu s hodnotami x[i], y[j], t[k].
    # Tedy prostě standardní kartészké indexování: z hodnot x, y a t se vygeneruje 3D mřížka a X, Y a T určují
    # hodnoty x, y a t v daných bodech. Např. hodnota x v bodě s indexy i,j,k je X[i,j,k], atd.

    if tfprint_tensor_shapes_and_types_for_debug and tfprint_tensor_shapes_and_types_for_function_generate_training_points:
        tfprint_tensor_shape_and_type("X", X)
        tfprint_tensor_shape_and_type("Y", Y)
        tfprint_tensor_shape_and_type("T", T)

    if tfprint_tensor_values_for_debug and tfprint_tensor_values_for_function_generate_training_points:
        tfprint_tensor_values("X", X, -1)
        tfprint_tensor_values("Y", Y, -1)
        tfprint_tensor_values("T", T, -1)

    # Delete unneeded tensors to free up memory.
    del x, y, t

    eq_points_tuple, boundary_points_tuple, initial_points_tuple = generate_eq_boundary_and_intial_points_tuples(X, Y, T)
    return eq_points_tuple, boundary_points_tuple, initial_points_tuple, X, Y, T


tfprint_tensor_shapes_and_types_for_function_shuffle_points = False
tfprint_tensor_values_for_function_shuffle_points = False


@tf.function(jit_compile=True, reduce_retracing=True)
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

    t_ini, x_ini, y_ini = initial_points_tuple
    initial_random_indicies = tf.random.shuffle(tf.range(tf.size(t_ini)))
    t_ini = tf.gather(t_ini, initial_random_indicies)
    x_ini = tf.gather(x_ini, initial_random_indicies)
    y_ini = tf.gather(y_ini, initial_random_indicies)
    initial_points_tuple = (t_ini, x_ini, y_ini)


    boundary_points_bottom_tuple, boundary_points_top_tuple, boundary_points_left_tuple, boundary_points_right_tuple = boundary_points_tuple

    t_boundary_bottom, x_boundary_bottom, y_boundary_bottom = boundary_points_bottom_tuple
    boundary_bottom_random_indicies = tf.random.shuffle(tf.range(tf.size(t_boundary_bottom)))
    t_boundary_bottom = tf.gather(t_boundary_bottom, boundary_bottom_random_indicies)
    x_boundary_bottom = tf.gather(x_boundary_bottom, boundary_bottom_random_indicies)
    y_boundary_bottom = tf.gather(y_boundary_bottom, boundary_bottom_random_indicies)
    boundary_points_bottom_tuple = (t_boundary_bottom, x_boundary_bottom, y_boundary_bottom)

    t_boundary_top, x_boundary_top, y_boundary_top = boundary_points_top_tuple
    boundary_top_random_indicies = tf.random.shuffle(tf.range(tf.size(t_boundary_top)))
    t_boundary_top = tf.gather(t_boundary_top, boundary_top_random_indicies)
    x_boundary_top = tf.gather(x_boundary_top, boundary_top_random_indicies)
    y_boundary_top = tf.gather(y_boundary_top, boundary_top_random_indicies)
    boundary_points_top_tuple = (t_boundary_top, x_boundary_top, y_boundary_top)

    t_boundary_left, x_boundary_left, y_boundary_left = boundary_points_left_tuple
    boundary_left_random_indicies = tf.random.shuffle(tf.range(tf.size(t_boundary_left)))
    t_boundary_left = tf.gather(t_boundary_left, boundary_left_random_indicies)
    x_boundary_left = tf.gather(x_boundary_left, boundary_left_random_indicies)
    y_boundary_left = tf.gather(y_boundary_left, boundary_left_random_indicies)
    boundary_points_left_tuple = (t_boundary_left, x_boundary_left, y_boundary_left)

    t_boundary_right, x_boundary_right, y_boundary_right = boundary_points_right_tuple
    boundary_right_random_indicies = tf.random.shuffle(tf.range(tf.size(t_boundary_right)))
    t_boundary_right = tf.gather(t_boundary_right, boundary_right_random_indicies)
    x_boundary_right = tf.gather(x_boundary_right, boundary_right_random_indicies)
    y_boundary_right = tf.gather(y_boundary_right, boundary_right_random_indicies)
    boundary_points_right_tuple = (t_boundary_right, x_boundary_right, y_boundary_right)

    boundary_points_tuple = (boundary_points_bottom_tuple, boundary_points_top_tuple, boundary_points_left_tuple, boundary_points_right_tuple)

    if tfprint_tensor_shapes_and_types_for_debug and tfprint_tensor_shapes_and_types_for_function_shuffle_points:
        tfprint_tensor_shape_and_type("t_eq", t_eq)
        tfprint_tensor_shape_and_type("x_eq", x_eq)
        tfprint_tensor_shape_and_type("y_eq", y_eq)
        tfprint_tensor_shape_and_type("t_ini", t_ini)
        tfprint_tensor_shape_and_type("x_ini", x_ini)
        tfprint_tensor_shape_and_type("y_ini", y_ini)
        tfprint_tensor_shape_and_type("t_boundary_bottom", t_boundary_bottom)
        tfprint_tensor_shape_and_type("x_boundary_bottom", x_boundary_bottom)
        tfprint_tensor_shape_and_type("y_boundary_bottom", y_boundary_bottom)
        tfprint_tensor_shape_and_type("t_boundary_top", t_boundary_top)
        tfprint_tensor_shape_and_type("x_boundary_top", x_boundary_top)
        tfprint_tensor_shape_and_type("y_boundary_top", y_boundary_top)
        tfprint_tensor_shape_and_type("t_boundary_left", t_boundary_left)
        tfprint_tensor_shape_and_type("x_boundary_left", x_boundary_left)
        tfprint_tensor_shape_and_type("y_boundary_left", y_boundary_left)
        tfprint_tensor_shape_and_type("t_boundary_right", t_boundary_right)
        tfprint_tensor_shape_and_type("x_boundary_right", x_boundary_right)
        tfprint_tensor_shape_and_type("y_boundary_right", y_boundary_right)

    if tfprint_tensor_values_for_debug and tfprint_tensor_values_for_function_shuffle_points:
        tfprint_tensor_values("t_eq", t_eq, -1)
        tfprint_tensor_values("x_eq", x_eq, -1)
        tfprint_tensor_values("y_eq", y_eq, -1)
        tfprint_tensor_values("t_ini", t_ini, -1)
        tfprint_tensor_values("x_ini", x_ini, -1)
        tfprint_tensor_values("y_ini", y_ini, -1)
        tfprint_tensor_values("t_boundary_bottom", t_boundary_bottom, -1)
        tfprint_tensor_values("x_boundary_bottom", x_boundary_bottom, -1)
        tfprint_tensor_values("y_boundary_bottom", y_boundary_bottom, -1)
        tfprint_tensor_values("t_boundary_top", t_boundary_top, -1)
        tfprint_tensor_values("x_boundary_top", x_boundary_top, -1)
        tfprint_tensor_values("y_boundary_top", y_boundary_top, -1)
        tfprint_tensor_values("t_boundary_left", t_boundary_left, -1)
        tfprint_tensor_values("x_boundary_left", x_boundary_left, -1)
        tfprint_tensor_values("y_boundary_left", y_boundary_left, -1)
        tfprint_tensor_values("t_boundary_right", t_boundary_right, -1)
        tfprint_tensor_values("x_boundary_right", x_boundary_right, -1)
        tfprint_tensor_values("y_boundary_right", y_boundary_right, -1)

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

    tuple_with_slices = (t_slice, x_slice, y_slice)

    if tfprint_tensor_shapes_and_types_for_debug and tfprint_tensor_shapes_and_types_for_function__generate_training_points_slice:
        tfprint_tensor_shape_and_type("t_slice", t_slice)
        tfprint_tensor_shape_and_type("x_slice", x_slice)
        tfprint_tensor_shape_and_type("y_slice", y_slice)

    if tfprint_tensor_values_for_debug and tfprint_tensor_values_for_function__generate_training_points_slice:
        tfprint_tensor_values("t_slice", t_slice, -1)
        tfprint_tensor_values("x_slice", x_slice, -1)
        tfprint_tensor_values("y_slice", y_slice, -1)

    return tuple_with_slices


@tf.function(jit_compile=True)
def generate_training_points_slices(
    tuple_with_eq_points,
    tuple_with_boundary_points,
    tuple_with_initial_points,
    batch_index,
    num_of_batches,
):
    print(f"Tracing generate_training_points_slices with batch_index={batch_index}.")
    tuple_with_eq_points_slice = _generate_training_points_slice(tuple_with_eq_points, batch_index, num_of_batches)

    tuple_with_bottom_boundary_points, tuple_with_top_boundary_points, tuple_with_left_boundary_points, tuple_with_right_boundary_points = \
        tuple_with_boundary_points
    tuple_with_bottom_boundary_points_slice = _generate_training_points_slice(tuple_with_bottom_boundary_points, batch_index, num_of_batches)
    tuple_with_top_boundary_points_slice    = _generate_training_points_slice(tuple_with_top_boundary_points,    batch_index, num_of_batches)
    tuple_with_left_boundary_points_slice   = _generate_training_points_slice(tuple_with_left_boundary_points,   batch_index, num_of_batches)
    tuple_with_right_boundary_points_slice  = _generate_training_points_slice(tuple_with_right_boundary_points,  batch_index, num_of_batches)
    tuple_with_boundary_points_slice = (tuple_with_bottom_boundary_points_slice,
                                        tuple_with_top_boundary_points_slice,
                                        tuple_with_left_boundary_points_slice,
                                        tuple_with_right_boundary_points_slice)

    tuple_with_initial_points_slice = _generate_training_points_slice(tuple_with_initial_points, batch_index, num_of_batches)

    return (tuple_with_eq_points_slice, tuple_with_boundary_points_slice, tuple_with_initial_points_slice)


def _calculate_approximation_of_integral_with_final_time_fit(
    u_nn,
    desired_function_at_final_time,
    X,
    Y,
    T,
):
    print("Tracing _calculate_approximation_of_integral_with_final_time_fit.")
    x_evaluation_points = ( X[:-1, :-1, -1] + X[1:, 1:, -1] ) / 2
    y_evaluation_points = ( Y[:-1, :-1, -1] + Y[1:, 1:, -1] ) / 2
    t_evaluation_points = ( T[:-1, :-1, -1] + T[1:, 1:, -1] ) / 2

    x_evaluation_points = tf.reshape(x_evaluation_points, shape=[-1, 1])
    y_evaluation_points = tf.reshape(y_evaluation_points, shape=[-1, 1])
    t_evaluation_points = tf.reshape(t_evaluation_points, shape=[-1, 1])

    txy_evaluation_points = tf.concat([t_evaluation_points, x_evaluation_points, y_evaluation_points,], axis=-1)

    # Because X, Y and T are created from linspace, so equally spaced points,
    #   we can calculate the distance between neighbouring points on the grid (=step) as difference between any points,
    #   for example points with index 0 and 1.
    x_step = X[0, 0, 0] - X[1, 0, 0]
    y_step = Y[0, 0, 0] - Y[0, 1, 0]
    xy_grid_square_area = x_step * y_step

    u_Txy = u_nn(txy_evaluation_points, training=True)
    desired_function_at_final_time_at_xy = desired_function_at_final_time(x_evaluation_points, y_evaluation_points)

    approximation_of_integral_with_final_time_fit = tf.math.reduce_sum(tf.math.squared_difference(u_Txy, desired_function_at_final_time_at_xy))
    approximation_of_integral_with_final_time_fit = xy_grid_square_area * approximation_of_integral_with_final_time_fit

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
        print_tensor_shape_and_type(f"{approximation_of_integral_with_final_time_fit=}")
        print()

    return approximation_of_integral_with_final_time_fit


@tf.function(jit_compile=True, reduce_retracing=True)
def _calculate_approximation_of_boundary_integral_for_given_points(
    u_nn,
    x_evaluation_points,
    y_evaluation_points,
    t_evaluation_points,
    boundary_grid_square_area,
):
    print("Tracing _calculate_approximation_of_boundary_integral_for_given_points.")
    x_evaluation_points = tf.reshape(x_evaluation_points, shape=[-1, 1])
    y_evaluation_points = tf.reshape(y_evaluation_points, shape=[-1, 1])
    t_evaluation_points = tf.reshape(t_evaluation_points, shape=[-1, 1])

    # tf.print("x_evaluation_points shape:", x_evaluation_points.shape)
    # tf.print("y_evaluation_points shape:", y_evaluation_points.shape)
    # tf.print("t_evaluation_points shape:", t_evaluation_points.shape)

    txy_evaluation_points = tf.concat([t_evaluation_points, x_evaluation_points, y_evaluation_points,], axis=-1)

    u_nn_values = u_nn(txy_evaluation_points, training=True)
    approximation_of_boundary_integral = boundary_grid_square_area * tf.reduce_sum(tf.math.abs(u_nn_values))

    # if print_debug_information:
    #     print(f"BOUNDARY: {boundary_part}")
    #     print_tensor_for_debug(f"{x_evaluation_points=}")
    #     print_tensor_for_debug(f"{y_evaluation_points=}")
    #     print_tensor_for_debug(f"{t_evaluation_points=}")
    #     print_tensor_for_debug(f"{boundary_grid_square_area=}")
    #     print_tensor_for_debug(f"{approximation_of_boundary_integral=}")
    #     print()

    return approximation_of_boundary_integral


def _calculate_approximation_of_boundary_integral_for_left_boundary_part(
    u_nn,
    X,
    Y,
    T,
):
    print("Tracing _calculate_approximation_of_boundary_integral_for_left_boundary_part.")
    x_evaluation_points = (X[0, :-1, :-1] + X[0, 1:, 1:]) / 2
    y_evaluation_points = (Y[0, :-1, :-1] + Y[0, 1:, 1:]) / 2
    t_evaluation_points = (T[0, :-1, :-1] + T[0, 1:, 1:]) / 2

    # Because X, Y and T are created from linspace, so equally spaced points,
    #   we can calculate the distance between neighbouring points on the grid (=step) as difference between any points,
    #   for example points with index 0 and 1.
    y_step = Y[0, 0, 0] - Y[0, 1, 0]
    t_step = T[0, 0, 0] - T[0, 0, 1]
    boundary_grid_square_area = y_step * t_step

    return _calculate_approximation_of_boundary_integral_for_given_points(u_nn,
                                                                          x_evaluation_points,
                                                                          y_evaluation_points,
                                                                          t_evaluation_points,
                                                                          boundary_grid_square_area)


def _calculate_approximation_of_boundary_integral_for_right_boundary_part(
    u_nn,
    X,
    Y,
    T,
):
    print("Tracing _calculate_approximation_of_boundary_integral_for_right_boundary_part.")
    x_evaluation_points = (X[-1, :-1, :-1] + X[-1, 1:, 1:]) / 2
    y_evaluation_points = (Y[-1, :-1, :-1] + Y[-1, 1:, 1:]) / 2
    t_evaluation_points = (T[-1, :-1, :-1] + T[-1, 1:, 1:]) / 2

    y_step = Y[-1, 0, 0] - Y[-1, 1, 0]
    t_step = T[-1, 0, 0] - T[-1, 0, 1]
    boundary_grid_square_area = y_step * t_step

    return _calculate_approximation_of_boundary_integral_for_given_points(u_nn,
                                                                          x_evaluation_points,
                                                                          y_evaluation_points,
                                                                          t_evaluation_points,
                                                                          boundary_grid_square_area)


def _calculate_approximation_of_boundary_integral_for_bottom_boundary_part(
    u_nn,
    X,
    Y,
    T,
):
    print("Tracing _calculate_approximation_of_boundary_integral_for_bottom_boundary_part.")
    x_evaluation_points = (X[:-1, 0, :-1] + X[1:, 0, 1:]) / 2
    y_evaluation_points = (Y[:-1, 0, :-1] + Y[1:, 0, 1:]) / 2
    t_evaluation_points = (T[:-1, 0, :-1] + T[1:, 0, 1:]) / 2

    x_step = X[0, 0, 0] - X[1, 0, 0]
    t_step = T[0, 0, 0] - T[0, 0, 1]
    boundary_grid_square_area = x_step * t_step

    return _calculate_approximation_of_boundary_integral_for_given_points(u_nn,
                                                                          x_evaluation_points,
                                                                          y_evaluation_points,
                                                                          t_evaluation_points,
                                                                          boundary_grid_square_area)


def _calculate_approximation_of_boundary_integral_for_top_boundary_part(
    u_nn,
    X,
    Y,
    T,
):
    print("Tracing _calculate_approximation_of_boundary_integral_for_top_boundary_part.")
    x_evaluation_points = (X[:-1, -1, :-1] + X[1:, -1, 1:]) / 2
    y_evaluation_points = (Y[:-1, -1, :-1] + Y[1:, -1, 1:]) / 2
    t_evaluation_points = (T[:-1, -1, :-1] + T[1:, -1, 1:]) / 2

    x_step = X[0, -1, 0] - X[1, -1, 0]
    t_step = T[0, -1, 0] - T[0, -1, 1]
    boundary_grid_square_area = x_step * t_step

    return _calculate_approximation_of_boundary_integral_for_given_points(u_nn,
                                                                          x_evaluation_points,
                                                                          y_evaluation_points,
                                                                          t_evaluation_points,
                                                                          boundary_grid_square_area)


# TODO: Podívat se, jak dlouho trvá tohle a jestli by to nešlo zrychlit.
@tf.function(jit_compile=True, reduce_retracing=True)
def calculate_functional_loss(
    u_nn,
    q_nn,
    desired_function_at_final_time,
    alpha,
    X,
    Y,
    T,
):
    print("Tracing calculate_functional_loss.")
    approximation_of_integral_with_final_time_fit = _calculate_approximation_of_integral_with_final_time_fit(
        u_nn, desired_function_at_final_time, X, Y, T)

    # TODO: Fitování neuronky na finální funkci bylo horší s tím okrajovým integrálem - zeptat se, zda je opravdu v tom cost funkcionálu potřeba.
    #   A provést více testů s timhle členem a bez něj, aby se zjistilo, jak moc má na fitování a jiný výsledky vliv.
    # approximation_of_boundary_integral_at_left_boundary = _calculate_approximation_of_boundary_integral_for_left_boundary_part(u_nn, X, Y, T)

    # approximation_of_boundary_integral_at_right_boundary = _calculate_approximation_of_boundary_integral_for_right_boundary_part(u_nn, X, Y, T)

    # approximation_of_boundary_integral_at_bottom_boundary = _calculate_approximation_of_boundary_integral_for_bottom_boundary_part(u_nn, X, Y, T)

    # approximation_of_boundary_integral_at_top_boundary = _calculate_approximation_of_boundary_integral_for_top_boundary_part(u_nn, X, Y, T)

    # approximation_of_boundary_integral = (approximation_of_boundary_integral_at_left_boundary
    #                                       + approximation_of_boundary_integral_at_right_boundary
    #                                       + approximation_of_boundary_integral_at_bottom_boundary
    #                                       + approximation_of_boundary_integral_at_top_boundary)

    cost_functional_loss_term_value = 1/2 * approximation_of_integral_with_final_time_fit  # + alpha/2 * approximation_of_boundary_integral

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
        u_pred = tf.reshape(u_pred, [-1])

        u_x = derivative_tape.gradient(u_pred, x_eq)
        u_y = derivative_tape.gradient(u_pred, y_eq)
    u_xx = derivative_tape.gradient(u_x, x_eq)
    u_yy = derivative_tape.gradient(u_y, y_eq)
    u_t = derivative_tape.gradient(u_pred, t_eq)

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
        tfprint_tensor_values("t_eq", t_eq, -1)
        tfprint_tensor_values("x_eq", x_eq, -1)
        tfprint_tensor_values("y_eq", y_eq, -1)
        tfprint_tensor_values("txy_eq", txy_eq, -1)
        tfprint_tensor_values("u_pred", u_pred, -1)
        tfprint_tensor_values("u_x", u_x, -1)
        tfprint_tensor_values("u_y", u_y, -1)
        tfprint_tensor_values("u_xx", u_xx, -1)
        tfprint_tensor_values("u_yy", u_yy, -1)
        tfprint_tensor_values("u_t", u_t, -1)

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
    equation_residual = u_t - heat_coef * u_xx - heat_coef * u_yy - RHS

    if tfprint_tensor_shapes_and_types_for_debug and tfprint_tensor_shapes_and_types_for_function__calculate_equation_residual:
        tfprint_tensor_shape_and_type("RHS                                      ", RHS)
        tfprint_tensor_shape_and_type("u_t - heat_coef * u_xx - heat_coef * u_yy", u_t - heat_coef * u_xx - heat_coef * u_yy)
        tfprint_tensor_shape_and_type("equation_residual                        ", equation_residual)

    if tfprint_tensor_values_for_debug and tfprint_tensor_values_for_function__calculate_equation_residual:
        tfprint_tensor_values("RHS                                      ", RHS, -1)
        tfprint_tensor_values("u_t - heat_coef * u_xx - heat_coef * u_yy", u_t - heat_coef * u_xx - heat_coef * u_yy, -1)
        tfprint_tensor_values("equation_residual                        ", equation_residual, -1)

    del u_t, u_xx, u_yy, RHS

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
    u_ini_pred = u_nn(txy_ini, training=True)
    u_ini_pred = tf.reshape(u_ini_pred, [-1])
    u_ini_target = initial_condition_function(x_ini, y_ini, u_pred, heat_coef)
    initial_condition_residual = u_ini_pred - u_ini_target

    del txy_ini, u_ini_pred, u_ini_target

    return initial_condition_residual


tfprint_tensor_shapes_and_types_for_function__calculate_boundary_residual = False
tfprint_tensor_values_for_function__calculate_boundary_residual = False


def _calculate_boundary_residual(
    boundary_points_tuple,
    u_nn,
    q_nn,
):
    boundary_points_bottom_tuple, boundary_points_top_tuple, boundary_points_left_tuple, boundary_points_right_tuple = boundary_points_tuple
    t_boundary_bottom,  x_boundary_bottom,  y_boundary_bottom   = boundary_points_bottom_tuple
    t_boundary_top,     x_boundary_top,     y_boundary_top      = boundary_points_top_tuple
    t_boundary_left,    x_boundary_left,    y_boundary_left     = boundary_points_left_tuple
    t_boundary_right,   x_boundary_right,   y_boundary_right    = boundary_points_right_tuple

    with tf.GradientTape(watch_accessed_variables=False, persistent=True) as derivative_tape:
        derivative_tape.watch(t_boundary_bottom)
        derivative_tape.watch(x_boundary_bottom)
        derivative_tape.watch(y_boundary_bottom)
        txy_boundary_bottom = tf.stack([t_boundary_bottom, x_boundary_bottom, y_boundary_bottom], axis=1)
        u_boundary_bottom_pred = u_nn(txy_boundary_bottom, training=True)

        derivative_tape.watch(t_boundary_top)
        derivative_tape.watch(x_boundary_top)
        derivative_tape.watch(y_boundary_top)
        txy_boundary_top = tf.stack([t_boundary_top, x_boundary_top, y_boundary_top], axis=1)
        u_boundary_top_pred = u_nn(txy_boundary_top, training=True)

        derivative_tape.watch(t_boundary_left)
        derivative_tape.watch(x_boundary_left)
        derivative_tape.watch(y_boundary_left)
        txy_boundary_left = tf.stack([t_boundary_left, x_boundary_left, y_boundary_left], axis=1)
        u_boundary_left_pred = u_nn(txy_boundary_left, training=True)

        derivative_tape.watch(t_boundary_right)
        derivative_tape.watch(x_boundary_right)
        derivative_tape.watch(y_boundary_right)
        txy_boundary_right = tf.stack([t_boundary_right, x_boundary_right, y_boundary_right], axis=1)
        u_boundary_right_pred = u_nn(txy_boundary_right, training=True)

    u_y_boundary_bottom = derivative_tape.gradient(u_boundary_bottom_pred, y_boundary_bottom)
    q_boundary_bottom_pred = q_nn(txy_boundary_bottom, training=True)
    boundary_bottom_residual = -u_y_boundary_bottom - tf.squeeze(q_boundary_bottom_pred)

    if tfprint_tensor_shapes_and_types_for_debug and tfprint_tensor_shapes_and_types_for_function__calculate_boundary_residual:
        tfprint_tensor_shape_and_type("txy_boundary_bottom", txy_boundary_bottom)
        tfprint_tensor_shape_and_type("u_y_boundary_bottom", u_y_boundary_bottom)
        tfprint_tensor_shape_and_type("u_boundary_bottom_pred", u_boundary_bottom_pred)
        tfprint_tensor_shape_and_type("q_boundary_bottom_pred", q_boundary_bottom_pred)
        tfprint_tensor_shape_and_type("boundary_bottom_residual", boundary_bottom_residual)

    if tfprint_tensor_values_for_debug and tfprint_tensor_values_for_function__calculate_boundary_residual:
        tfprint_tensor_values("txy_boundary_bottom", txy_boundary_bottom, -1)
        tfprint_tensor_values("u_y_boundary_bottom", u_y_boundary_bottom, -1)
        tfprint_tensor_values("u_boundary_bottom_pred", u_boundary_bottom_pred, -1)
        tfprint_tensor_values("q_boundary_bottom_pred", q_boundary_bottom_pred, -1)
        tfprint_tensor_values("boundary_bottom_residual", boundary_bottom_residual, -1)

    del txy_boundary_bottom, u_boundary_bottom_pred, u_y_boundary_bottom, q_boundary_bottom_pred


    u_y_boundary_top = derivative_tape.gradient(u_boundary_top_pred, y_boundary_top)
    q_boundary_top_pred = q_nn(txy_boundary_top, training=True)
    boundary_top_residual = u_y_boundary_top - tf.squeeze(q_boundary_top_pred)

    if tfprint_tensor_shapes_and_types_for_debug and tfprint_tensor_shapes_and_types_for_function__calculate_boundary_residual:
        tfprint_tensor_shape_and_type("txy_boundary_top", txy_boundary_top)
        tfprint_tensor_shape_and_type("u_y_boundary_top", u_y_boundary_top)
        tfprint_tensor_shape_and_type("u_boundary_top_pred", u_boundary_top_pred)
        tfprint_tensor_shape_and_type("q_boundary_top_pred", q_boundary_top_pred)
        tfprint_tensor_shape_and_type("boundary_top_residual", boundary_top_residual)

    if tfprint_tensor_values_for_debug and tfprint_tensor_values_for_function__calculate_boundary_residual:
        tfprint_tensor_values("txy_boundary_top", txy_boundary_top, -1)
        tfprint_tensor_values("u_y_boundary_top", u_y_boundary_top, -1)
        tfprint_tensor_values("u_boundary_top_pred", u_boundary_top_pred, -1)
        tfprint_tensor_values("q_boundary_top_pred", q_boundary_top_pred, -1)
        tfprint_tensor_values("boundary_top_residual", boundary_top_residual, -1)

    del txy_boundary_top, u_boundary_top_pred, u_y_boundary_top, q_boundary_top_pred


    u_x_boundary_left = derivative_tape.gradient(u_boundary_left_pred, x_boundary_left)
    q_boundary_left_pred = q_nn(txy_boundary_left, training=True)
    boundary_left_residual = -u_x_boundary_left - tf.squeeze(q_boundary_left_pred)

    if tfprint_tensor_shapes_and_types_for_debug and tfprint_tensor_shapes_and_types_for_function__calculate_boundary_residual:
        tfprint_tensor_shape_and_type("txy_boundary_left", txy_boundary_left)
        tfprint_tensor_shape_and_type("u_x_boundary_left", u_x_boundary_left)
        tfprint_tensor_shape_and_type("u_boundary_left_pred", u_boundary_left_pred)
        tfprint_tensor_shape_and_type("q_boundary_left_pred", q_boundary_left_pred)
        tfprint_tensor_shape_and_type("boundary_left_residual", boundary_left_residual)

    if tfprint_tensor_values_for_debug and tfprint_tensor_values_for_function__calculate_boundary_residual:
        tfprint_tensor_values("txy_boundary_left", txy_boundary_left, -1)
        tfprint_tensor_values("u_x_boundary_left", u_x_boundary_left, -1)
        tfprint_tensor_values("u_boundary_left_pred", u_boundary_left_pred, -1)
        tfprint_tensor_values("q_boundary_left_pred", q_boundary_left_pred, -1)
        tfprint_tensor_values("boundary_left_residual", boundary_left_residual, -1)

    del txy_boundary_left, u_boundary_left_pred, u_x_boundary_left, q_boundary_left_pred


    u_x_boundary_right = derivative_tape.gradient(u_boundary_right_pred, x_boundary_right)
    q_boundary_right_pred = q_nn(txy_boundary_right, training=True)
    boundary_right_residual = u_x_boundary_right - tf.squeeze(q_boundary_right_pred)

    if tfprint_tensor_shapes_and_types_for_debug and tfprint_tensor_shapes_and_types_for_function__calculate_boundary_residual:
        tfprint_tensor_shape_and_type("txy_boundary_right", txy_boundary_right)
        tfprint_tensor_shape_and_type("u_x_boundary_right", u_x_boundary_right)
        tfprint_tensor_shape_and_type("u_boundary_right_pred", u_boundary_right_pred)
        tfprint_tensor_shape_and_type("q_boundary_right_pred", q_boundary_right_pred)
        tfprint_tensor_shape_and_type("boundary_right_residual", boundary_right_residual)

    if tfprint_tensor_values_for_debug and tfprint_tensor_values_for_function__calculate_boundary_residual:
        tfprint_tensor_values("txy_boundary_right", txy_boundary_right, -1)
        tfprint_tensor_values("u_x_boundary_right", u_x_boundary_right, -1)
        tfprint_tensor_values("u_boundary_right_pred", u_boundary_right_pred, -1)
        tfprint_tensor_values("q_boundary_right_pred", q_boundary_right_pred, -1)
        tfprint_tensor_values("boundary_right_residual", boundary_right_residual, -1)

    del txy_boundary_right, u_boundary_right_pred, u_x_boundary_right, q_boundary_right_pred

    if print_tensor_shapes_and_types_for_debug:
        print_tensor_shape_and_type(f"{boundary_bottom_residual=}")
        print_tensor_shape_and_type(f"{boundary_top_residual=}")
        print_tensor_shape_and_type(f"{boundary_left_residual=}")
        print_tensor_shape_and_type(f"{boundary_right_residual=}")
        print()

    boundary_residual = tf.stack([boundary_bottom_residual, boundary_top_residual, boundary_left_residual, boundary_right_residual], axis=-1)

    if tfprint_tensor_shapes_and_types_for_debug and tfprint_tensor_shapes_and_types_for_function__calculate_boundary_residual:
        tfprint_tensor_shape_and_type("boundary_residual", boundary_residual)

    if tfprint_tensor_values_for_debug and tfprint_tensor_values_for_function__calculate_boundary_residual:
        tfprint_tensor_values("boundary_residual", boundary_residual, -1)

    del boundary_bottom_residual, boundary_top_residual, boundary_left_residual, boundary_right_residual

    return boundary_residual


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
    boundary_condition_weight,
    initial_condition_weight,
    cost_functional_weight,
    optimizer,
    loss_fn,
    train_only_u_nn=False,
):
    print("Tracing train step.")
    # This is here just to catch if the function was retracing in every batch call.


    with tf.GradientTape(persistent=True) as weights_update_tape:
        equation_residual, u_pred = _calculate_equation_residual(eq_points_tuple, u_nn, equation_rhs_function, heat_coef)
        equation_loss = loss_fn(equation_residual)

        initial_condition_residual = _calculate_initial_condition_residual(initial_points_tuple, u_nn, u_pred, initial_condition_function, heat_coef)
        init_cond_loss = tf.cast(initial_condition_weight, tf.float32) * loss_fn(initial_condition_residual)

        boundary_residual = _calculate_boundary_residual(boundary_points_tuple, u_nn, q_nn)
        boundary_cond_loss = tf.cast(boundary_condition_weight, tf.float32) * loss_fn(boundary_residual)

        cost_functional_loss = tf.cast(cost_functional_weight, tf.float32) * calculate_functional_loss(u_nn,
                                                                                                       q_nn,
                                                                                                       desired_function_at_final_time,
                                                                                                       alpha,
                                                                                                       X,
                                                                                                       Y,
                                                                                                       T,)


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
    boundary_condition_weight,
    initial_condition_weight,
    cost_functional_weight,
):
    print("Tracing evaluate_norms_after_training.")

    eq_points_tuple, boundary_points_tuple, initial_points_tuple = generate_eq_boundary_and_intial_points_tuples(X, Y, T)

    equation_residual, u_pred = _calculate_equation_residual(eq_points_tuple, u_nn, equation_rhs_function, heat_coef)
    L1_equation_norm = L1_norm(equation_residual)
    L2_equation_norm = L2_norm(equation_residual)
    max_equation_norm = max_norm(equation_residual)

    initial_condition_residual = _calculate_initial_condition_residual(initial_points_tuple, u_nn, u_pred, initial_condition_function, heat_coef)
    L1_init_cond_norm = tf.cast(initial_condition_weight, tf.float32) * L1_norm(initial_condition_residual)
    L2_init_cond_norm = tf.cast(initial_condition_weight, tf.float32) * L2_norm(initial_condition_residual)
    max_init_cond_norm = tf.cast(initial_condition_weight, tf.float32) * max_norm(initial_condition_residual)

    boundary_residual = _calculate_boundary_residual(boundary_points_tuple, u_nn, q_nn)
    L1_boundary_cond_norm = tf.cast(boundary_condition_weight, tf.float32) * L1_norm(boundary_residual)
    L2_boundary_cond_norm = tf.cast(boundary_condition_weight, tf.float32) * L2_norm(boundary_residual)
    max_boundary_cond_norm = tf.cast(boundary_condition_weight, tf.float32) * max_norm(boundary_residual)

    cost_functional_loss = tf.cast(cost_functional_weight, tf.float32) * calculate_functional_loss(u_nn,
                                                                                                   q_nn,
                                                                                                   desired_function_at_final_time,
                                                                                                   alpha,
                                                                                                   X,
                                                                                                   Y,
                                                                                                   T,)

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
    t_start,
    t_stop,
    num_t_training_points,
    x_start,
    x_stop,
    num_x_training_points,
    y_start,
    y_stop,
    num_y_training_points,
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
    eq_points_tuple, boundary_points_tuple, initial_points_tuple, X, Y, T = generate_training_points(t_start,
                                                                                                     t_stop,
                                                                                                     num_t_training_points,
                                                                                                     x_start,
                                                                                                     x_stop,
                                                                                                     num_x_training_points,
                                                                                                     y_start,
                                                                                                     y_stop,
                                                                                                     num_y_training_points)

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
                                        X,
                                        Y,
                                        T,
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
                                                       X,
                                                       Y,
                                                       T,
                                                       tf.constant(boundary_condition_weight),
                                                       tf.constant(initial_condition_weight),
                                                       tf.constant(cost_functional_weight))

    if callbacks is not None:
        for callback in callbacks:
            callback.on_train_end()

    return loss_history_dict, weighted_norm_dict, training_time
