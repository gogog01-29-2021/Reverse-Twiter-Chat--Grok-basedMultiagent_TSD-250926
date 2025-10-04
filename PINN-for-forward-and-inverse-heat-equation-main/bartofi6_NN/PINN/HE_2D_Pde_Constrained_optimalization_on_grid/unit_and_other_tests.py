from ..HE_2D_Pde_Constrained_optimalization_on_circle.unit_and_other_tests import get_nn_with_given_activation_function
from .HE_2D_Pde_Constrained_optimalization_on_grid import generate_training_points, shuffle_points, generate_training_points_slices, \
    _calculate_equation_residual, _calculate_initial_condition_residual, _calculate_boundary_residual
from . import HE_2D_Pde_Constrained_optimalization_on_grid as optimization_on_grid
from ...training import tfprint_tensor_shape_and_type

import tensorflow as tf


def test_training_points_generation():
    optimization_on_grid.tfprint_tensor_shapes_and_types_for_debug = True
    optimization_on_grid.tfprint_tensor_shapes_and_types_for_function_generate_training_points = True
    optimization_on_grid.tfprint_tensor_values_for_debug = True
    optimization_on_grid.tfprint_tensor_values_for_function_generate_training_points = True
    optimization_on_grid.jit_compile_point_generation = False

    t_start = 0.0
    t_stop = 1.0
    num_t_training_points = 3
    x_start = -1.0
    x_stop = 1.0
    num_of_grid_x_training_points = 5
    y_start = -1.5
    y_stop = 1.5
    num_of_grid_y_training_points = 5

    eq_points_tuple, boundary_points_tuple, initial_points_tuple, X, Y, T = generate_training_points(t_start,
                                                                                                     t_stop,
                                                                                                     num_t_training_points,
                                                                                                     x_start,
                                                                                                     x_stop,
                                                                                                     num_of_grid_x_training_points,
                                                                                                     y_start,
                                                                                                     y_stop,
                                                                                                     num_of_grid_y_training_points)


def test_shuffle_points():
    optimization_on_grid.tfprint_tensor_shapes_and_types_for_debug = True
    optimization_on_grid.tfprint_tensor_shapes_and_types_for_function_shuffle_points = True
    optimization_on_grid.tfprint_tensor_values_for_debug = True
    optimization_on_grid.tfprint_tensor_values_for_function_shuffle_points = True

    t_start = 0.0
    t_stop = 1.0
    num_t_training_points = 3
    x_start = -1.0
    x_stop = 1.0
    num_of_grid_x_training_points = 5
    y_start = -1.5
    y_stop = 1.5
    num_of_grid_y_training_points = 5

    eq_points_tuple, boundary_points_tuple, initial_points_tuple, X, Y, T = generate_training_points(t_start,
                                                                                                     t_stop,
                                                                                                     num_t_training_points,
                                                                                                     x_start,
                                                                                                     x_stop,
                                                                                                     num_of_grid_x_training_points,
                                                                                                     y_start,
                                                                                                     y_stop,
                                                                                                     num_of_grid_y_training_points)

    shuffle_points(eq_points_tuple, boundary_points_tuple, initial_points_tuple)


def test_generate_training_points_slices():
    optimization_on_grid.tfprint_tensor_shapes_and_types_for_debug = True
    optimization_on_grid.tfprint_tensor_shapes_and_types_for_function__generate_training_points_slice = True
    optimization_on_grid.tfprint_tensor_shapes_and_types_for_function_generate_training_points = True
    optimization_on_grid.tfprint_tensor_values_for_debug = True
    optimization_on_grid.tfprint_tensor_values_for_function__generate_training_points_slice = True

    t_start = 0.0
    t_stop = 1.0
    num_t_training_points = 3
    x_start = -1.0
    x_stop = 1.0
    num_of_grid_x_training_points = 10
    y_start = -1.5
    y_stop = 1.5
    num_of_grid_y_training_points = 10

    eq_points_tuple, boundary_points_tuple, initial_points_tuple, X, Y, T = generate_training_points(t_start,
                                                                                                     t_stop,
                                                                                                     num_t_training_points,
                                                                                                     x_start,
                                                                                                     x_stop,
                                                                                                     num_of_grid_x_training_points,
                                                                                                     y_start,
                                                                                                     y_stop,
                                                                                                     num_of_grid_y_training_points)

    num_of_batches = 7
    for batch_index in range(num_of_batches):
        print()
        print(f"Batch index: {batch_index}/{num_of_batches-1}")
        generate_training_points_slices(eq_points_tuple, boundary_points_tuple, initial_points_tuple, batch_index, num_of_batches)


def test__calculate_equation_residual():
    optimization_on_grid.tfprint_tensor_shapes_and_types_for_debug = True
    optimization_on_grid.tfprint_tensor_shapes_and_types_for_function__calculate_equation_residual = True
    optimization_on_grid.tfprint_tensor_values_for_debug = True
    optimization_on_grid.tfprint_tensor_values_for_function__calculate_equation_residual = True

    t = tf.constant([2.0, -3.0, 4.0, -1.0], dtype=tf.float32)
    x = tf.constant([-3.5, 2.0, -2.0, 2.0], dtype=tf.float32)
    y = tf.constant([3.0, 2.5, 3.0, -1.0], dtype=tf.float32)
    heat_coef = 0.5
    txy_tuple = (t, x, y)

    def test_calculation(test_nn, test_function, correct_residual_value):
        calculated_residual, _ = _calculate_equation_residual(txy_tuple, test_nn, test_function, heat_coef)
        try:
            tf.debugging.assert_equal(calculated_residual, correct_residual_value)
        except tf.errors.InvalidArgumentError:
            print("Status: FAIL")
            print(f"Calculated residual approximation: {calculated_residual}")
            print(f"Correct residual approximation:    {correct_residual_value}")
        else:
            print("Status: PASS")
            print(f"Calculated residual approximation: {calculated_residual}")
            print(f"Correct residual approximation:    {correct_residual_value}")


    print()
    print("f(t,x,y, u, alfa) = 0.15 t^2 - 0.0067 t^3 + 0.01 x^2 - 0.05 x - 4.5 y^3 - 0.5 y^2 + 0.054 u^2 - 0.25 u + alfa^1.6 - 10")
    print("u(t,x,y) = 0.26 t - 0.05 t^3 + 4.3 x^4 - 3.7 x^3 + 0.35 y^6 - 1.9 y^5 + 6.5")

    def test_function(t, x, y, u, heat_coef):
        tfprint_tensor_shape_and_type("t", t)
        tfprint_tensor_shape_and_type("x", x)
        tfprint_tensor_shape_and_type("y", y)
        tfprint_tensor_shape_and_type("u", u)
        return 0.15 * t**2 - 0.0067 * t**3 + 0.01 * x**2 - 0.05 * x - 4.5 * y**3 - 0.5 * y**2 + 0.054 * u**2 - 0.25 * u + heat_coef**1.6 - 10
    test_nn = get_nn_with_given_activation_function(lambda t, x, y: 0.26 * t - 0.05 * t**3 + 4.3 * x**4 - 3.7 * x**3 + 0.35 * y**6 - 1.9 * y**5 + 6.5)

    u = 0.26 * t - 0.05 * t**3 + 4.3 * x**4 - 3.7 * x**3 + 0.35 * y**6 - 1.9 * y**5 + 6.5
    correct_residual_value = (0.26 - 0.05*3 * t**2) - heat_coef*(4.3*4*3 * x**2 - 3.7*3*2 * x) - heat_coef*(0.35*6*5 * y**4 - 1.9*5*4 * y**3) \
        - (0.15 * t**2 - 0.0067 * t**3 + 0.01 * x**2 - 0.05 * x - 4.5 * y**3 - 0.5 * y**2 + 0.054 * u**2 - 0.25 * u + heat_coef**1.6 - 10)

    test_calculation(test_nn, test_function, correct_residual_value)


def test__calculate_initial_condition_residual():
    optimization_on_grid.tfprint_tensor_shapes_and_types_for_debug = True
    optimization_on_grid.tfprint_tensor_shapes_and_types_for_function__calculate_initial_condition_residual = True
    optimization_on_grid.tfprint_tensor_values_for_debug = True
    optimization_on_grid.tfprint_tensor_values_for_function__calculate_initial_condition_residual = True

    t = tf.constant([0.0, 0.0, 0.0, 0.0], dtype=tf.float32)
    x = tf.constant([-3.5, 2.0, -2.0, 2.0], dtype=tf.float32)
    y = tf.constant([3.0, 2.5, 3.0, -1.0], dtype=tf.float32)
    heat_coef = 0.5
    txy_tuple = (t, x, y)

    def test_calculation(test_nn, u_pred, test_function, correct_residual_value):
        calculated_residual = _calculate_initial_condition_residual(txy_tuple, test_nn, u_pred, test_function, heat_coef)
        try:
            tf.debugging.assert_equal(calculated_residual, correct_residual_value)
        except tf.errors.InvalidArgumentError:
            print("Status: FAIL")
            print(f"Calculated residual approximation: {calculated_residual}")
            print(f"Correct residual approximation:    {correct_residual_value}")
        else:
            print("Status: PASS")
            print(f"Calculated residual approximation: {calculated_residual}")
            print(f"Correct residual approximation:    {correct_residual_value}")


    print()
    print("f(t,x,y, u, alfa) = 0.15 t^2 - 0.0067 t^3 + 0.01 x^2 - 0.05 x - 4.5 y^3 - 0.5 y^2 + 0.054 u^2 - 0.25 u + alfa^1.6 - 10")
    print("u(t,x,y) = 0.26 t - 0.05 t^3 + 4.3 x^4 - 3.7 x^3 + 0.35 y^6 - 1.9 y^5 + 6.5")

    def test_function(x, y, u, heat_coef):
        tfprint_tensor_shape_and_type("test function: x:", x)
        tfprint_tensor_shape_and_type("test function: y:", y)
        tfprint_tensor_shape_and_type("test function: u:", u)
        return 0.01 * x**2 - 0.05 * x - 4.5 * y**3 - 0.5 * y**2 + 0.054 * u**2 - 0.25 * u + heat_coef**1.6 - 10

    test_nn = get_nn_with_given_activation_function(lambda t, x, y: 0.26 * t - 0.05 * t**3 + 4.3 * x**4 - 3.7 * x**3 + 0.35 * y**6 - 1.9 * y**5 + 6.5)

    u_0 = 4.3 * x**4 - 3.7 * x**3 + 0.35 * y**6 - 1.9 * y**5 + 6.5
    correct_residual_value = u_0 - (0.01 * x**2 - 0.05 * x - 4.5 * y**3 - 0.5 * y**2 + 0.054 * u_0**2 - 0.25 * u_0 + heat_coef**1.6 - 10)

    u_pred = test_nn(tf.stack(txy_tuple, axis=-1))
    u_pred = tf.reshape(u_pred, [-1])

    test_calculation(test_nn, u_pred, test_function, correct_residual_value)


def test__calculate_boundary_residual():
    optimization_on_grid.tfprint_tensor_shapes_and_types_for_debug = True
    optimization_on_grid.tfprint_tensor_shapes_and_types_for_function__calculate_boundary_residual = True
    optimization_on_grid.tfprint_tensor_values_for_debug = True
    optimization_on_grid.tfprint_tensor_values_for_function__calculate_boundary_residual = True

    t = tf.constant([2.0, -3.0, 4.0, -1.0], dtype=tf.float32)
    x = tf.constant([-3.5, 2.0, -2.0, 2.0], dtype=tf.float32)
    y = tf.constant([3.0, 2.5, 3.0, -1.0], dtype=tf.float32)
    txy_tuple_bottom_boundary = (t, x, y)
    txy_tuple_top_boundary = (t, x, y)
    txy_tuple_left_boundary = (t, x, y)
    txy_tuple_right_boundary = (t, x, y)
    txy_tuple = (txy_tuple_bottom_boundary, txy_tuple_top_boundary, txy_tuple_left_boundary, txy_tuple_right_boundary)

    def test_calculation(test_unn, test_qnn, correct_residual_value):
        calculated_residual = _calculate_boundary_residual(txy_tuple, test_unn, test_qnn)
        try:
            tf.debugging.assert_equal(calculated_residual, correct_residual_value)
        except tf.errors.InvalidArgumentError:
            print("Status: FAIL")
            print(f"Calculated residual approximation: {calculated_residual}")
            print(f"Correct residual approximation:    {correct_residual_value}")
        else:
            print("Status: PASS")
            print(f"Calculated residual approximation: {calculated_residual}")
            print(f"Correct residual approximation:    {correct_residual_value}")

    test_unn = get_nn_with_given_activation_function(
        lambda t, x, y: 0.26 * t - 0.05 * t**3 + 4.3 * x**4 - 3.7 * x**3 + 0.35 * y**6 - 1.9 * y**5 + 6.5)
    test_qnn = get_nn_with_given_activation_function(
        lambda t, x, y: 0.15 * t**2 - 0.0067 * t**3 + 0.01 * x**2 - 0.05 * x - 4.5 * y**3 - 0.5 * y**2 - 10)

    q = 0.15 * t**2 - 0.0067 * t**3 + 0.01 * x**2 - 0.05 * x - 4.5 * y**3 - 0.5 * y**2 - 10
    correct_residual_value_bottom_boundary = -(0.35*6 * y**5 - 1.9*5 * y**4) - q
    correct_residual_value_top_boundary = (0.35*6 * y**5 - 1.9*5 * y**4) - q
    correct_residual_value_left_boundary = -(4.3*4 * x**3 - 3.7*3 * x**2) - q
    correct_residual_value_right_boundary = (4.3*4 * x**3 - 3.7*3 * x**2) - q
    correct_residual_value = tf.stack([correct_residual_value_bottom_boundary,
                                       correct_residual_value_top_boundary,
                                       correct_residual_value_left_boundary,
                                       correct_residual_value_right_boundary],
                                      axis=-1)

    test_calculation(test_unn, test_qnn, correct_residual_value)
