import tensorflow as tf
import math
from .HE_2D_Pde_Constrained_optimalization_correct_setup import _calculate_q_nn_cooling_soft_penalty, \
    _calculate_q_nn_values_for_cooling_soft_penalty, \
    generate_points_for_evaluation_of_function_fit_integral, \
    shuffle_points_for_integral_evaluation, \
    slice_points_for_integral_evaluation, \
    calculate_approximation_of_function_fit_integral
from ..HE_2D_Pde_Constrained_optimalization_on_circle.unit_and_other_tests import get_nn_with_given_activation_function
from . import HE_2D_Pde_Constrained_optimalization_correct_setup as correct_optimization_setup


def test__calculate_q_nn_cooling_soft_penalty():
    t1 = tf.constant([2.0, -3.0, 4.0, -1.0], dtype=tf.float32)
    x1 = tf.constant([-3.5, 2.0, -2.0, 2.0], dtype=tf.float32)
    y1 = tf.constant([3.0, 2.5, 3.0, -1.0], dtype=tf.float32)
    txy_tuple_bottom_boundary = (t1, x1, y1)
    txy_tuple_left_boundary = (t1, x1, y1)

    t2 = tf.constant([2.5, -4.0, 2.0, -1.0], dtype=tf.float32)
    x2 = tf.constant([-3.0, 1.5, -2.0, 3.0], dtype=tf.float32)
    y2 = tf.constant([2.3, 4.0, 1.5, -0.5], dtype=tf.float32)
    txy_tuple_top_boundary = (t2, x2, y2)
    txy_tuple_right_boundary = (t2, x2, y2)
    txy_boundary_points_tuple = (txy_tuple_bottom_boundary, txy_tuple_top_boundary, txy_tuple_left_boundary, txy_tuple_right_boundary)

    def test_calculation(test_qnn, correct_penalty_value):
        calculated_penalty = _calculate_q_nn_cooling_soft_penalty(test_qnn, txy_boundary_points_tuple)
        try:
            tf.debugging.assert_equal(calculated_penalty, correct_penalty_value)
        except tf.errors.InvalidArgumentError:
            print("Status: FAIL")
            print(f"Calculated residual approximation: {calculated_penalty}")
            print(f"Correct residual approximation:    {correct_penalty_value}")
        else:
            print("Status: PASS")
            print(f"Calculated residual approximation: {calculated_penalty}")
            print(f"Correct residual approximation:    {correct_penalty_value}")

    test_qnn = get_nn_with_given_activation_function(
        lambda t, x, y: 0.15 * t**2 - 0.0067 * t**3 + 0.01 * x**2 - 0.05 * x - 4.5 * y**3 - 0.5 * y**2 - 10)

    q_bottom = 0.15 * t1**2 - 0.0067 * t1**3 + 0.01 * x1**2 - 0.05 * x1 - 4.5 * y1**3 - 0.5 * y1**2 - 10
    q_left = 0.15 * t1**2 - 0.0067 * t1**3 + 0.01 * x1**2 - 0.05 * x1 - 4.5 * y1**3 - 0.5 * y1**2 - 10
    q_top = 0.15 * t2**2 - 0.0067 * t2**3 + 0.01 * x2**2 - 0.05 * x2 - 4.5 * y2**3 - 0.5 * y2**2 - 10
    q_right = 0.15 * t2**2 - 0.0067 * t2**3 + 0.01 * x2**2 - 0.05 * x2 - 4.5 * y2**3 - 0.5 * y2**2 - 10

    q = tf.concat([q_bottom, q_top, q_left, q_right], axis=0)

    only_negative_q_values_or_zero = tf.math.minimum(q, tf.zeros_like(q))
    correct_penalty_value = tf.math.reduce_mean(tf.math.square(only_negative_q_values_or_zero))

    test_calculation(test_qnn, correct_penalty_value)


def test__calculate_q_nn_values_for_cooling_soft_penalty():
    t1 = tf.constant([2.0, -3.0, 4.0, -1.0], dtype=tf.float32)
    x1 = tf.constant([-3.5, 2.0, -2.0, 2.0], dtype=tf.float32)
    y1 = tf.constant([3.0, 2.5, 3.0, -1.0], dtype=tf.float32)
    txy_tuple_bottom_boundary = (t1, x1, y1)
    txy_tuple_left_boundary = (t1, x1, y1)

    t2 = tf.constant([2.5, -4.0, 2.0, -1.0], dtype=tf.float32)
    x2 = tf.constant([-3.0, 1.5, -2.0, 3.0], dtype=tf.float32)
    y2 = tf.constant([2.3, 4.0, 1.5, -0.5], dtype=tf.float32)
    txy_tuple_top_boundary = (t2, x2, y2)
    txy_tuple_right_boundary = (t2, x2, y2)
    txy_boundary_points_tuple = (txy_tuple_bottom_boundary, txy_tuple_top_boundary, txy_tuple_left_boundary, txy_tuple_right_boundary)

    def test_calculation(test_qnn, correct_q_nn_value):
        calculated_q_nn_values = _calculate_q_nn_values_for_cooling_soft_penalty(test_qnn, txy_boundary_points_tuple)
        try:
            tf.debugging.assert_equal(calculated_q_nn_values, correct_q_nn_value)
        except tf.errors.InvalidArgumentError:
            print("Status: FAIL")
            print(f"Calculated residual approximation: {calculated_q_nn_values}")
            print(f"Correct residual approximation:    {correct_q_nn_value}")
        else:
            print("Status: PASS")
            print(f"Calculated residual approximation: {calculated_q_nn_values}")
            print(f"Correct residual approximation:    {correct_q_nn_value}")

    test_qnn = get_nn_with_given_activation_function(
        lambda t, x, y: 0.15 * t**2 - 0.0067 * t**3 + 0.01 * x**2 - 0.05 * x - 4.5 * y**3 - 0.5 * y**2 - 10)

    q_bottom = 0.15 * t1**2 - 0.0067 * t1**3 + 0.01 * x1**2 - 0.05 * x1 - 4.5 * y1**3 - 0.5 * y1**2 - 10
    q_left = 0.15 * t1**2 - 0.0067 * t1**3 + 0.01 * x1**2 - 0.05 * x1 - 4.5 * y1**3 - 0.5 * y1**2 - 10
    q_top = 0.15 * t2**2 - 0.0067 * t2**3 + 0.01 * x2**2 - 0.05 * x2 - 4.5 * y2**3 - 0.5 * y2**2 - 10
    q_right = 0.15 * t2**2 - 0.0067 * t2**3 + 0.01 * x2**2 - 0.05 * x2 - 4.5 * y2**3 - 0.5 * y2**2 - 10

    q = tf.concat([q_bottom, q_top, q_left, q_right], axis=0)
    only_negative_q_values_or_zero = tf.math.minimum(q, tf.zeros_like(q))

    test_calculation(test_qnn, only_negative_q_values_or_zero)


def test_generate_points_for_evaluation_of_function_fit_integral():
    correct_optimization_setup.tfprint_tensor_shapes_and_types_for_debug = True
    correct_optimization_setup.tfprint_tensor_shapes_and_types_for_function_generate_points_for_evaluation_of_function_fit_integral = True
    correct_optimization_setup.tfprint_tensor_values_for_debug = True
    correct_optimization_setup.tfprint_tensor_values_for_function_generate_points_for_evaluation_of_function_fit_integral = True

    t_final = 1.134
    x_start = 0.014
    x_stop = 0.987
    num_of_x_points = 3
    y_start = -0.0054
    y_stop = 1.1068
    num_of_y_points = 3
    circle_radius = 1.0
    circle_center_in_xy = [0.0, 0.0]
    dtype = tf.float32

    generate_points_for_evaluation_of_function_fit_integral(t_final,
                                                            x_start,
                                                            x_stop,
                                                            num_of_x_points,
                                                            y_start,
                                                            y_stop,
                                                            num_of_y_points,
                                                            circle_radius,
                                                            circle_center_in_xy,
                                                            dtype)


def test_calculate_approximation_of_function_fit_integral():
    correct_optimization_setup.tfprint_tensor_shapes_and_types_for_debug = True
    correct_optimization_setup.tfprint_tensor_shapes_and_types_for_function_generate_points_for_evaluation_of_function_fit_integral = True
    correct_optimization_setup.tfprint_tensor_values_for_debug = True
    correct_optimization_setup.tfprint_tensor_values_for_function_generate_points_for_evaluation_of_function_fit_integral = True

    final_time = 2.0
    x_start = 2.0
    x_stop = 4.0
    num_of_x_points = 3
    y_start = -4.5
    y_stop = -3.5
    num_of_y_points = 2
    circle_radius = 2.0
    circle_center_in_xy = [3.0, -4.0]
    dtype = tf.float32

    txy_evaluation_points, radial_distance, polar_angle, xy_grid_cell_area = generate_points_for_evaluation_of_function_fit_integral(
                                                                                    final_time,
                                                                                    x_start,
                                                                                    x_stop,
                                                                                    num_of_x_points,
                                                                                    y_start,
                                                                                    y_stop,
                                                                                    num_of_y_points,
                                                                                    circle_radius,
                                                                                    circle_center_in_xy,
                                                                                    dtype)

    print(txy_evaluation_points)
    print(radial_distance)
    print(polar_angle)
    print(xy_grid_cell_area)

    def test_calculation(test_function, test_nn, correct_integral_approximation_value):
        calculated_integral_approximation = calculate_approximation_of_function_fit_integral(test_nn,
                                                                                             test_function,
                                                                                             txy_evaluation_points,
                                                                                             radial_distance,
                                                                                             polar_angle,
                                                                                             xy_grid_cell_area)
        try:
            tf.debugging.assert_equal(calculated_integral_approximation, correct_integral_approximation_value)
        except tf.errors.InvalidArgumentError:
            print("Status: FAIL")
            print(f"Calculated integral approximation: {calculated_integral_approximation}")
            print(f"Correct integral approximation:    {correct_integral_approximation_value}")
        else:
            print("Status: PASS")
            print(f"Calculated integral approximation: {calculated_integral_approximation}")
            print(f"Correct integral approximation:    {correct_integral_approximation_value}")


    print()
    print("Testing calculate_approximation_of_function_fit_integral:")


    print("Fitted function: f(x,y) = r^2 sin(2θ)")
    print("Neural network:  u(t,x,y) = 5 t^3 - 4 x^2 + 6 y^4")
    test_function = lambda r, angle: r**2 * tf.math.sin(2*angle)
    test_nn = get_nn_with_given_activation_function(lambda t, x, y: 5.0 * t**3 - 4.0 * x**2 + 6.0 * y**4)
    correct_integral_approximation_value = xy_grid_cell_area * (
        + (5.0 * final_time**3 - 4.0 * (2.0)**2 + 6.0 * (-4.5)**4 - ((-1.0 / circle_radius)**2 + (-0.5 / circle_radius)**2) * math.sin(2 * math.atan2(-0.5, -1.0)))**2
        + (5.0 * final_time**3 - 4.0 * (3.0)**2 + 6.0 * (-4.5)**4 - ((0.0  / circle_radius)**2 + (-0.5 / circle_radius)**2) * math.sin(2 * math.atan2(-0.5, 0.0))) **2
        + (5.0 * final_time**3 - 4.0 * (4.0)**2 + 6.0 * (-4.5)**4 - ((1.0  / circle_radius)**2 + (-0.5 / circle_radius)**2) * math.sin(2 * math.atan2(-0.5, 1.0))) **2
        + (5.0 * final_time**3 - 4.0 * (2.0)**2 + 6.0 * (-3.5)**4 - ((-1.0 / circle_radius)**2 + (0.5  / circle_radius)**2) * math.sin(2 * math.atan2(0.5, -1.0))) **2
        + (5.0 * final_time**3 - 4.0 * (3.0)**2 + 6.0 * (-3.5)**4 - ((0.0  / circle_radius)**2 + (0.5  / circle_radius)**2) * math.sin(2 * math.atan2(0.5, 0.0)))  **2
        + (5.0 * final_time**3 - 4.0 * (4.0)**2 + 6.0 * (-3.5)**4 - ((1.0  / circle_radius)**2 + (0.5  / circle_radius)**2) * math.sin(2 * math.atan2(0.5, 1.0)))  **2
    )
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print("Fitted function: f(x,y) = r^2 sin(2θ)")
    print("Neural network:  u(t,x,y) = 3 x^0.5 - 2 y^3")
    test_function = lambda r, angle: r**2 * tf.math.sin(2*angle)
    test_nn = get_nn_with_given_activation_function(lambda t, x, y: 3.0 * x**0.5 - 2.0 * y**3)
    correct_integral_approximation_value = xy_grid_cell_area * (
        + (3.0 * (2.0)**0.5 - 2.0 * (-4.5)**3 - ((-1.0 / circle_radius)**2 + (-0.5 / circle_radius)**2) * math.sin(2 * math.atan2(-0.5, -1.0)))**2
        + (3.0 * (3.0)**0.5 - 2.0 * (-4.5)**3 - ((0.0  / circle_radius)**2 + (-0.5 / circle_radius)**2) * math.sin(2 * math.atan2(-0.5, 0.0))) **2
        + (3.0 * (4.0)**0.5 - 2.0 * (-4.5)**3 - ((1.0  / circle_radius)**2 + (-0.5 / circle_radius)**2) * math.sin(2 * math.atan2(-0.5, 1.0))) **2
        + (3.0 * (2.0)**0.5 - 2.0 * (-3.5)**3 - ((-1.0 / circle_radius)**2 + (0.5  / circle_radius)**2) * math.sin(2 * math.atan2(0.5, -1.0))) **2
        + (3.0 * (3.0)**0.5 - 2.0 * (-3.5)**3 - ((0.0  / circle_radius)**2 + (0.5  / circle_radius)**2) * math.sin(2 * math.atan2(0.5, 0.0)))  **2
        + (3.0 * (4.0)**0.5 - 2.0 * (-3.5)**3 - ((1.0  / circle_radius)**2 + (0.5  / circle_radius)**2) * math.sin(2 * math.atan2(0.5, 1.0)))  **2
    )
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print("Fitted function: f(x,y) = 2 r^2 - 1")
    print("Neural network:  u(t,x,y) = 3 x^0.5 - 2 y^3")
    test_function = lambda r, angle: 2 * r**2 - 1
    test_nn = get_nn_with_given_activation_function(lambda t, x, y: 3.0 * x**0.5 - 2.0 * y**3)
    correct_integral_approximation_value = xy_grid_cell_area * (
        + (3.0 * (2.0)**0.5 - 2.0 * (-4.5)**3 - (2 * ((-1.0 / circle_radius)**2 + (-0.5 / circle_radius)**2) - 1))**2
        + (3.0 * (3.0)**0.5 - 2.0 * (-4.5)**3 - (2 * ((0.0  / circle_radius)**2 + (-0.5 / circle_radius)**2) - 1))**2
        + (3.0 * (4.0)**0.5 - 2.0 * (-4.5)**3 - (2 * ((1.0  / circle_radius)**2 + (-0.5 / circle_radius)**2) - 1))**2
        + (3.0 * (2.0)**0.5 - 2.0 * (-3.5)**3 - (2 * ((-1.0 / circle_radius)**2 + (0.5  / circle_radius)**2) - 1))**2
        + (3.0 * (3.0)**0.5 - 2.0 * (-3.5)**3 - (2 * ((0.0  / circle_radius)**2 + (0.5  / circle_radius)**2) - 1))**2
        + (3.0 * (4.0)**0.5 - 2.0 * (-3.5)**3 - (2 * ((1.0  / circle_radius)**2 + (0.5  / circle_radius)**2) - 1))**2
    )
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


def test_shuffle_points_for_integral_evaluation():
    t_final = 1.134
    x_start = 0.014
    x_stop = 0.987
    num_of_x_points = 3
    y_start = -0.0054
    y_stop = 1.1068
    num_of_y_points = 3
    circle_radius = 10.0
    circle_center_in_xy = [0.0, 0.0]
    dtype = tf.float32

    txy_evaluation_points, radial_distance, polar_angle, xy_grid_square_area = \
        generate_points_for_evaluation_of_function_fit_integral(t_final,
                                                                x_start,
                                                                x_stop,
                                                                num_of_x_points,
                                                                y_start,
                                                                y_stop,
                                                                num_of_y_points,
                                                                circle_radius,
                                                                circle_center_in_xy,
                                                                dtype)

    shuffled_txy_evaluation_points, shuffled_radial_distance, shuffled_polar_angle = \
        shuffle_points_for_integral_evaluation(
            txy_evaluation_points,
            radial_distance,
            polar_angle)

    print(txy_evaluation_points)
    print(shuffled_txy_evaluation_points)
    print(radial_distance)
    print(shuffled_radial_distance)
    print(polar_angle)
    print(shuffled_polar_angle)

    txy_evaluation_points, radial_distance, polar_angle, xy_grid_square_area = \
        generate_points_for_evaluation_of_function_fit_integral(t_final,
                                                                x_start,
                                                                x_stop,
                                                                num_of_x_points,
                                                                y_start,
                                                                y_stop,
                                                                num_of_y_points,
                                                                circle_radius,
                                                                circle_center_in_xy,
                                                                dtype)

    shuffled_txy_evaluation_points, shuffled_radial_distance, shuffled_polar_angle = \
        shuffle_points_for_integral_evaluation(
            txy_evaluation_points,
            radial_distance,
            polar_angle)

    print("\nSecond run")
    print(shuffled_txy_evaluation_points)
    print(shuffled_radial_distance)
    print(shuffled_polar_angle)

    txy_evaluation_points, radial_distance, polar_angle, xy_grid_square_area = \
        generate_points_for_evaluation_of_function_fit_integral(t_final,
                                                                x_start,
                                                                x_stop,
                                                                num_of_x_points,
                                                                y_start,
                                                                y_stop,
                                                                num_of_y_points,
                                                                circle_radius,
                                                                circle_center_in_xy,
                                                                dtype)

    shuffled_txy_evaluation_points, shuffled_radial_distance, shuffled_polar_angle = \
        shuffle_points_for_integral_evaluation(
            txy_evaluation_points,
            radial_distance,
            polar_angle)

    print("\nThird run")
    print(shuffled_txy_evaluation_points)
    print(shuffled_radial_distance)
    print(shuffled_polar_angle)


def test_slice_points_for_integral_evaluation():
    t_final = 1.134
    x_start = 0.014
    x_stop = 0.987
    num_of_x_points = 100
    y_start = -0.0054
    y_stop = 1.1068
    num_of_y_points = 100
    circle_radius = 10.0
    circle_center_in_xy = [0.0, 0.0]
    dtype = tf.float32

    txy_evaluation_points, radial_distance, polar_angle, xy_grid_square_area = \
        generate_points_for_evaluation_of_function_fit_integral(t_final,
                                                                x_start,
                                                                x_stop,
                                                                num_of_x_points,
                                                                y_start,
                                                                y_stop,
                                                                num_of_y_points,
                                                                circle_radius,
                                                                circle_center_in_xy,
                                                                dtype)

    print(f"txy_evaluation_points: {txy_evaluation_points}")
    print(f"radial_distance: {radial_distance}")
    print(f"polar_angle: {polar_angle}")
    num_of_batches = 15
    for batch_index in range(num_of_batches):
        sliced_txy_evaluation_points, sliced_radial_distance, sliced_polar_angle = \
            slice_points_for_integral_evaluation(
                txy_evaluation_points,
                radial_distance,
                polar_angle,
                tf.constant(batch_index),
                num_of_batches)

        print(f"\nBatch {batch_index+1}/{num_of_batches}:")
        print(sliced_txy_evaluation_points.shape)
        print(sliced_radial_distance.shape)
        print(sliced_polar_angle.shape)
