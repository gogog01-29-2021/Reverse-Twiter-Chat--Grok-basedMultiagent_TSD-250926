from .HE_2D_Pde_Constrained_optimalization_on_circle import *
from .HE_2D_Pde_Constrained_optimalization_on_circle import _calculate_approximation_of_integral_with_final_time_fit, _calculate_equation_residual, _calculate_initial_condition_residual, _calculate_boundary_residual
from . import HE_2D_Pde_Constrained_optimalization_on_circle as optimization_on_circle
import tensorflow as tf
import math


def write_values_of_tensors_from_txy_tuple(tuple, name_appendix):
    t, x, y = tuple
    tfprint_tensor_values("t" + name_appendix, t, -1)
    tfprint_tensor_values("x" + name_appendix, x, -1)
    tfprint_tensor_values("y" + name_appendix, y, -1)


def get_nn_that_returns_constant(constant):
    nn = tf.keras.Sequential()
    nn.add(tf.keras.Input(shape=(3,)))
    output_layer = tf.keras.layers.Dense(1,
                                         kernel_initializer="zeros",
                                         bias_initializer=tf.keras.initializers.Constant(constant),
                                         dtype=tf.float32,
                                         trainable=False)
    nn.add(output_layer)
    return nn


def get_nn_with_weights_initialized_to_zero():
    nn = tf.keras.Sequential()
    nn.add(tf.keras.Input(shape=(3,)))
    output_layer = tf.keras.layers.Dense(1, kernel_initializer="zeros", trainable=False)
    nn.add(output_layer)
    return nn


def get_nn_that_returns_t():
    nn = get_nn_with_weights_initialized_to_zero()
    output_layer = nn.get_layer(index=0)
    output_layer_kernel_weights = output_layer.weights[0]
    output_layer_kernel_weights.assign([[1], [0], [0]])
    return nn


def get_nn_that_returns_x():
    nn = get_nn_with_weights_initialized_to_zero()
    output_layer = nn.get_layer(index=0)
    output_layer_kernel_weights = output_layer.weights[0]
    output_layer_kernel_weights.assign([[0], [1], [0]])
    return nn


def get_nn_that_returns_y():
    nn = get_nn_with_weights_initialized_to_zero()
    output_layer = nn.get_layer(index=0)
    output_layer_kernel_weights = output_layer.weights[0]
    output_layer_kernel_weights.assign([[0], [0], [1]])
    return nn


class Output_dense_layer_with_activation_function(tf.keras.layers.Layer):
    def __init__(self, activation_function):
        """
        activation function takes arguments "t, x, y"
        """
        super().__init__()
        self.activation_function = activation_function

    def call(self, txy):
        t = txy[:, 0]
        x = txy[:, 1]
        y = txy[:, 2]

        return tf.expand_dims(self.activation_function(t, x, y), 1)


def get_nn_with_given_activation_function(activation_function):
    nn = tf.keras.Sequential()
    nn.add(tf.keras.Input(shape=(3,)))
    output_layer = Output_dense_layer_with_activation_function(activation_function)
    nn.add(output_layer)
    return nn


def test_generate_training_points_in_circle_in_xy_plane_from_grid():
    optimization_on_circle.tfprint_tensor_shapes_and_types_for_debug = True
    optimization_on_circle.tfprint_tensor_shapes_and_types_for_function_generate_training_points_in_circle_in_xy_plane_from_grid = True
    optimization_on_circle.tfprint_tensor_values_for_debug = True
    optimization_on_circle.tfprint_tensor_values_for_function_generate_training_points_in_circle_in_xy_plane_from_grid = True

    circle_center_in_xy = [0.0, 0.0]
    circle_radius = 1.0
    t_start = 0.0
    t_stop = 1.0
    num_t_training_points = 2
    num_of_grid_x_training_points = 5
    num_of_grid_y_training_points = 5
    num_xy_training_points_on_boundary = 4
    
    eq_points_tuple, boundary_points_tuple, initial_points_tuple, X_grid, Y_grid, T_grid = \
        generate_training_points_in_circle_in_xy_plane_from_grid(circle_center_in_xy,
                                                                 circle_radius,
                                                                 t_start,
                                                                 t_stop,
                                                                 num_t_training_points,
                                                                 num_of_grid_x_training_points,
                                                                 num_of_grid_y_training_points,
                                                                 num_xy_training_points_on_boundary)

    t_eq, x_eq, y_eq = eq_points_tuple
    t_b, x_b, y_b = boundary_points_tuple
    t_ini, x_ini, y_ini = initial_points_tuple

    # print("t_eq:  ", t_eq)
    # print("x_eq:  ", x_eq)
    # print("y_eq:  ", y_eq)
    # print("t_b:   ", t_b)
    # print("x_b:   ", x_b)
    # print("y_b:   ", y_b)
    # print("t_ini: ", t_ini)
    # print("x_ini: ", x_ini)
    # print("y_ini: ", y_ini)
    # print("X_grid:", X_grid)
    # print("Y_grid:", Y_grid)
    # print("T_grid:", T_grid)


def test_shuffle_points():
    optimization_on_circle.tfprint_tensor_shapes_and_types_for_debug = True
    optimization_on_circle.tfprint_tensor_shapes_and_types_for_function_generate_training_points_in_circle_in_xy_plane_from_grid = False
    optimization_on_circle.tfprint_tensor_shapes_and_types_for_function_shuffle_points = False # Printing must be off because of jit_compile.
    optimization_on_circle.tfprint_tensor_values_for_debug = True
    optimization_on_circle.tfprint_tensor_values_for_function_generate_training_points_in_circle_in_xy_plane_from_grid = False
    optimization_on_circle.tfprint_tensor_values_for_function_shuffle_points = False # Printing must be off because of jit_compile.

    circle_center_in_xy = [0.0, 0.0]
    circle_radius = 1.0
    t_start = 0.0
    t_stop = 1.0
    num_t_training_points = 2
    num_of_grid_x_training_points = 5
    num_of_grid_y_training_points = 5
    num_xy_training_points_on_boundary = 4
    
    eq_points_tuple, boundary_points_tuple, initial_points_tuple, X_grid, Y_grid, T_grid = \
        generate_training_points_in_circle_in_xy_plane_from_grid(circle_center_in_xy,
                                                                 circle_radius,
                                                                 t_start,
                                                                 t_stop,
                                                                 num_t_training_points,
                                                                 num_of_grid_x_training_points,
                                                                 num_of_grid_y_training_points,
                                                                 num_xy_training_points_on_boundary)

    eq_points_tuple_sorted = tf.sort(eq_points_tuple)
    boundary_points_tuple_sorted = tf.sort(boundary_points_tuple)
    initial_points_tuple_sorted = tf.sort(initial_points_tuple)

    shuffled_eq_points_tuple, shuffled_boundary_points_tuple, shuffled_initial_points_tuple = shuffle_points(eq_points_tuple,
                                                                                                             boundary_points_tuple,
                                                                                                             initial_points_tuple,)

    shuffled_eq_points_tuple_sorted = tf.sort(shuffled_eq_points_tuple)
    shuffled_boundary_points_tuple_sorted = tf.sort(shuffled_boundary_points_tuple)
    shuffled_initial_points_tuple_sorted = tf.sort(shuffled_initial_points_tuple)

    with tf.control_dependencies([eq_points_tuple_sorted,
                                  boundary_points_tuple_sorted,
                                  initial_points_tuple_sorted,
                                  shuffled_eq_points_tuple,
                                  shuffled_boundary_points_tuple,
                                  shuffled_initial_points_tuple]):
        tf.debugging.assert_equal(eq_points_tuple_sorted, shuffled_eq_points_tuple_sorted)
        tf.debugging.assert_equal(boundary_points_tuple_sorted, shuffled_boundary_points_tuple_sorted)
        tf.debugging.assert_equal(initial_points_tuple_sorted, shuffled_initial_points_tuple_sorted)

    print("test_shuffle_points: OK")
    return True


def test_generate_training_points_slices():
    optimization_on_circle.tfprint_tensor_shapes_and_types_for_debug = True
    optimization_on_circle.tfprint_tensor_shapes_and_types_for_function_generate_training_points_in_circle_in_xy_plane_from_grid = True
    optimization_on_circle.tfprint_tensor_shapes_and_types_for_function__generate_training_points_slice = False
    optimization_on_circle.tfprint_tensor_values_for_debug = True
    optimization_on_circle.tfprint_tensor_values_for_function_generate_training_points_in_circle_in_xy_plane_from_grid = True
    optimization_on_circle.tfprint_tensor_values_for_function__generate_training_points_slice = False

    circle_center_in_xy = [0.0, 0.0]
    circle_radius = 1.0
    t_start = 0.0
    t_stop = 1.0
    num_t_training_points = 2
    num_of_grid_x_training_points = 5
    num_of_grid_y_training_points = 5
    num_xy_training_points_on_boundary = 4
    
    eq_points_tuple, boundary_points_tuple, initial_points_tuple, X_grid, Y_grid, T_grid = \
        generate_training_points_in_circle_in_xy_plane_from_grid(circle_center_in_xy,
                                                                 circle_radius,
                                                                 t_start,
                                                                 t_stop,
                                                                 num_t_training_points,
                                                                 num_of_grid_x_training_points,
                                                                 num_of_grid_y_training_points,
                                                                 num_xy_training_points_on_boundary)

    tf.print()

    write_values_of_tensors_from_txy_tuple(eq_points_tuple, "_eq")

    write_values_of_tensors_from_txy_tuple(boundary_points_tuple, "_b")

    write_values_of_tensors_from_txy_tuple(initial_points_tuple, "_ini")

    num_of_batches = 2
    for batch_index in range(num_of_batches):
        eq_points_tuple_slice, boundary_points_tuple_slice, initial_points_tuple_slice,  = generate_training_points_slices(eq_points_tuple,
                                                                                                                           boundary_points_tuple,
                                                                                                                           initial_points_tuple,
                                                                                                                           tf.constant(batch_index),
                                                                                                                           num_of_batches,)

        tf.print(f"batch index {batch_index+1}:")
        write_values_of_tensors_from_txy_tuple(eq_points_tuple_slice, "_eq")
        write_values_of_tensors_from_txy_tuple(boundary_points_tuple_slice, "_b")
        write_values_of_tensors_from_txy_tuple(initial_points_tuple_slice, "_ini")
        tf.print()


def test__calculate_approximation_of_integral_with_final_time_fit___old_implementation_without_Zernike_polynomials():
    final_time = 2.0
    t = tf.linspace(0.0, final_time, 2)
    x = tf.linspace(0.0, 4.0, 3)
    y = tf.linspace(0.0, 4.0, 3)
    xy_grid_cell_area = 4.0
    T, X, Y = tf.meshgrid(t, x, y, indexing="ij")
    circle_center_in_xy = [3.0, 3.0]
    circle_radius = 100.0

    print(f"T: {T}")
    print(f"X: {X}")
    print(f"Y: {Y}")


    def test_calculation(test_function, test_nn, correct_integral_approximation_value):
        calculated_integral_approximation = _calculate_approximation_of_integral_with_final_time_fit(test_nn, test_function, X, Y, T, circle_center_in_xy, circle_radius)
        try:
            tf.debugging.assert_equal(calculated_integral_approximation, correct_integral_approximation_value)
        except tf.errors.InvalidArgumentError as e:
            print("Status: FAIL")
            print(f"Calculated integral approximation: {calculated_integral_approximation}")
            print(f"Correct integral approximation:    {correct_integral_approximation_value}")
        else:
            print("Status: PASS")


    print()
    print("Testing _calculate_approximation_of_integral_with_final_time_fit:")


    print("Fitted function: f(T,x,y) = x")
    print("Neural network:  u(t,x,y) = x")
    test_function = lambda x, y: x
    test_nn = get_nn_that_returns_x()
    correct_integral_approximation_value = 0.0
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("Fitted function: f(T,x,y) = x")
    print("Neural network:  u(t,x,y) = y")
    test_function = lambda x, y: x
    test_nn = get_nn_that_returns_y()
    correct_integral_approximation_value = xy_grid_cell_area * float(0 + 2**2 + 2**2 + 0)
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("Fitted function: f(T,x,y) = x")
    print("Neural network:  u(t,x,y) = T")
    test_function = lambda x, y: x
    test_nn = get_nn_that_returns_constant(final_time)
    correct_integral_approximation_value = xy_grid_cell_area * float(2*(1 - final_time)**2 + 2*(3 - final_time)**2)
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("Fitted function: f(T,x,y) = x")
    print("Neural network:  u(t,x,y) = 1")
    test_function = lambda x, y: x
    test_nn = get_nn_that_returns_constant(1.0)
    correct_integral_approximation_value = xy_grid_cell_area * (2.0**2 + 2.0**2)
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("Fitted function: f(T,x,y) = y")
    print("Neural network:  u(t,x,y) = x")
    test_function = lambda x, y: y
    test_nn = get_nn_that_returns_x()
    correct_integral_approximation_value = xy_grid_cell_area * float(0 + (1-3)**2 + (3-1)**2 + 0)
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("Fitted function: f(T,x,y) = y")
    print("Neural network:  u(t,x,y) = y")
    test_function = lambda x, y: y
    test_nn = get_nn_that_returns_y()
    correct_integral_approximation_value = xy_grid_cell_area * float(0)
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("Fitted function: f(T,x,y) = y")
    print("Neural network:  u(t,x,y) = T")
    test_function = lambda x, y: y
    test_nn = get_nn_that_returns_constant(final_time)
    correct_integral_approximation_value = xy_grid_cell_area * float(2*(final_time - 1)**2 + 2*(final_time - 3)**2)
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("Fitted function: f(T,x,y) = y")
    print("Neural network:  u(t,x,y) = 1")
    test_function = lambda x, y: y
    test_nn = get_nn_that_returns_constant(1.0)
    correct_integral_approximation_value = xy_grid_cell_area * float(2*0 + 2*(3-1)**2)
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("Fitted function: f(T,x,y) = T")
    print("Neural network:  u(t,x,y) = x")
    test_function = lambda x, y: final_time
    test_nn = get_nn_that_returns_x()
    correct_integral_approximation_value = xy_grid_cell_area * float(2*(1 - final_time)**2 + 2*(3 - final_time)**2)
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("Fitted function: f(T,x,y) = T")
    print("Neural network:  u(t,x,y) = y")
    test_function = lambda x, y: final_time
    test_nn = get_nn_that_returns_y()
    correct_integral_approximation_value = xy_grid_cell_area * float(2*(1 - final_time)**2 + 2*(3 - final_time)**2)
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("Fitted function: f(T,x,y) = T")
    print("Neural network:  u(t,x,y) = T")
    test_function = lambda x, y: final_time
    test_nn = get_nn_that_returns_constant(final_time)
    correct_integral_approximation_value = xy_grid_cell_area * float(0)
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("Fitted function: f(T,x,y) = T")
    print("Neural network:  u(t,x,y) = 1")
    test_function = lambda x, y: final_time
    test_nn = get_nn_that_returns_constant(1.0)
    correct_integral_approximation_value = xy_grid_cell_area * float(4*(1 - final_time)**2)
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


def test__calculate_approximation_of_integral_with_final_time_fit():
    final_time = 2.0
    t = tf.linspace(0.0, final_time, 2)
    x = tf.linspace(1.5, 4.5, 4)
    y = tf.linspace(-5.0, -3.0, 3)
    xy_grid_cell_area = 1.0
    T, X, Y = tf.meshgrid(t, x, y, indexing="ij")
    circle_center_in_xy = [3.0, -4.0]
    circle_radius = 2.0

    print(f"T: {T}")
    print(f"X: {X}")
    print(f"Y: {Y}")


    def test_calculation(test_function, test_nn, correct_integral_approximation_value):
        calculated_integral_approximation = _calculate_approximation_of_integral_with_final_time_fit(test_nn, test_function, X, Y, T, circle_center_in_xy, circle_radius)
        try:
            tf.debugging.assert_equal(calculated_integral_approximation, correct_integral_approximation_value)
        except tf.errors.InvalidArgumentError as e:
            print("Status: FAIL")
            print(f"Calculated integral approximation: {calculated_integral_approximation}")
            print(f"Correct integral approximation:    {correct_integral_approximation_value}")
        else:
            print("Status: PASS")


    print()
    print("Testing _calculate_approximation_of_integral_with_final_time_fit:")


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


def test_Output_dense_layer_with_activation_function():
    txy = tf.constant([[2, 3, 4]], dtype=tf.float32)
    test_nn = get_nn_with_given_activation_function(lambda t, x, y: x**1)

    with tf.GradientTape(persistent=True) as tape:
        t = txy[:, 0]
        x = txy[:, 1]
        y = txy[:, 2]
        tape.watch([t, x, y])
        txy = tf.stack([t, x, y], -1)
        u = test_nn(txy)
        u_x = tape.gradient(u, x)
    u_xx = tape.gradient(u_x, x)
    print(u_x)
    print(u_xx)


def test__calculate_equation_residual():
    tf.config.run_functions_eagerly(True)

    t = tf.constant([2.0], dtype=tf.float32)
    x = tf.constant([-1.0], dtype=tf.float32)
    y = tf.constant([3.0], dtype=tf.float32)
    txy_tuple = (t, x, y)


    def test_calculation(test_function, test_nn, correct_residual_value):
        calculated_residual, u_pred = _calculate_equation_residual(txy_tuple, test_nn, test_function, heat_coef=1.0)
        try:
            tf.debugging.assert_equal(calculated_residual, correct_residual_value)
        except tf.errors.InvalidArgumentError as e:
            print("Status: FAIL")
            print(f"Calculated residual approximation: {calculated_residual}")
            print(f"Correct residual approximation:    {correct_residual_value}")
        else:
            print("Status: PASS")


    print()
    print("RHS function: f(t,x,y) = 0x + 1")
    print("Neural network:  u(t,x,y) = x^0")
    test_function = lambda t, x, y, u, heat_coef: 0*x + 1
    test_nn = get_nn_with_given_activation_function(lambda t, x, y: x**0)
    correct_integral_approximation_value = float(-1)
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("RHS function: f(t,x,y) = 0x + 1")
    print("Neural network:  u(t,x,y) = t")
    test_function = lambda t, x, y, u, heat_coef: 0*x + 1
    test_nn = get_nn_with_given_activation_function(lambda t, x, y: t**1)
    correct_integral_approximation_value = float(0)
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("RHS function: f(t,x,y) = 0x + 1")
    print("Neural network:  u(t,x,y) = x")
    test_function = lambda t, x, y, u, heat_coef: 0*x + 1
    test_nn = get_nn_with_given_activation_function(lambda t, x, y: x**1)
    # CAREFULL: Just writing x will result in second derivative of u w.r. to x (u_xx) to be None,
    #   since when output is just x, then u_x will be just constant and tensorflow might not connect it to x,
    #   like it will not do gradient of constant as 0, but as None.
    #   So use x**1 to make tensorflow track it as operation.
    correct_integral_approximation_value = float(-1)
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("RHS function: f(t,x,y) = 0x + 1")
    print("Neural network:  u(t,x,y) = x^2")
    test_function = lambda t, x, y, u, heat_coef: 0*x + 1
    test_nn = get_nn_with_given_activation_function(lambda t, x, y: x**2)
    correct_integral_approximation_value = float(-3)
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("RHS function: f(t,x,y) = 0x + 1")
    print("Neural network:  u(t,x,y) = 2t + 3x^2")
    test_function = lambda t, x, y, u, heat_coef: 0*x + 1
    test_nn = get_nn_with_given_activation_function(lambda t, x, y: 2*t + 3*x**2)
    correct_integral_approximation_value = float(-5)
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("RHS function: f(t,x,y) = 0x")
    print("Neural network:  u(t,x,y) = t^2 / 2 + x^3 / 6")
    test_function = lambda t, x, y, u, heat_coef: 0*x
    test_nn = get_nn_with_given_activation_function(lambda t, x, y: t**2 / 2 + x**3 / 6)
    correct_integral_approximation_value = float(3)
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("RHS function: f(t,x,y) = 0x + 5")
    print("Neural network:  u(t,x,y) = t^2 / 2 + x^3 / 6")
    test_function = lambda t, x, y, u, heat_coef: 0*x + 5
    test_nn = get_nn_with_given_activation_function(lambda t, x, y: t**2 / 2 + x**3 / 6)
    correct_integral_approximation_value = float(-2)
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("RHS function: f(t,x,y) = 0x - 8")
    print("Neural network:  u(t,x,y) = t^2 / 2 + x^3 / 6")
    test_function = lambda t, x, y, u, heat_coef: 0*x - 8
    test_nn = get_nn_with_given_activation_function(lambda t, x, y: t**2 / 2 + x**3 / 6)
    correct_integral_approximation_value = float(11)
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("RHS function: f(t,x,y) = 0x + 1")
    print("Neural network:  u(t,x,y) = t^2 / 2 + y^3 / 6")
    test_function = lambda t, x, y, u, heat_coef: 0*x + 1
    test_nn = get_nn_with_given_activation_function(lambda t, x, y: t**2 / 2 + y**3 / 6)
    correct_integral_approximation_value = float(-2)
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("RHS function: f(t,x,y) = 0x - 8")
    print("Neural network:  u(t,x,y) = 5 t^4 - 10 x^5 + 6 y^7")
    test_function = lambda t, x, y, u, heat_coef: 0*x - 8
    test_nn = get_nn_with_given_activation_function(lambda t, x, y: 5 * t**4 - 10 * x**5 + 6 * y**7)
    correct_integral_approximation_value = float(5 * 4 * 2**3 + 10 * 5 * 4 * (-1)**3 - 6 * 7 * 6 * 3**5 - (-8))
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("RHS function: f(t,x,y) = 0x - 8")
    print("Neural network:  u(t,x,y) = 5 t^3 - 10 x^4 + 6 y^4")
    test_function = lambda t, x, y, u, heat_coef: 0*x - 8
    test_nn = get_nn_with_given_activation_function(lambda t, x, y: 5 * t**3 - 10 * x**4 + 6 * y**4)
    correct_integral_approximation_value = float(5 * 3 * 2**2 + 10 * 4 * 3 * (-1)**2 - 6 * 4 * 3 * 3**2 - (-8))
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("RHS function: f(t,x,y) = 1")
    print("Neural network:  u(t,x,y) = x")
    test_function = lambda t, x, y, u, heat_coef: 1
    test_nn = get_nn_with_given_activation_function(lambda t, x, y: x**1)
    correct_integral_approximation_value = float(-1)
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("RHS function: f(t,x,y) = x + y + u")
    print("Neural network:  u(t,x,y) = t^2 / 2 - x^3 / 6 - y^3 / 6")
    test_function = lambda t, x, y, u, heat_coef: x + y + u
    test_nn = get_nn_with_given_activation_function(lambda t, x, y: t**2 / 2 - x**3 / 6 - y**3 / 6)
    correct_integral_approximation_value = float((2-1+3) + 1 - 3 - (2**2 / 2 - (-1)**3 / 6 - 3**3 / 6))
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("RHS function: f(t,x,y) = t + u")
    print("Neural network:  u(t,x,y) = t^2 / 2 - x^3 / 6 - y^3 / 6")
    test_function = lambda t, x, y, u, heat_coef: t + u
    test_nn = get_nn_with_given_activation_function(lambda t, x, y: t**2 / 2 - x**3 / 6 - y**3 / 6)
    correct_integral_approximation_value = float((2-1+3) - 2 - (2**2 / 2 - (-1)**3 / 6 - 3**3 / 6))
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


def test__calculate_initial_condition_residual():
    tf.config.run_functions_eagerly(True)

    t = tf.constant([0.0], dtype=tf.float32)
    x = tf.constant([-1.0], dtype=tf.float32)
    y = tf.constant([3.0], dtype=tf.float32)
    txy_tuple = (t, x, y)


    def test_calculation(test_function, test_nn, correct_residual_value):
        u_pred = test_nn(tf.stack([t, x, y], 1))
        calculated_residual = _calculate_initial_condition_residual(txy_tuple, test_nn, u_pred, test_function, heat_coef=1.0)
        try:
            tf.debugging.assert_equal(calculated_residual, correct_residual_value)
        except tf.errors.InvalidArgumentError as e:
            print("Status: FAIL")
            print(f"Calculated residual approximation: {calculated_residual}")
            print(f"Correct residual approximation:    {correct_residual_value}")
        else:
            print("Status: PASS")


    print()
    print("Initial condition function: f(x,y,u) = 0x + 1")
    print("Neural network:             u(t,x,y) = x^0")
    test_function = lambda x, y, u, heat_coef: 0*x + 1
    test_nn = get_nn_with_given_activation_function(lambda t, x, y: x**0)
    correct_integral_approximation_value = float(0)
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("Initial condition function: f(x,y,u) = 0x + 1")
    print("Neural network:             u(t,x,y) = t")
    test_function = lambda x, y, u, heat_coef: 0*x + 1
    test_nn = get_nn_with_given_activation_function(lambda t, x, y: t)
    correct_integral_approximation_value = float(-1)
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("Initial condition function: f(x,y,u) = 0x + 1")
    print("Neural network:             u(t,x,y) = x")
    test_function = lambda x, y, u, heat_coef: 0*x + 1
    test_nn = get_nn_with_given_activation_function(lambda t, x, y: x)
    correct_integral_approximation_value = float(-2)
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("Initial condition function: f(x,y,u) = 0x + 1")
    print("Neural network:             u(t,x,y) = x^2")
    test_function = lambda x, y, u, heat_coef: 0*x + 1
    test_nn = get_nn_with_given_activation_function(lambda t, x, y: x**2)
    correct_integral_approximation_value = float(0)
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("Initial condition function: f(x,y,u) = 0x + 1")
    print("Neural network:             u(t,x,y) = 2t + 3x^2")
    test_function = lambda x, y, u, heat_coef: 0*x + 1
    test_nn = get_nn_with_given_activation_function(lambda t, x, y: 2*t + 3*x**2)
    correct_integral_approximation_value = float(2)
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("Initial condition function: f(x,y,u) = 0x")
    print("Neural network:             u(t,x,y) = t^2 / 2 + x^3 / 6")
    test_function = lambda x, y, u, heat_coef: 0*x
    test_nn = get_nn_with_given_activation_function(lambda t, x, y: t**2 / 2 + x**3 / 6)
    correct_integral_approximation_value = float(-1/6 - 1)
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("Initial condition function: f(x,y,u) = 0x + 5")
    print("Neural network:             u(t,x,y) = t^2 / 2 + x^3 / 6")
    test_function = lambda x, y, u, heat_coef: 0*x + 5
    test_nn = get_nn_with_given_activation_function(lambda t, x, y: t**2 / 2 + x**3 / 6)
    correct_integral_approximation_value = float(-1/6 - 5)
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("Initial condition function: f(x,y,u) = 0x - 8")
    print("Neural network:             u(t,x,y) = t^2 / 2 + x^3 / 6")
    test_function = lambda x, y, u, heat_coef: 0*x - 8
    test_nn = get_nn_with_given_activation_function(lambda t, x, y: t**2 / 2 + x**3 / 6)
    correct_integral_approximation_value = float(-1/6 + 8)
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("Initial condition function: f(x,y,u) = 0x + 4")
    print("Neural network:             u(t,x,y) = t^2 / 2 + y^3 / 6")
    test_function = lambda x, y, u, heat_coef: 0*x + 4
    test_nn = get_nn_with_given_activation_function(lambda t, x, y: t**2 / 2 + y**3 / 6)
    correct_integral_approximation_value = float(27/6 - 4)
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("Initial condition function: f(x,y,u) = 0x - 8")
    print("Neural network:             u(t,x,y) = 5 t^4 - 10 x^5 + 6 y^7")
    test_function = lambda x, y, u, heat_coef: 0*x - 8
    test_nn = get_nn_with_given_activation_function(lambda t, x, y: 5 * t**4 - 10 * x**5 + 6 * y**7)
    correct_integral_approximation_value = float(10 + 6 * 3**7 + 8)
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("Initial condition function: f(x,y,u) = 0x - 8")
    print("Neural network:             u(t,x,y) = 5 t^3 - 10 x^4 + 6 y^4")
    test_function = lambda x, y, u, heat_coef: 0*x - 8
    test_nn = get_nn_with_given_activation_function(lambda t, x, y: 5 * t**3 - 10 * x**4 + 6 * y**4)
    correct_integral_approximation_value = float(-10 + 6 * 3**4 + 8)
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("Initial condition function: f(x,y,u) = 1")
    print("Neural network:             u(t,x,y) = x")
    test_function = lambda x, y, u, heat_coef: 1
    test_nn = get_nn_with_given_activation_function(lambda t, x, y: x)
    correct_integral_approximation_value = float(-2)
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("Initial condition function: f(x,y,u) = x + y")
    print("Neural network:             u(t,x,y) = x")
    test_function = lambda x, y, u, heat_coef: x + y
    test_nn = get_nn_with_given_activation_function(lambda t, x, y: x)
    correct_integral_approximation_value = float(-1 - (3-1))
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("Initial condition function: f(x,y,u) = x + y + u")
    print("Neural network:             u(t,x,y) = 2x")
    test_function = lambda x, y, u, heat_coef: x + y + u
    test_nn = get_nn_with_given_activation_function(lambda t, x, y: 2*x)
    correct_integral_approximation_value = float(-1 - (2-1))
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


    print()
    print("Initial condition function: f(x,y,u) = x + y + u")
    print("Neural network:             u(t,x,y) = 3y - 50x")
    test_function = lambda x, y, u, heat_coef: x + y + u
    test_nn = get_nn_with_given_activation_function(lambda t, x, y: 3*y - 50*x)
    correct_integral_approximation_value = float(-1 - (2-1))
    test_calculation(test_function, test_nn, correct_integral_approximation_value)


def test__calculate_boundary_residual():
    # tf.config.run_functions_eagerly(True)

    circle_center_in_xy = [2.0, 3.0]
    t = tf.constant([2.0, -3.0, 4.0, -1.0], dtype=tf.float32)
    x = tf.constant([6.0, 2.0, -2.0, 2.0], dtype=tf.float32)
    y = tf.constant([3.0, 7.0, 3.0, -1.0], dtype=tf.float32)
    txy_tuple = (t, x, y)

    def test_calculation(test_nn, test_qnn, circle_center_in_xy, correct_residual_value):
        calculated_residual = _calculate_boundary_residual(txy_tuple, test_nn, test_qnn, circle_center_in_xy)
        try:
            tf.debugging.assert_equal(calculated_residual, correct_residual_value)
        except tf.errors.InvalidArgumentError as e:
            print("Status: FAIL")
            print(f"Calculated residual approximation: {calculated_residual}")
            print(f"Correct residual approximation:    {correct_residual_value}")
        else:
            print("Status: PASS")


    def normalize(tensor):
        new_tensor_as_list = []
        number_of_points_in_tensor = tf.size(tensor[:, 0])
        for i in range(number_of_points_in_tensor):
            tensor_i_x = tensor[i, 0].numpy()
            tensor_i_y = tensor[i, 1].numpy()
            tensor_i_norm = pow(tensor_i_x**2 + tensor_i_y**2, 0.5)

            if tensor_i_norm != 0.0:
                new_tensor_as_list.append([tensor_i_x / tensor_i_norm, tensor_i_y / tensor_i_norm])
            else:
                new_tensor_as_list.append([0.0, 0.0])

        print(f"normalized values: {new_tensor_as_list}")
        return tf.constant(new_tensor_as_list, dtype=tf.float32)


    boundary_unit_normal = normalize(tf.stack([x - circle_center_in_xy[0], y - circle_center_in_xy[1]], 1))
    boundary_unit_normal_x = boundary_unit_normal[:, 0]
    boundary_unit_normal_y = boundary_unit_normal[:, 1]
    print(f"boundary unit normal: {boundary_unit_normal}")


    def perform_tests_for_given_circle_center(circle_center_in_xy):
        print()
        print("q_nn(t,x,y) = t")
        print("u(t,x,y)    = t")
        test_qnn = get_nn_with_given_activation_function(lambda t, x, y: t)
        test_nn = get_nn_with_given_activation_function(lambda t, x, y: t)
        correct_residual_value = -t
        test_calculation(test_nn, test_qnn, circle_center_in_xy, correct_residual_value)

        print()
        print("q_nn(t,x,y) = t")
        print("u(t,x,y)    = x")
        test_qnn = get_nn_with_given_activation_function(lambda t, x, y: t)
        test_nn = get_nn_with_given_activation_function(lambda t, x, y: x)
        correct_residual_value = boundary_unit_normal_x - t
        test_calculation(test_nn, test_qnn, circle_center_in_xy, correct_residual_value)

        print()
        print("q_nn(t,x,y) = t")
        print("u(t,x,y)    = x^2 / 2")
        test_qnn = get_nn_with_given_activation_function(lambda t, x, y: t)
        test_nn = get_nn_with_given_activation_function(lambda t, x, y: x**2 / 2)
        correct_residual_value = boundary_unit_normal_x * x - t
        test_calculation(test_nn, test_qnn, circle_center_in_xy, correct_residual_value)

        print()
        print("q_nn(t,x,y) = t")
        print("u(t,x,y)    = y")
        test_qnn = get_nn_with_given_activation_function(lambda t, x, y: t)
        test_nn = get_nn_with_given_activation_function(lambda t, x, y: y)
        correct_residual_value = boundary_unit_normal_y - t
        test_calculation(test_nn, test_qnn, circle_center_in_xy, correct_residual_value)

        print()
        print("q_nn(t,x,y) = t")
        print("u(t,x,y)    = y^2 / 2")
        test_qnn = get_nn_with_given_activation_function(lambda t, x, y: t)
        test_nn = get_nn_with_given_activation_function(lambda t, x, y: y**2 / 2)
        correct_residual_value = boundary_unit_normal_y * y - t
        test_calculation(test_nn, test_qnn, circle_center_in_xy, correct_residual_value)

        print()
        print("q_nn(t,x,y) = t")
        print("u(t,x,y)    = x^2 / 2 + y^2 / 2")
        test_qnn = get_nn_with_given_activation_function(lambda t, x, y: t)
        test_nn = get_nn_with_given_activation_function(lambda t, x, y: x**2 / 2 + y**2 / 2)
        correct_residual_value = \
            boundary_unit_normal_x * x + boundary_unit_normal_y * y - t
        test_calculation(test_nn, test_qnn, circle_center_in_xy, correct_residual_value)

        print()
        print("q_nn(t,x,y) = x")
        print("u(t,x,y)    = t")
        test_qnn = get_nn_with_given_activation_function(lambda t, x, y: x)
        test_nn = get_nn_with_given_activation_function(lambda t, x, y: t)
        correct_residual_value = -x
        test_calculation(test_nn, test_qnn, circle_center_in_xy, correct_residual_value)

        print()
        print("q_nn(t,x,y) = x")
        print("u(t,x,y)    = x")
        test_qnn = get_nn_with_given_activation_function(lambda t, x, y: x)
        test_nn = get_nn_with_given_activation_function(lambda t, x, y: x)
        correct_residual_value = boundary_unit_normal_x - x
        test_calculation(test_nn, test_qnn, circle_center_in_xy, correct_residual_value)

        print()
        print("q_nn(t,x,y) = x")
        print("u(t,x,y)    = x^2 / 2")
        test_qnn = get_nn_with_given_activation_function(lambda t, x, y: x)
        test_nn = get_nn_with_given_activation_function(lambda t, x, y: x**2 / 2)
        correct_residual_value = boundary_unit_normal_x * x - x
        test_calculation(test_nn, test_qnn, circle_center_in_xy, correct_residual_value)

        print()
        print("q_nn(t,x,y) = x")
        print("u(t,x,y)    = y")
        test_qnn = get_nn_with_given_activation_function(lambda t, x, y: x)
        test_nn = get_nn_with_given_activation_function(lambda t, x, y: y)
        correct_residual_value = boundary_unit_normal_y - x
        test_calculation(test_nn, test_qnn, circle_center_in_xy, correct_residual_value)

        print()
        print("q_nn(t,x,y) = x")
        print("u(t,x,y)    = y^2 / 2")
        test_qnn = get_nn_with_given_activation_function(lambda t, x, y: x)
        test_nn = get_nn_with_given_activation_function(lambda t, x, y: y**2 / 2)
        correct_residual_value = boundary_unit_normal_y * y - x
        test_calculation(test_nn, test_qnn, circle_center_in_xy, correct_residual_value)

        print()
        print("q_nn(t,x,y) = x")
        print("u(t,x,y)    = x^2 / 2 + y^2 / 2")
        test_qnn = get_nn_with_given_activation_function(lambda t, x, y: x)
        test_nn = get_nn_with_given_activation_function(lambda t, x, y: x**2 / 2 + y**2 / 2)
        correct_residual_value = \
            boundary_unit_normal_x * x + boundary_unit_normal_y * y - x
        test_calculation(test_nn, test_qnn, circle_center_in_xy, correct_residual_value)

        print()
        print("q_nn(t,x,y) = y")
        print("u(t,x,y)    = t")
        test_qnn = get_nn_with_given_activation_function(lambda t, x, y: y)
        test_nn = get_nn_with_given_activation_function(lambda t, x, y: t)
        correct_residual_value = -y
        test_calculation(test_nn, test_qnn, circle_center_in_xy, correct_residual_value)

        print()
        print("q_nn(t,x,y) = y")
        print("u(t,x,y)    = x")
        test_qnn = get_nn_with_given_activation_function(lambda t, x, y: y)
        test_nn = get_nn_with_given_activation_function(lambda t, x, y: x)
        correct_residual_value = boundary_unit_normal_x - y
        test_calculation(test_nn, test_qnn, circle_center_in_xy, correct_residual_value)

        print()
        print("q_nn(t,x,y) = y")
        print("u(t,x,y)    = x^2 / 2")
        test_qnn = get_nn_with_given_activation_function(lambda t, x, y: y)
        test_nn = get_nn_with_given_activation_function(lambda t, x, y: x**2 / 2)
        correct_residual_value = boundary_unit_normal_x * x - y
        test_calculation(test_nn, test_qnn, circle_center_in_xy, correct_residual_value)

        print()
        print("q_nn(t,x,y) = y")
        print("u(t,x,y)    = y")
        test_qnn = get_nn_with_given_activation_function(lambda t, x, y: y)
        test_nn = get_nn_with_given_activation_function(lambda t, x, y: y)
        correct_residual_value = boundary_unit_normal_y - y
        test_calculation(test_nn, test_qnn, circle_center_in_xy, correct_residual_value)

        print()
        print("q_nn(t,x,y) = y")
        print("u(t,x,y)    = y^2 / 2")
        test_qnn = get_nn_with_given_activation_function(lambda t, x, y: y)
        test_nn = get_nn_with_given_activation_function(lambda t, x, y: y**2 / 2)
        correct_residual_value = boundary_unit_normal_y * y - y
        test_calculation(test_nn, test_qnn, circle_center_in_xy, correct_residual_value)

        print()
        print("q_nn(t,x,y) = y")
        print("u(t,x,y)    = x^2 / 2 + y^2 / 2")
        test_qnn = get_nn_with_given_activation_function(lambda t, x, y: y)
        test_nn = get_nn_with_given_activation_function(lambda t, x, y: x**2 / 2 + y**2 / 2)
        correct_residual_value = \
            boundary_unit_normal_x * x + boundary_unit_normal_y * y - y
        test_calculation(test_nn, test_qnn, circle_center_in_xy, correct_residual_value)

        print()
        print("q_nn(t,x,y) = 5 t^2 + 3 x^3 - 2 y^4")
        print("u(t,x,y)    = t^3 - 5 x^5 + 5 y^2")
        test_qnn = get_nn_with_given_activation_function(lambda t, x, y: 5 * t**2 + 3 * x**3 - 2 * y**4)
        test_nn = get_nn_with_given_activation_function(lambda t, x, y: t**3 - 5 * x**5 + 5 * y**2)
        correct_residual_value = \
            boundary_unit_normal_x * (-5 * 5 * x**4) + boundary_unit_normal_y * (5 * 2 * y) - (5 * t**2 + 3 * x**3 - 2 * y**4)
        test_calculation(test_nn, test_qnn, circle_center_in_xy, correct_residual_value)

        print()
        print("q_nn(t,x,y) = 5 t^2 + 3 x^3 - 2 y^4")
        print("u(t,x,y)    = t^6 - 5 x^5 + 5 y^2")
        test_qnn = get_nn_with_given_activation_function(lambda t, x, y: 5 * t**2 + 3 * x**3 - 2 * y**4)
        test_nn = get_nn_with_given_activation_function(lambda t, x, y: t**6 - 5 * x**5 + 5 * y**2)
        correct_residual_value = \
            boundary_unit_normal_x * (-5 * 5 * x**4) + boundary_unit_normal_y * (5 * 2 * y) - (5 * t**2 + 3 * x**3 - 2 * y**4)
        test_calculation(test_nn, test_qnn, circle_center_in_xy, correct_residual_value)

        return


    print()
    print("#########################################")
    print(f"circle ceter: {circle_center_in_xy}")
    perform_tests_for_given_circle_center(circle_center_in_xy)

    # print()
    # print("#########################################")
    # circle_center_in_xy = [-1.0, 0.0]
    # print(f"circle ceter: {circle_center_in_xy}")
    # perform_tests_for_given_circle_center(circle_center_in_xy)
