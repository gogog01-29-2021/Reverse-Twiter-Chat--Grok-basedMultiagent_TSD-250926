import tensorflow as tf
import keras

from ..utils import *

# __all__ = [
#     "Sequential_with_loss_based_on_input_and_params",
#     "mean_squared_residual_1D",
#     "Poisson_problem_1D_pointwise",
#     "Poisson_problem_1D_pointwise_tapeless",
#     "Poisson_problem_1D_pointwise_tapeless_old",
#     "Poisson_problem_1D_pointwise_hard_dirichlet_boundary",
#     "Poisson_problem_1D_pointwise_print",
#     "Poisson_problem_1D_pointwise_gpinn",
#     "Poisson_problem_1D_pointwise_gpinn_without_boundary",
#     "Poisson_problem_1D_pointwise_gpinn_hard_boundary",
#     "Poisson_problem_1D_pointwise_dynamic_loss_coeficients",
#     "Poisson_problem_1D_pointwise_gpinn_max_norm",
#     "Poisson_problem_1D_pointwise_gpinn_MSE_and_max_norm",
#     "Poisson_problem_1D_pointwise_random_resampling_in_train_step",
#     ]


class Sequential_with_loss_based_on_input_and_params(keras.Sequential):
    # TODO Přepsat popisek do stylu jako např. tf.math.squared_difference(), kde to mají docela hezky.
    # Např. mají sekci pro argumenty a tak.
    # TODO: update description of loss fn parameters
    """Sequential model that has loss function based on input of the neural network. The loss function is passed to the model when creating it as the loss_function parameter.
    
    The loss function is supposed to have arguments in the form (input, y_true, y_pred) and return a value representing the loss.
    The input argument is the current batch of input data received by the fit() function. Therefore the shape of it is (batch_size, input_shape). For example it could be (64, 100) if we have batch of size 64 and the input shape of one data sample is an array with 100 values, or it could be (64, 100, 1) if the input is a conv1D layer with 1 channel and 100 points the mask/filter passes over.
    The y_true argument holds the true values passed in by the y argument of the fit() method of the model.
    The y_pred argument is the prediction made by the model by calling the model on the current batch of input data.

    The calculation of the loss inside the function should ideally be done using tensorflow operations - so it should be standard operations on tensorflow tensors or operations from tensorflow module, for example tf.math.reduce_mean, etc. This is so that tensorflow can compute the gradient of the loss function.

    EXAMPLE:
    1) mean squared error of y_pred and input
    ```
    import tensorflow as tf

    # The y_true argument still has to be present, even if it isn't used.
    def loss_fn(input, y_true, y_pred):
        loss_value = 0
        #compute the squared error
        squared_error_tensor = tf.math.squared_difference(x=input, y=y_pred)
        # take the mean along all axis
        mean_squared_error = tf.math.reduce_mean(squared_error_tensor, axis=None)
        return mean_squared_error
    ```
    
    """

    def __init__(self, *args, loss_function=None, loss_function_args_dict={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_function = loss_function
        self.loss_function_args_dict = loss_function_args_dict

    def train_step(self, data):
        x, y_true = data

        # weights update:
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True) # forward pass
            loss_value = self.loss_function(input=x, y_true=y_true, y_pred=y_pred, **self.loss_function_args_dict)  # compute loss

        # compute gradients with respect to trainable parameters of the network like weights and such:
        # gradients = tape.gradient(loss_value, self.trainable_variables)
        # update the weights:
        # self.optimizer.update_step(gradient=gradients, variable=self.trainable_variables)

        self.optimizer.minimize(loss_value, self.trainable_variables, tape=tape)
        # tf.keras.optimizers.Adam().

        return {
            "loss": loss_value
        }


class Poisson_problem_1D_pointwise(keras.Sequential):
    """
    Model for Poisson problem with fixed right hand side (RHS). Output shape should be the same as the input
    shape, with the shape being (batch_of_points, 1)

    Parameters
    ----------
    RHS_function : Callable function that takes in batch of points as an input
    with shape (batch_size, 1) and returns the right hand side evaluated in those
    points. The output can be of shape (batch_size, 1) or (batch_size).
    Should be made using tensorflow operations (operations on tensors and operations
    from modules tensorflow, tensorflow.math, tensorflow.linalg,and such), so it can be differentiated.
    
    .
    """
    def __init__(self, RHS_function, left_boundary_point, right_boundary_point, *args, equation_loss_coef=1, boundary_loss_coef=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.RHS_function = RHS_function
        self.equation_loss_coef = equation_loss_coef
        self.boundary_loss_coef = boundary_loss_coef
        self.left_boundary_point = left_boundary_point
        self.right_boundary_point = right_boundary_point

    # @tf.function
    def train_step(self, data):
        x, y_true = data

        input_type = x.dtype
        left_boundary_point = tf.constant([[self.left_boundary_point]], dtype=input_type) # needs shape (1, 1) so it can be fed into a dense layer
        right_boundary_point = tf.constant([[self.right_boundary_point]], dtype=input_type) # needs shape (1, 1) so it can be fed into a dense layer

        with tf.GradientTape() as weights_update_tape:
            with tf.GradientTape(watch_accessed_variables=False, persistent=True) as derivation_tape:
                derivation_tape.watch(x) # shape of x: (batch_size, 1)
                u = self(x, training=True) # shape: (batch_size, 1)
                u_x = derivation_tape.gradient(target=u, sources=x)   # shape: (batch_of_points, 1)

            u_xx = derivation_tape.gradient(target=u_x, sources=x) # shape: (batch_of_points, 1)
            del derivation_tape

            RHS_evaluated = self.RHS_function(x)  # shape: (batch_size, 1)
            equation_loss = self.equation_loss_coef * tf.math.reduce_mean( tf.math.squared_difference(u_xx, RHS_evaluated) )


            u_left_boundary = self(left_boundary_point, training=True) # shape: (1, 1)
            u_right_boundary = self(right_boundary_point, training=True)   # shape: (1, 1)            
            boundary_loss = self.boundary_loss_coef * tf.reduce_sum(u_left_boundary**2 + u_right_boundary**2) / 2

            # loss = boundary_loss + equation_loss
            loss_dict = {
                "equation loss": equation_loss,
                "boundary loss": boundary_loss
            }
        
        # self.optimizer.minimize(loss_dict, self.trainable_variables, tape=weights_update_tape)
        grads = weights_update_tape.gradient(target=loss_dict, sources=self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss_dict


class Poisson_problem_1D_pointwise_functioning_older(keras.Sequential):
    """
    Model for Poisson problem with fixed right hand side (RHS). Output shape should be the same as the input
    shape, with the shape being (batch_of_points, 1)

    Parameters
    ----------
    RHS_function : Callable function that takes in batch of points as an input
    with shape (batch_size, 1) and returns the right hand side evaluated in those
    points. The output can be of shape (batch_size, 1) or (batch_size).
    Should be made using tensorflow operations (operations on tensors and operations
    from modules tensorflow, tensorflow.math, tensorflow.linalg,and such), so it can be differentiated.
    
    .
    """
    def __init__(self, RHS_function, left_boundary_point, right_boundary_point, *args, equation_loss_coef=1, boundary_loss_coef=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.RHS_function = RHS_function
        self.equation_loss_coef = equation_loss_coef
        self.boundary_loss_coef = boundary_loss_coef
        self.left_boundary_point = left_boundary_point
        self.right_boundary_point = right_boundary_point

    # @tf.function
    def train_step(self, data):
        x, y_true = data

        input_type = x.dtype
        left_boundary_point = tf.constant([[self.left_boundary_point]], dtype=input_type) # needs shape (1, 1) so it can be fed into a dense layer
        # left_boundary_point = tf.Variable(initial_value=[[self.left_boundary_point]], trainable=True, dtype=input_type)

        right_boundary_point = tf.constant([[self.right_boundary_point]], dtype=input_type) # needs shape (1, 1) so it can be fed into a dense layer
        # right_boundary_point = tf.Variable(initial_value=[[self.right_boundary_point]], trainable=True, dtype=input_type)

        with tf.GradientTape() as weights_update_tape:
            with tf.GradientTape(watch_accessed_variables=False, persistent=True) as second_der_tape:
                second_der_tape.watch(x)    # shape of x: (batch_size, 1)

                with tf.GradientTape(watch_accessed_variables=False, persistent=True) as first_der_tape:
                    first_der_tape.watch(x) # shape of x: (batch_size, 1)
                    prediction = self(x, training=True)   # shape: (batch_size, 1)

                # CAREFULL: derivatives wouldn't work correctly like this if I had multiple outpust, there would be some
                # modifications needed: taking batch_jacobian for first derivative and extracting the diagonal derivatives.
                # # gradients for multiple outputs would just sum the derivatives of individial outpust for fixed x
                # # (it would just act as gradient does regurarly for multiple targets, see docs: 
                # # https://www.tensorflow.org/guide/autodiff#gradients_of_non-scalar_targets)
                u_x = first_der_tape.gradient(target=prediction, sources=x)   # shape: (batch_of_points, 1)
                del first_der_tape

                # tf.print("u_x:")
                # print_tensor(u_x)

            u_xx = second_der_tape.batch_jacobian(target=u_x, source=x) # shape: (batch_of_points, 1, 1)
            del second_der_tape

            # tf.print("u_xx:")
            # print_tensor(u_xx)

            u_xx = tf.squeeze(u_xx) # shape: (batch_size)
            RHS_evaluated = tf.squeeze( self.RHS_function(x) )  # shape: (batch_size)
            # residual = u_xx - RHS_evaluated
            # equation_loss = self.equation_loss_coef * tf.math.reduce_mean( tf.math.square(residual) )
            equation_loss = self.equation_loss_coef * tf.math.reduce_mean( tf.math.squared_difference(u_xx, RHS_evaluated) )


            prediction_left_boundary = self(left_boundary_point, training=True) # shape: (1, 1)
            prediction_right_boundary = self(right_boundary_point, training=True)   # shape: (1, 1)
            
            loss_left_boundary = tf.math.squared_difference(prediction_left_boundary, tf.zeros_like(prediction_left_boundary))
            loss_right_boundary = tf.math.squared_difference(prediction_right_boundary, tf.zeros_like(prediction_right_boundary))
            boundary_loss = self.boundary_loss_coef * (loss_left_boundary + loss_right_boundary) / 2

            # loss = boundary_loss + equation_loss
            loss_dict = {
                "equation loss": equation_loss,
                "boundary loss": boundary_loss
            }
        
        # self.optimizer.minimize(loss_dict, self.trainable_variables, tape=weights_update_tape)
        grads = weights_update_tape.gradient(target=loss_dict, sources=self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss_dict


class Poisson_problem_1D_pointwise_print(keras.Sequential):
    """
    .
    """
    def __init__(self, RHS_function, left_boundary_point, right_boundary_point, *args, equation_loss_coef=1, boundary_loss_coef=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.RHS_function = RHS_function
        self.equation_loss_coef = equation_loss_coef
        self.boundary_loss_coef = boundary_loss_coef
        self.left_boundary_point = left_boundary_point
        self.right_boundary_point = right_boundary_point
        self.epoch_number = tf.Variable(-1, trainable=False)
            # the starting value is -1 as the train_step function gets called
            # (probably) 2 times for tracing or something before the actually 
            # training starts, so then the epoch numbering will start from 1

        self._previous_x_ref = None
        self._previous_u_x_ref = None

    # @tf.function
    def train_step(self, data):
        self.epoch_number.assign_add(1)
        tf.print("EPOCH", end=" ")
        tf.print(self.epoch_number, end="\n\n")

        x, y_true = data

        print_tensor_ref(x, "x")
        
        if self._previous_x_ref != None:
            x__ref_is_same = x.ref == self._previous_x_ref
            message = f"x ref is same: {x__ref_is_same}"
            tf.print(message)
        self._previous_x_ref = x.ref

        print_shape_and_value(x, "x")
        previous_x_ref = x.ref

        input_type = x.dtype
        left_boundary_point = tf.constant([[self.left_boundary_point]], dtype=input_type) # needs shape (1, 1) so it can be fed into a dense layer
        print_shape_and_value(left_boundary_point, "left_boundary_point")

        right_boundary_point = tf.constant([[self.right_boundary_point]], dtype=input_type) # needs shape (1, 1) so it can be fed into a dense layer
        print_shape_and_value(right_boundary_point, "right_boundary_point")

        with tf.GradientTape() as weights_update_tape:
            with tf.GradientTape(watch_accessed_variables=False, persistent=True) as second_der_tape:
                second_der_tape.watch(x)    # shape of x: (batch_size, 1)
                second_der_tape.watch(left_boundary_point)
                second_der_tape.watch(right_boundary_point)

                with tf.GradientTape(watch_accessed_variables=False, persistent=True) as first_der_tape:
                    first_der_tape.watch(x) # shape of x: (batch_size, 1)
                    first_der_tape.watch(left_boundary_point)
                    first_der_tape.watch(right_boundary_point)
                    prediction = self(x, training=True)   # shape: (batch_size, 1)
                    prediction_left_boundary = self(left_boundary_point, training=True) # shape: (1, 1)
                    prediction_right_boundary = self(right_boundary_point, training=True)   # shape: (1, 1)

                    print_tensor_ref(prediction, "prediction")
                    print_shape_and_value(prediction, "prediction")
                    print_shape_and_value(prediction_left_boundary, "prediction_left_boundary")
                    print_shape_and_value(prediction_right_boundary, "prediction_right_boundary")

                # CAREFULL: derivatives wouldn't work correctly like this if I had multiple outpust, there would be some
                # modifications needed: taking batch_jacobian for first derivative and extracting the diagonal derivatives.
                # # gradients for multiple outputs would just sum the derivatives of individial outpust for fixed x
                # # (it would just act as gradient does regurarly for multiple targets, see docs: 
                # # https://www.tensorflow.org/guide/autodiff#gradients_of_non-scalar_targets)
                u_x = first_der_tape.gradient(target=prediction, sources=x)   # shape: (batch_of_points, 1)
                u_x_left_boundary = first_der_tape.gradient(target=prediction_left_boundary, sources=left_boundary_point)   # shape: (1, 1)
                u_x_right_boundary = first_der_tape.gradient(target=prediction_right_boundary, sources=right_boundary_point)    # shape: (1, 1)
                del first_der_tape

                print_tensor_ref(u_x, "u_x")
                if self._previous_u_x_ref != None:
                    u_x__ref_is_same = u_x.ref == self._previous_u_x_ref
                    message = f"u_x ref is same: {u_x__ref_is_same}"
                    tf.print(message)
                self._previous_u_x_ref = u_x.ref
                
                print_shape_and_value(u_x, "u_x")
                print_shape_and_value(u_x_left_boundary, "u_x_left_boundary")
                print_shape_and_value(u_x_right_boundary, "u_x_right_boundary")

            u_xx = second_der_tape.batch_jacobian(target=u_x, source=x) # shape: (batch_of_points, 1, 1)
            u_xx_left_boundary = second_der_tape.batch_jacobian(target=u_x_left_boundary, source=left_boundary_point) # shape: (1, 1, 1)
            u_xx_right_boundary = second_der_tape.batch_jacobian(target=u_x_right_boundary, source=right_boundary_point) # shape: (1, 1, 1)
            del second_der_tape


            print_shape_and_value(u_xx, "u_xx")
            print_shape_and_value(u_xx_left_boundary, "u_xx_left_boundary")
            print_shape_and_value(u_xx_right_boundary, "u_xx_right_boundary")

            u_xx = tf.squeeze(u_xx) # shape: (batch_size)
            RHS_evaluated = tf.squeeze( self.RHS_function(x) )  # shape: (batch_size)
            
            # tf.print("u_xx after squeeze shape:")
            # tf.print(u_xx.shape)
            # tf.print("RHS_evaluated after squeeze shape:")
            # tf.print(RHS_evaluated.shape)
            residual = tf.expand_dims(u_xx - RHS_evaluated, axis=-1)
            print_shape_and_value(residual, "residual")

            # equation_loss = self.equation_loss_coef * tf.math.reduce_mean( tf.math.square(residual) )
            equation_loss = self.equation_loss_coef * tf.math.reduce_mean( tf.math.squared_difference(u_xx, RHS_evaluated) )
            print_shape_and_value(equation_loss, "equation_loss")

            u_xx_left_boundary = tf.squeeze(u_xx_left_boundary)
            RHS_at_left_boundary = tf.squeeze( self.RHS_function(left_boundary_point) )
            residual_left_boundary = u_xx_left_boundary - RHS_at_left_boundary
            loss_left_boundary = tf.math.squared_difference(u_xx_left_boundary, RHS_at_left_boundary)

            print_shape_and_value(u_xx_left_boundary, "u_xx_left_boundary after squeeze")
            print_shape_and_value(RHS_at_left_boundary, "RHS_at_left_boundary after squeeze")
            print_shape_and_value(residual_left_boundary, "residual_left_boundary")
            print_shape_and_value(loss_left_boundary, "loss_left_boundary")

            u_xx_right_boundary = tf.squeeze(u_xx_right_boundary)
            RHS_at_right_boundary = tf.squeeze( self.RHS_function(right_boundary_point) )
            residual_right_boundary = u_xx_right_boundary - RHS_at_right_boundary
            loss_right_boundary = tf.math.squared_difference(u_xx_right_boundary, RHS_at_right_boundary)

            print_shape_and_value(u_xx_right_boundary, "u_xx_right_boundary after squeeze")
            print_shape_and_value(RHS_at_right_boundary, "RHS_at_right_boundary after squeeze")
            print_shape_and_value(residual_right_boundary, "residual_right_boundary")
            print_shape_and_value(loss_right_boundary, "loss_right_boundary")

            boundary_loss = self.boundary_loss_coef * (loss_left_boundary + loss_right_boundary) / 2
            print_shape_and_value(boundary_loss, "boundary_loss")

            # loss = boundary_loss + equation_loss
            loss_dict = {
                "equation loss": equation_loss,
                "boundary loss": boundary_loss
            }

            tf.print("losses:")
            print_tensor(list(loss_dict.values()))
        
        # self.optimizer.minimize(loss_dict, self.trainable_variables, tape=weights_update_tape)
        grads = weights_update_tape.gradient(target=loss_dict, sources=self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        tf.print("grads:")
        print_tensor(grads)

        print("tracing step, printed using python print")
        # this is here to know, when the tracing is done and the actually loops start

        return loss_dict


class Poisson_problem_1D_pointwise_faster(keras.Sequential):
    """
    .
    """
    def __init__(self, RHS_function, left_boundary_point, right_boundary_point, *args, equation_loss_coef=1, boundary_loss_coef=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.RHS_function = RHS_function
        self.equation_loss_coef = equation_loss_coef
        self.boundary_loss_coef = boundary_loss_coef
        self.left_boundary_point = left_boundary_point
        self.right_boundary_point = right_boundary_point

    # @tf.function
    def train_step(self, data):
        x, y_true = data

        input_type = x.dtype
        left_boundary_point = tf.constant([[self.left_boundary_point]], dtype=input_type) # needs shape (1, 1) so it can be fed into a dense layer
        # left_boundary_point = tf.Variable(initial_value=[[self.left_boundary_point]], trainable=True, dtype=input_type)

        right_boundary_point = tf.constant([[self.right_boundary_point]], dtype=input_type) # needs shape (1, 1) so it can be fed into a dense layer
        # right_boundary_point = tf.Variable(initial_value=[[self.right_boundary_point]], trainable=True, dtype=input_type)

        with tf.GradientTape() as weights_update_tape:
            with tf.GradientTape(watch_accessed_variables=False, persistent=True) as second_der_tape:
                second_der_tape.watch(x)    # shape of x: (batch_size, 1)

                with tf.GradientTape(watch_accessed_variables=False, persistent=True) as first_der_tape:
                    first_der_tape.watch(x) # shape of x: (batch_size, 1)
                    prediction = self(x, training=True)   # shape: (batch_size, 1)

                # CAREFULL: derivatives wouldn't work correctly like this if I had multiple outpust, there would be some
                # modifications needed: taking batch_jacobian for first derivative and extracting the diagonal derivatives.
                # # gradients for multiple outputs would just sum the derivatives of individial outpust for fixed x
                # # (it would just act as gradient does regurarly for multiple targets, see docs: 
                # # https://www.tensorflow.org/guide/autodiff#gradients_of_non-scalar_targets)
                u_x = first_der_tape.gradient(target=prediction, sources=x)   # shape: (batch_of_points, 1)
                del first_der_tape

                # tf.print("u_x:")
                # print_tensor(u_x)

            u_xx = second_der_tape.gradient(u_x, x) # shape: (batch_of_points, 1)
            del second_der_tape

            # tf.print("u_xx:")
            # print_tensor(u_xx)

            # u_xx = tf.squeeze(u_xx) # shape: (batch_size)
            # RHS_evaluated = tf.squeeze( self.RHS_function(x) )  # shape: (batch_size)
            # residual = u_xx - RHS_evaluated
            # equation_loss = self.equation_loss_coef * tf.math.reduce_mean( tf.math.square(residual) )
            RHS_evaluated = self.RHS_function(x) # shape (batch_size, 1)
            equation_loss = self.equation_loss_coef * tf.math.reduce_mean( tf.math.squared_difference(u_xx, RHS_evaluated) )

            prediction_left_boundary = self(left_boundary_point, training=True) # shape: (1, 1)
            prediction_right_boundary = self(right_boundary_point, training=True)   # shape: (1, 1)

            loss_left_boundary = tf.math.squared_difference(prediction_left_boundary, tf.zeros_like(prediction_left_boundary))
            loss_right_boundary = tf.math.squared_difference(prediction_right_boundary, tf.zeros_like(prediction_right_boundary))
            boundary_loss = self.boundary_loss_coef * (loss_left_boundary + loss_right_boundary) / 2

            # loss = boundary_loss + equation_loss
            loss_dict = {
                "equation loss": equation_loss,
                "boundary loss": boundary_loss
            }
        
        # self.optimizer.minimize(loss_dict, self.trainable_variables, tape=weights_update_tape)
        grads = weights_update_tape.gradient(target=loss_dict, sources=self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss_dict


class Poisson_problem_1D_pointwise_random_resampling_in_train_step(keras.Sequential):
    """
    .
    """
    def __init__(
        self,
        RHS_function,
        left_boundary_point,
        right_boundary_point,
        number_of_points_to_sample,
        *args,
        equation_loss_coef=1,
        boundary_loss_coef=1,
        input_type=tf.float64,
        **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.RHS_function = RHS_function
        self.equation_loss_coef = equation_loss_coef
        self.boundary_loss_coef = boundary_loss_coef
        self.left_boundary_point = left_boundary_point
        self.right_boundary_point = right_boundary_point
        self.number_of_points_to_sample = number_of_points_to_sample
        self.input_type = input_type

    # @tf.function
    def train_step(self, data):
        x = tf.random.uniform(
            [self.number_of_points_to_sample, 1],
            minval=self.left_boundary_point,
            maxval=self.right_boundary_point,
            dtype=self.input_type
            )

        left_boundary_point = tf.constant([[self.left_boundary_point]], dtype=self.input_type) # needs shape (1, 1) so it can be fed into a dense layer

        right_boundary_point = tf.constant([[self.right_boundary_point]], dtype=self.input_type) # needs shape (1, 1) so it can be fed into a dense layer

        with tf.GradientTape() as weights_update_tape:
            with tf.GradientTape(watch_accessed_variables=False, persistent=True) as second_der_tape:
                second_der_tape.watch(x)    # shape of x: (batch_size, 1)
                second_der_tape.watch(left_boundary_point)
                second_der_tape.watch(right_boundary_point)

                with tf.GradientTape(watch_accessed_variables=False, persistent=True) as first_der_tape:
                    first_der_tape.watch(x) # shape of x: (batch_size, 1)
                    first_der_tape.watch(left_boundary_point)
                    first_der_tape.watch(right_boundary_point)
                    prediction = self(x, training=True)   # shape: (batch_size, 1)
                    prediction_left_boundary = self(left_boundary_point, training=True) # shape: (1, 1)
                    prediction_right_boundary = self(right_boundary_point, training=True)   # shape: (1, 1)

                # CAREFULL: derivatives wouldn't work correctly like this if I had multiple outpust, there would be some
                # modifications needed: taking batch_jacobian for first derivative and extracting the diagonal derivatives.
                # # gradients for multiple outputs would just sum the derivatives of individial outpust for fixed x
                # # (it would just act as gradient does regurarly for multiple targets, see docs: 
                # # https://www.tensorflow.org/guide/autodiff#gradients_of_non-scalar_targets)
                u_x = first_der_tape.gradient(target=prediction, sources=x)   # shape: (batch_of_points, 1)
                u_x_left_boundary = first_der_tape.gradient(target=prediction_left_boundary, sources=left_boundary_point)   # shape: (1, 1)
                u_x_right_boundary = first_der_tape.gradient(target=prediction_right_boundary, sources=right_boundary_point)    # shape: (1, 1)
                del first_der_tape

            u_xx = second_der_tape.batch_jacobian(target=u_x, source=x) # shape: (batch_of_points, 1, 1)
            u_xx_left_boundary = second_der_tape.batch_jacobian(target=u_x_left_boundary, source=left_boundary_point) # shape: (1, 1, 1)
            u_xx_right_boundary = second_der_tape.batch_jacobian(target=u_x_right_boundary, source=right_boundary_point) # shape: (1, 1, 1)
            del second_der_tape

            u_xx = tf.squeeze(u_xx) # shape: (batch_size)
            RHS_evaluated = tf.squeeze( self.RHS_function(x) )  # shape: (batch_size)
            # residual = u_xx - RHS_evaluated
            # equation_loss = self.equation_loss_coef * tf.math.reduce_mean( tf.math.square(residual) )
            equation_loss = self.equation_loss_coef * tf.math.reduce_mean( tf.math.squared_difference(u_xx, RHS_evaluated) )

            u_xx_left_boundary = tf.squeeze(u_xx_left_boundary)
            RHS_at_left_boundary = tf.squeeze( self.RHS_function(left_boundary_point) )
            loss_left_boundary = tf.math.squared_difference(u_xx_left_boundary, RHS_at_left_boundary)

            u_xx_right_boundary = tf.squeeze(u_xx_right_boundary)
            RHS_at_right_boundary = tf.squeeze( self.RHS_function(right_boundary_point) )
            loss_right_boundary = tf.math.squared_difference(u_xx_right_boundary, RHS_at_right_boundary)

            boundary_loss = self.boundary_loss_coef * (loss_left_boundary + loss_right_boundary) / 2

            # loss = boundary_loss + equation_loss
            loss_dict = {
                "equation loss": equation_loss,
                "boundary loss": boundary_loss
            }
        
        # self.optimizer.minimize(loss_dict, self.trainable_variables, tape=weights_update_tape)
        grads = weights_update_tape.gradient(target=loss_dict, sources=self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss_dict


class Poisson_problem_1D_pointwise_RAR(keras.Sequential):
    """
    .
    """
    def __init__(
        self,
        RHS_function,
        left_boundary_point,
        right_boundary_point,
        RAR_error_threshhold,
        RAR_add_points_after_number_of_epochs,
        *args,
        equation_loss_coef=1,
        boundary_loss_coef=1,
        RAR_number_of_points_to_add=1,
        **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.RHS_function = RHS_function
        self.equation_loss_coef = equation_loss_coef
        self.boundary_loss_coef = boundary_loss_coef
        self.left_boundary_point = left_boundary_point
        self.right_boundary_point = right_boundary_point

    # @tf.function
    def train_step(self, data):
        x, y_true = data

        input_type = x.dtype
        left_boundary_point = tf.constant([[self.left_boundary_point]], dtype=input_type) # needs shape (1, 1) so it can be fed into a dense layer
        # left_boundary_point = tf.Variable(initial_value=[[self.left_boundary_point]], trainable=True, dtype=input_type)

        right_boundary_point = tf.constant([[self.right_boundary_point]], dtype=input_type) # needs shape (1, 1) so it can be fed into a dense layer
        # right_boundary_point = tf.Variable(initial_value=[[self.right_boundary_point]], trainable=True, dtype=input_type)

        with tf.GradientTape() as weights_update_tape:
            with tf.GradientTape(watch_accessed_variables=False, persistent=True) as second_der_tape:
                second_der_tape.watch(x)    # shape of x: (batch_size, 1)
                second_der_tape.watch(left_boundary_point)
                second_der_tape.watch(right_boundary_point)

                with tf.GradientTape(watch_accessed_variables=False, persistent=True) as first_der_tape:
                    first_der_tape.watch(x) # shape of x: (batch_size, 1)
                    first_der_tape.watch(left_boundary_point)
                    first_der_tape.watch(right_boundary_point)
                    prediction = self(x, training=True)   # shape: (batch_size, 1)
                    prediction_left_boundary = self(left_boundary_point, training=True) # shape: (1, 1)
                    prediction_right_boundary = self(right_boundary_point, training=True)   # shape: (1, 1)

                # CAREFULL: derivatives wouldn't work correctly like this if I had multiple outpust, there would be some
                # modifications needed: taking batch_jacobian for first derivative and extracting the diagonal derivatives.
                # # gradients for multiple outputs would just sum the derivatives of individial outpust for fixed x
                # # (it would just act as gradient does regurarly for multiple targets, see docs: 
                # # https://www.tensorflow.org/guide/autodiff#gradients_of_non-scalar_targets)
                u_x = first_der_tape.gradient(target=prediction, sources=x)   # shape: (batch_of_points, 1)
                u_x_left_boundary = first_der_tape.gradient(target=prediction_left_boundary, sources=left_boundary_point)   # shape: (1, 1)
                u_x_right_boundary = first_der_tape.gradient(target=prediction_right_boundary, sources=right_boundary_point)    # shape: (1, 1)
                del first_der_tape

            u_xx = second_der_tape.batch_jacobian(target=u_x, source=x) # shape: (batch_of_points, 1, 1)
            u_xx_left_boundary = second_der_tape.batch_jacobian(target=u_x_left_boundary, source=left_boundary_point) # shape: (1, 1, 1)
            u_xx_right_boundary = second_der_tape.batch_jacobian(target=u_x_right_boundary, source=right_boundary_point) # shape: (1, 1, 1)
            del second_der_tape

            u_xx = tf.squeeze(u_xx) # shape: (batch_size)
            RHS_evaluated = tf.squeeze( self.RHS_function(x) )  # shape: (batch_size)
            # residual = u_xx - RHS_evaluated
            # equation_loss = self.equation_loss_coef * tf.math.reduce_mean( tf.math.square(residual) )
            equation_loss = self.equation_loss_coef * tf.math.reduce_mean( tf.math.squared_difference(u_xx, RHS_evaluated) )

            u_xx_left_boundary = tf.squeeze(u_xx_left_boundary)
            RHS_at_left_boundary = tf.squeeze( self.RHS_function(left_boundary_point) )
            loss_left_boundary = tf.math.squared_difference(u_xx_left_boundary, RHS_at_left_boundary)

            u_xx_right_boundary = tf.squeeze(u_xx_right_boundary)
            RHS_at_right_boundary = tf.squeeze( self.RHS_function(right_boundary_point) )
            loss_right_boundary = tf.math.squared_difference(u_xx_right_boundary, RHS_at_right_boundary)

            boundary_loss = self.boundary_loss_coef * (loss_left_boundary + loss_right_boundary) / 2

            # loss = boundary_loss + equation_loss
            loss_dict = {
                "equation loss": equation_loss,
                "boundary loss": boundary_loss
            }
        
        # self.optimizer.minimize(loss_dict, self.trainable_variables, tape=weights_update_tape)
        grads = weights_update_tape.gradient(target=loss_dict, sources=self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss_dict


class Poisson_problem_1D_pointwise_gpinn(keras.Sequential):
    """
    .
    """
    def __init__(self,
    RHS_function,
    left_boundary_point,
    right_boundary_point,
    *args,
    equation_loss_coef=1,
    boundary_loss_coef=1,
    residual_gradient_coef=0.01,
    **kwargs):
        super().__init__(*args, **kwargs)
        self.RHS_function = RHS_function
        self.equation_loss_coef = equation_loss_coef
        self.boundary_loss_coef = boundary_loss_coef
        self.residual_gradient_coef = residual_gradient_coef
        self.left_boundary_point = left_boundary_point
        self.right_boundary_point = right_boundary_point

    def train_step(self, data):
        x, y_true = data

        # batch_size = len(x[:,0])
        input_type = x.dtype
        left_boundary_point = tf.constant([[self.left_boundary_point]], dtype=input_type) # needs shape (1, 1) so it can be fed into a dense layer
        right_boundary_point = tf.constant([[self.right_boundary_point]], dtype=input_type) # needs shape (1, 1) so it can be fed into a dense layer

        with tf.GradientTape() as weights_update_tape:
            with tf.GradientTape(watch_accessed_variables=False, persistent=True) as gpinn_tape:
                gpinn_tape.watch(x)
                gpinn_tape.watch(left_boundary_point)
                gpinn_tape.watch(right_boundary_point)
                with tf.GradientTape(watch_accessed_variables=False, persistent=True) as second_der_tape:
                    second_der_tape.watch(x)    # shape of x: (batch_size, 1)
                    second_der_tape.watch(left_boundary_point)
                    second_der_tape.watch(right_boundary_point)

                    with tf.GradientTape(watch_accessed_variables=False, persistent=True) as first_der_tape:
                        first_der_tape.watch(x) # shape of x: (batch_size, 1)
                        first_der_tape.watch(left_boundary_point)
                        first_der_tape.watch(right_boundary_point)
                        prediction = self(x, training=True)   # shape: (batch_size, 1)
                        prediction_left_boundary = self(left_boundary_point, training=True) # shape: (1, 1)
                        prediction_right_boundary = self(right_boundary_point, training=True)   # shape: (1, 1)

                    # CAREFULL: derivatives wouldn't work correctly like this if I had multiple outpust, there would be some
                    # modifications needed: taking batch_jacobian for first derivative and extracting the diagonal derivatives.
                    # # gradients for multiple outputs would just sum the derivatives of individial outpust for fixed x
                    # # (it would just act as gradient does regurarly for multiple targets, see docs: 
                    # # https://www.tensorflow.org/guide/autodiff#gradients_of_non-scalar_targets)
                    u_x = first_der_tape.gradient(target=prediction, sources=x)   # shape: (batch_of_points, 1)
                    u_x_left_boundary = first_der_tape.gradient(target=prediction_left_boundary, sources=left_boundary_point)   # shape: (1, 1)
                    u_x_right_boundary = first_der_tape.gradient(target=prediction_right_boundary, sources=right_boundary_point)    # shape: (1, 1)
                    del first_der_tape

                u_xx = second_der_tape.batch_jacobian(target=u_x, source=x) # shape: (batch_of_points, 1, 1)
                u_xx_left_boundary = second_der_tape.batch_jacobian(target=u_x_left_boundary, source=left_boundary_point) # shape: (1, 1, 1)
                u_xx_right_boundary = second_der_tape.batch_jacobian(target=u_x_right_boundary, source=right_boundary_point) # shape: (1, 1, 1)
                del second_der_tape

                u_xx = tf.squeeze(u_xx) # shape: (batch_size)
                RHS_evaluated = tf.squeeze( self.RHS_function(x) )  # shape: (batch_size)
                residual = u_xx - RHS_evaluated # shape: (batch_size)
                residual = tf.expand_dims(residual, axis=-1)    # resize to shape: (batch_size, 1) so taht batch_jacobian
                                                                # method can be called later

                u_xx_left_boundary = tf.squeeze(u_xx_left_boundary) # shape: (,)
                RHS_at_left_boundary = tf.squeeze( self.RHS_function(left_boundary_point) ) # shape: (,)
                residual_left_boundary = u_xx_left_boundary - RHS_at_left_boundary  # shape: (,)

                u_xx_right_boundary = tf.squeeze(u_xx_right_boundary)   # shape: (,)
                RHS_at_right_boundary = tf.squeeze( self.RHS_function(right_boundary_point) )   # shape: (,)
                residual_right_boundary = u_xx_right_boundary - RHS_at_right_boundary   # shape: (,)

            # print(residual)
            # print(x)
            dr_dx = gpinn_tape.batch_jacobian(residual, x)  # shape: (batch_size, 1, 1)
            # print(dr_dx.shape)
            dr_dx_left_boundary = gpinn_tape.jacobian(residual_left_boundary, left_boundary_point)  # shape: (1, 1)
            dr_dx_left_boundary = tf.expand_dims(dr_dx_left_boundary, axis=-1)  # shape: (1, 1, 1)
                # change shape so that concat can be called
            # print(dr_dx_left_boundary.shape)
            dr_dx_right_boundary = gpinn_tape.jacobian(residual_right_boundary, right_boundary_point)   # shape: (1, 1)
            dr_dx_right_boundary = tf.expand_dims(dr_dx_right_boundary, axis=-1)    # shape: (1, 1, 1)
                # change shape so that concat can be called
            # print(dr_dx_right_boundary.shape)
            dr_dx_total = tf.concat([dr_dx, dr_dx_left_boundary, dr_dx_right_boundary], axis=0)

            # dr_dx_square = tf.math.square(dr_dx)    # shape: (batch_size, 1)
            # dr_dx_square_left_boundary = tf.math.square(dr_dx_left_boundary)    # shape: (1, 1)
            # dr_dx_square_right_boundary = tf.math.square(dr_dx_right_boundary)  # shape: (1, 1)
            # dr_dx_total = tf.concat([dr_dx_square, dr_dx_square_left_boundary, dr_dx_square_right_boundary], axis=0)
            # total_dr_dx_mean = (dr_dx_square + dr_dx_square_left_boundary + dr_dx_square_right_boundary) / (batch_size + 1 + 1)

            dr_dx_loss = self.residual_gradient_coef * tf.reduce_mean(tf.square(dr_dx_total))

            equation_loss = self.equation_loss_coef * tf.math.reduce_mean( tf.math.square(residual) )
            
            loss_left_boundary = tf.math.square(residual_left_boundary)
            loss_right_boundary = tf.math.square(residual_right_boundary)
            boundary_loss = self.boundary_loss_coef * (loss_left_boundary + loss_right_boundary) / 2

            loss_dict = {
                "equation loss": equation_loss,
                "boundary loss": boundary_loss,
                "residual gradient loss": dr_dx_loss
            }
        
        self.optimizer.minimize(loss_dict, self.trainable_variables, tape=weights_update_tape)
        # grads = weights_update_tape.gradient(target=loss_dict, sources=self.trainable_variables)
        # self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss_dict


class Poisson_problem_1D_pointwise_gpinn_max_norm(keras.Sequential):
    """
    .
    """
    def __init__(self,
    RHS_function,
    left_boundary_point,
    right_boundary_point,
    *args,
    equation_loss_coef=1,
    boundary_loss_coef=1,
    residual_gradient_coef=0.01,
    **kwargs):
        super().__init__(*args, **kwargs)
        self.RHS_function = RHS_function
        self.equation_loss_coef = equation_loss_coef
        self.boundary_loss_coef = boundary_loss_coef
        self.residual_gradient_coef = residual_gradient_coef
        self.left_boundary_point = left_boundary_point
        self.right_boundary_point = right_boundary_point

    def train_step(self, data):
        x, y_true = data

        # batch_size = len(x[:,0])
        input_type = x.dtype
        left_boundary_point = tf.constant([[self.left_boundary_point]], dtype=input_type) # needs shape (1, 1) so it can be fed into a dense layer
        right_boundary_point = tf.constant([[self.right_boundary_point]], dtype=input_type) # needs shape (1, 1) so it can be fed into a dense layer

        with tf.GradientTape() as weights_update_tape:
            with tf.GradientTape(watch_accessed_variables=False, persistent=True) as gpinn_tape:
                gpinn_tape.watch(x)
                gpinn_tape.watch(left_boundary_point)
                gpinn_tape.watch(right_boundary_point)
                with tf.GradientTape(watch_accessed_variables=False, persistent=True) as second_der_tape:
                    second_der_tape.watch(x)    # shape of x: (batch_size, 1)
                    second_der_tape.watch(left_boundary_point)
                    second_der_tape.watch(right_boundary_point)

                    with tf.GradientTape(watch_accessed_variables=False, persistent=True) as first_der_tape:
                        first_der_tape.watch(x) # shape of x: (batch_size, 1)
                        first_der_tape.watch(left_boundary_point)
                        first_der_tape.watch(right_boundary_point)
                        prediction = self(x, training=True)   # shape: (batch_size, 1)
                        prediction_left_boundary = self(left_boundary_point, training=True) # shape: (1, 1)
                        prediction_right_boundary = self(right_boundary_point, training=True)   # shape: (1, 1)

                    # CAREFULL: derivatives wouldn't work correctly like this if I had multiple outpust, there would be some
                    # modifications needed: taking batch_jacobian for first derivative and extracting the diagonal derivatives.
                    # # gradients for multiple outputs would just sum the derivatives of individial outpust for fixed x
                    # # (it would just act as gradient does regurarly for multiple targets, see docs: 
                    # # https://www.tensorflow.org/guide/autodiff#gradients_of_non-scalar_targets)
                    u_x = first_der_tape.gradient(target=prediction, sources=x)   # shape: (batch_of_points, 1)
                    u_x_left_boundary = first_der_tape.gradient(target=prediction_left_boundary, sources=left_boundary_point)   # shape: (1, 1)
                    u_x_right_boundary = first_der_tape.gradient(target=prediction_right_boundary, sources=right_boundary_point)    # shape: (1, 1)
                    del first_der_tape

                u_xx = second_der_tape.batch_jacobian(target=u_x, source=x) # shape: (batch_of_points, 1, 1)
                u_xx_left_boundary = second_der_tape.batch_jacobian(target=u_x_left_boundary, source=left_boundary_point) # shape: (1, 1, 1)
                u_xx_right_boundary = second_der_tape.batch_jacobian(target=u_x_right_boundary, source=right_boundary_point) # shape: (1, 1, 1)
                del second_der_tape

                u_xx = tf.squeeze(u_xx) # shape: (batch_size)
                RHS_evaluated = tf.squeeze( self.RHS_function(x) )  # shape: (batch_size)
                residual = u_xx - RHS_evaluated # shape: (batch_size)
                residual = tf.expand_dims(residual, axis=-1)    # resize to shape: (batch_size, 1) so taht batch_jacobian
                                                                # method can be called later

                u_xx_left_boundary = tf.squeeze(u_xx_left_boundary) # shape: (,)
                RHS_at_left_boundary = tf.squeeze( self.RHS_function(left_boundary_point) ) # shape: (,)
                residual_left_boundary = u_xx_left_boundary - RHS_at_left_boundary  # shape: (,)

                u_xx_right_boundary = tf.squeeze(u_xx_right_boundary)   # shape: (,)
                RHS_at_right_boundary = tf.squeeze( self.RHS_function(right_boundary_point) )   # shape: (,)
                residual_right_boundary = u_xx_right_boundary - RHS_at_right_boundary   # shape: (,)

            # print(residual)
            # print(x)
            dr_dx = gpinn_tape.batch_jacobian(residual, x)  # shape: (batch_size, 1, 1)
            # print(dr_dx.shape)
            dr_dx_left_boundary = gpinn_tape.jacobian(residual_left_boundary, left_boundary_point)  # shape: (1, 1)
            dr_dx_left_boundary = tf.expand_dims(dr_dx_left_boundary, axis=-1)  # shape: (1, 1, 1)
                # change shape so that concat can be called
            # print(dr_dx_left_boundary.shape)
            dr_dx_right_boundary = gpinn_tape.jacobian(residual_right_boundary, right_boundary_point)   # shape: (1, 1)
            dr_dx_right_boundary = tf.expand_dims(dr_dx_right_boundary, axis=-1)    # shape: (1, 1, 1)
                # change shape so that concat can be called
            # print(dr_dx_right_boundary.shape)
            dr_dx_total = tf.concat([dr_dx, dr_dx_left_boundary, dr_dx_right_boundary], axis=0)

            # dr_dx_square = tf.math.square(dr_dx)    # shape: (batch_size, 1)
            # dr_dx_square_left_boundary = tf.math.square(dr_dx_left_boundary)    # shape: (1, 1)
            # dr_dx_square_right_boundary = tf.math.square(dr_dx_right_boundary)  # shape: (1, 1)
            # dr_dx_total = tf.concat([dr_dx_square, dr_dx_square_left_boundary, dr_dx_square_right_boundary], axis=0)
            # total_dr_dx_mean = (dr_dx_square + dr_dx_square_left_boundary + dr_dx_square_right_boundary) / (batch_size + 1 + 1)

            dr_dx_loss = self.residual_gradient_coef * tf.reduce_mean(tf.square(dr_dx_total))

            equation_loss = self.equation_loss_coef * tf.math.reduce_max(tf.math.abs(residual))
            
            boundary_values_tensor = tf.stack([residual_left_boundary, residual_right_boundary], axis=0)
            boundary_loss = self.boundary_loss_coef * tf.math.reduce_max(tf.math.abs(boundary_values_tensor))

            loss_dict = {
                "equation loss": equation_loss,
                "boundary loss": boundary_loss,
                "residual gradient loss": dr_dx_loss
            }
        
        self.optimizer.minimize(loss_dict, self.trainable_variables, tape=weights_update_tape)
        # grads = weights_update_tape.gradient(target=loss_dict, sources=self.trainable_variables)
        # self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss_dict


class Poisson_problem_1D_pointwise_gpinn_MSE_and_max_norm(keras.Sequential):
    """
    .
    """
    def __init__(self,
    RHS_function,
    left_boundary_point,
    right_boundary_point,
    *args,
    MSE_equation_loss_coef=1,
    MSE_boundary_loss_coef=1,
    max_norm_equation_loss_coef=1,
    max_norm_boundary_loss_coef=1,
    residual_gradient_coef=0.01,
    **kwargs):
        super().__init__(*args, **kwargs)
        self.RHS_function = RHS_function
        self.MSE_equation_loss_coef = MSE_equation_loss_coef
        self.MSE_boundary_loss_coef = MSE_boundary_loss_coef
        self.max_norm_equation_loss_coef = max_norm_equation_loss_coef
        self.max_norm_boundary_loss_coef = max_norm_boundary_loss_coef
        self.residual_gradient_coef = residual_gradient_coef
        self.left_boundary_point = left_boundary_point
        self.right_boundary_point = right_boundary_point

    def train_step(self, data):
        x, y_true = data

        # batch_size = len(x[:,0])
        input_type = x.dtype
        left_boundary_point = tf.constant([[self.left_boundary_point]], dtype=input_type) # needs shape (1, 1) so it can be fed into a dense layer
        right_boundary_point = tf.constant([[self.right_boundary_point]], dtype=input_type) # needs shape (1, 1) so it can be fed into a dense layer

        with tf.GradientTape() as weights_update_tape:
            with tf.GradientTape(watch_accessed_variables=False, persistent=True) as gpinn_tape:
                gpinn_tape.watch(x)
                gpinn_tape.watch(left_boundary_point)
                gpinn_tape.watch(right_boundary_point)
                with tf.GradientTape(watch_accessed_variables=False, persistent=True) as second_der_tape:
                    second_der_tape.watch(x)    # shape of x: (batch_size, 1)
                    second_der_tape.watch(left_boundary_point)
                    second_der_tape.watch(right_boundary_point)

                    with tf.GradientTape(watch_accessed_variables=False, persistent=True) as first_der_tape:
                        first_der_tape.watch(x) # shape of x: (batch_size, 1)
                        first_der_tape.watch(left_boundary_point)
                        first_der_tape.watch(right_boundary_point)
                        prediction = self(x, training=True)   # shape: (batch_size, 1)
                        prediction_left_boundary = self(left_boundary_point, training=True) # shape: (1, 1)
                        prediction_right_boundary = self(right_boundary_point, training=True)   # shape: (1, 1)

                    # CAREFULL: derivatives wouldn't work correctly like this if I had multiple outpust, there would be some
                    # modifications needed: taking batch_jacobian for first derivative and extracting the diagonal derivatives.
                    # # gradients for multiple outputs would just sum the derivatives of individial outpust for fixed x
                    # # (it would just act as gradient does regurarly for multiple targets, see docs: 
                    # # https://www.tensorflow.org/guide/autodiff#gradients_of_non-scalar_targets)
                    u_x = first_der_tape.gradient(target=prediction, sources=x)   # shape: (batch_of_points, 1)
                    u_x_left_boundary = first_der_tape.gradient(target=prediction_left_boundary, sources=left_boundary_point)   # shape: (1, 1)
                    u_x_right_boundary = first_der_tape.gradient(target=prediction_right_boundary, sources=right_boundary_point)    # shape: (1, 1)
                    del first_der_tape

                u_xx = second_der_tape.batch_jacobian(target=u_x, source=x) # shape: (batch_of_points, 1, 1)
                u_xx_left_boundary = second_der_tape.batch_jacobian(target=u_x_left_boundary, source=left_boundary_point) # shape: (1, 1, 1)
                u_xx_right_boundary = second_der_tape.batch_jacobian(target=u_x_right_boundary, source=right_boundary_point) # shape: (1, 1, 1)
                del second_der_tape

                u_xx = tf.squeeze(u_xx) # shape: (batch_size)
                RHS_evaluated = tf.squeeze( self.RHS_function(x) )  # shape: (batch_size)
                residual = u_xx - RHS_evaluated # shape: (batch_size)
                residual = tf.expand_dims(residual, axis=-1)    # resize to shape: (batch_size, 1) so taht batch_jacobian
                                                                # method can be called later

                u_xx_left_boundary = tf.squeeze(u_xx_left_boundary) # shape: (,)
                RHS_at_left_boundary = tf.squeeze( self.RHS_function(left_boundary_point) ) # shape: (,)
                residual_left_boundary = u_xx_left_boundary - RHS_at_left_boundary  # shape: (,)

                u_xx_right_boundary = tf.squeeze(u_xx_right_boundary)   # shape: (,)
                RHS_at_right_boundary = tf.squeeze( self.RHS_function(right_boundary_point) )   # shape: (,)
                residual_right_boundary = u_xx_right_boundary - RHS_at_right_boundary   # shape: (,)

            # print(residual)
            # print(x)
            dr_dx = gpinn_tape.batch_jacobian(residual, x)  # shape: (batch_size, 1, 1)
            # print(dr_dx.shape)
            dr_dx_left_boundary = gpinn_tape.jacobian(residual_left_boundary, left_boundary_point)  # shape: (1, 1)
            dr_dx_left_boundary = tf.expand_dims(dr_dx_left_boundary, axis=-1)  # shape: (1, 1, 1)
                # change shape so that concat can be called
            # print(dr_dx_left_boundary.shape)
            dr_dx_right_boundary = gpinn_tape.jacobian(residual_right_boundary, right_boundary_point)   # shape: (1, 1)
            dr_dx_right_boundary = tf.expand_dims(dr_dx_right_boundary, axis=-1)    # shape: (1, 1, 1)
                # change shape so that concat can be called
            # print(dr_dx_right_boundary.shape)
            dr_dx_total = tf.concat([dr_dx, dr_dx_left_boundary, dr_dx_right_boundary], axis=0)

            # dr_dx_square = tf.math.square(dr_dx)    # shape: (batch_size, 1)
            # dr_dx_square_left_boundary = tf.math.square(dr_dx_left_boundary)    # shape: (1, 1)
            # dr_dx_square_right_boundary = tf.math.square(dr_dx_right_boundary)  # shape: (1, 1)
            # dr_dx_total = tf.concat([dr_dx_square, dr_dx_square_left_boundary, dr_dx_square_right_boundary], axis=0)
            # total_dr_dx_mean = (dr_dx_square + dr_dx_square_left_boundary + dr_dx_square_right_boundary) / (batch_size + 1 + 1)

            dr_dx_loss = self.residual_gradient_coef * tf.reduce_mean(tf.square(dr_dx_total))


            max_norm_equation_loss = self.max_norm_equation_loss_coef * tf.math.reduce_max(tf.math.abs(residual))
            
            max_norm_boundary_values_tensor = tf.stack([residual_left_boundary, residual_right_boundary], axis=0)
            max_norm_boundary_loss = self.max_norm_boundary_loss_coef * tf.math.reduce_max(tf.math.abs(max_norm_boundary_values_tensor))


            MSE_equation_loss = self.MSE_equation_loss_coef * tf.math.reduce_mean( tf.math.square(residual) )
            
            MSE_loss_left_boundary = tf.math.square(residual_left_boundary)
            MSE_loss_right_boundary = tf.math.square(residual_right_boundary)
            MSE_boundary_loss = self.MSE_boundary_loss_coef * (MSE_loss_left_boundary + MSE_loss_right_boundary) / 2

            loss_dict = {
                "max norm equation loss": max_norm_equation_loss,
                "max norm boundary loss": max_norm_boundary_loss,
                "MSE equation loss": MSE_equation_loss,
                "MSE boundary loss": MSE_boundary_loss,
                "residual gradient loss": dr_dx_loss
            }
        
        self.optimizer.minimize(loss_dict, self.trainable_variables, tape=weights_update_tape)
        # grads = weights_update_tape.gradient(target=loss_dict, sources=self.trainable_variables)
        # self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss_dict


class Poisson_problem_1D_pointwise_gpinn_without_boundary(keras.Sequential):
    """
    .
    """
    def __init__(self,
    RHS_function,
    left_boundary_point,
    right_boundary_point,
    *args,
    equation_loss_coef=1,
    boundary_loss_coef=1,
    residual_gradient_coef=0.01,
    **kwargs):
        super().__init__(*args, **kwargs)
        self.RHS_function = RHS_function
        self.equation_loss_coef = equation_loss_coef
        self.boundary_loss_coef = boundary_loss_coef
        self.residual_gradient_coef = residual_gradient_coef
        self.left_boundary_point = left_boundary_point
        self.right_boundary_point = right_boundary_point

    def train_step(self, data):
        x, y_true = data

        # batch_size = len(x[:,0])
        input_type = x.dtype
        left_boundary_point = tf.constant([[self.left_boundary_point]], dtype=input_type) # needs shape (1, 1) so it can be fed into a dense layer
        right_boundary_point = tf.constant([[self.right_boundary_point]], dtype=input_type) # needs shape (1, 1) so it can be fed into a dense layer

        with tf.GradientTape() as weights_update_tape:
            with tf.GradientTape(watch_accessed_variables=False) as gpinn_tape:
                gpinn_tape.watch(x)
                with tf.GradientTape(watch_accessed_variables=False, persistent=True) as second_der_tape:
                    second_der_tape.watch(x)    # shape of x: (batch_size, 1)
                    second_der_tape.watch(left_boundary_point)
                    second_der_tape.watch(right_boundary_point)

                    with tf.GradientTape(watch_accessed_variables=False, persistent=True) as first_der_tape:
                        first_der_tape.watch(x) # shape of x: (batch_size, 1)
                        first_der_tape.watch(left_boundary_point)
                        first_der_tape.watch(right_boundary_point)
                        prediction = self(x, training=True)   # shape: (batch_size, 1)
                        prediction_left_boundary = self(left_boundary_point, training=True) # shape: (1, 1)
                        prediction_right_boundary = self(right_boundary_point, training=True)   # shape: (1, 1)

                    # CAREFULL: derivatives wouldn't work correctly like this if I had multiple outpust, there would be some
                    # modifications needed: taking batch_jacobian for first derivative and extracting the diagonal derivatives.
                    # # gradients for multiple outputs would just sum the derivatives of individial outpust for fixed x
                    # # (it would just act as gradient does regurarly for multiple targets, see docs: 
                    # # https://www.tensorflow.org/guide/autodiff#gradients_of_non-scalar_targets)
                    u_x = first_der_tape.gradient(target=prediction, sources=x)   # shape: (batch_of_points, 1)
                    u_x_left_boundary = first_der_tape.gradient(target=prediction_left_boundary, sources=left_boundary_point)   # shape: (1, 1)
                    u_x_right_boundary = first_der_tape.gradient(target=prediction_right_boundary, sources=right_boundary_point)    # shape: (1, 1)
                    del first_der_tape

                u_xx = second_der_tape.batch_jacobian(target=u_x, source=x) # shape: (batch_of_points, 1, 1)
                u_xx_left_boundary = second_der_tape.batch_jacobian(target=u_x_left_boundary, source=left_boundary_point) # shape: (1, 1, 1)
                u_xx_right_boundary = second_der_tape.batch_jacobian(target=u_x_right_boundary, source=right_boundary_point) # shape: (1, 1, 1)
                del second_der_tape

                u_xx = tf.squeeze(u_xx) # shape: (batch_size)
                RHS_evaluated = tf.squeeze( self.RHS_function(x) )  # shape: (batch_size)
                residual = u_xx - RHS_evaluated # shape: (batch_size)
                residual = tf.expand_dims(residual, axis=-1)    # resize to shape: (batch_size, 1) so taht batch_jacobian
                                                                # method can be called later

                u_xx_left_boundary = tf.squeeze(u_xx_left_boundary) # shape: (,)
                RHS_at_left_boundary = tf.squeeze( self.RHS_function(left_boundary_point) ) # shape: (,)
                residual_left_boundary = u_xx_left_boundary - RHS_at_left_boundary  # shape: (,)

                u_xx_right_boundary = tf.squeeze(u_xx_right_boundary)   # shape: (,)
                RHS_at_right_boundary = tf.squeeze( self.RHS_function(right_boundary_point) )   # shape: (,)
                residual_right_boundary = u_xx_right_boundary - RHS_at_right_boundary   # shape: (,)

            dr_dx = gpinn_tape.batch_jacobian(residual, x)  # shape: (batch_size, 1, 1)

            dr_dx_loss = self.residual_gradient_coef * tf.reduce_mean(tf.square(dr_dx))

            equation_loss = self.equation_loss_coef * tf.math.reduce_mean( tf.math.square(residual) )
            
            loss_left_boundary = tf.math.square(residual_left_boundary)
            loss_right_boundary = tf.math.square(residual_right_boundary)
            boundary_loss = self.boundary_loss_coef * (loss_left_boundary + loss_right_boundary) / 2

            loss_dict = {
                "equation loss": equation_loss,
                "boundary loss": boundary_loss,
                "residual gradient loss": dr_dx_loss
            }
        
        self.optimizer.minimize(loss_dict, self.trainable_variables, tape=weights_update_tape)
        # grads = weights_update_tape.gradient(target=loss_dict, sources=self.trainable_variables)
        # self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss_dict


class Poisson_problem_1D_pointwise_gpinn_hard_boundary(keras.Sequential):
    """
    .
    """
    def __init__(self,
    RHS_function,
    left_boundary_point,
    right_boundary_point,
    *args,
    equation_loss_coef=1,
    residual_gradient_coef=0.01,
    **kwargs):
        super().__init__(*args, **kwargs)
        self.RHS_function = RHS_function
        self.equation_loss_coef = equation_loss_coef
        self.residual_gradient_coef = residual_gradient_coef
        self.left_boundary_point = left_boundary_point
        self.right_boundary_point = right_boundary_point

    def call(self, inputs, training=None, mask=None):
        inputs = tf.expand_dims(inputs, axis=-1)
        # print(inputs.shape)
        output = super().call(inputs=inputs, training=training, mask=mask)
        # print(output.shape)

        return (inputs - self.left_boundary_point) * (inputs - self.right_boundary_point) * output
    
    def train_step(self, data):
        x, y_true = data

        # batch_size = len(x[:,0])
        input_type = x.dtype
        left_boundary_point = tf.constant([[self.left_boundary_point]], dtype=input_type) # needs shape (1, 1) so it can be fed into a dense layer
        right_boundary_point = tf.constant([[self.right_boundary_point]], dtype=input_type) # needs shape (1, 1) so it can be fed into a dense layer

        with tf.GradientTape() as weights_update_tape:
            with tf.GradientTape(watch_accessed_variables=False) as gpinn_tape:
                gpinn_tape.watch(x)

                with tf.GradientTape(watch_accessed_variables=False) as second_der_tape:
                    second_der_tape.watch(x)    # shape of x: (batch_size, 1)

                    with tf.GradientTape(watch_accessed_variables=False) as first_der_tape:
                        first_der_tape.watch(x) # shape of x: (batch_size, 1)
                        prediction = self(x, training=True)   # shape: (batch_size, 1)

                    # CAREFULL: derivatives wouldn't work correctly like this if I had multiple outpust, there would be some
                    # modifications needed: taking batch_jacobian for first derivative and extracting the diagonal derivatives.
                    # # gradients for multiple outputs would just sum the derivatives of individial outpust for fixed x
                    # # (it would just act as gradient does regurarly for multiple targets, see docs: 
                    # # https://www.tensorflow.org/guide/autodiff#gradients_of_non-scalar_targets)

                    u_x = first_der_tape.gradient(target=prediction, sources=x)   # shape: (batch_of_points, 1)

                u_xx = second_der_tape.batch_jacobian(target=u_x, source=x) # shape: (batch_of_points, 1, 1)

                u_xx = tf.squeeze(u_xx) # shape: (batch_size)
                RHS_evaluated = tf.squeeze( self.RHS_function(x) )  # shape: (batch_size)
                residual = u_xx - RHS_evaluated # shape: (batch_size)
                residual = tf.expand_dims(residual, axis=-1)    # resize to shape: (batch_size, 1) so taht batch_jacobian
                                                                # method can be called later

            dr_dx = gpinn_tape.batch_jacobian(residual, x)  # shape: (batch_size, 1, 1)
            dr_dx_loss = self.residual_gradient_coef * tf.reduce_mean(tf.square(dr_dx))

            equation_loss = self.equation_loss_coef * tf.math.reduce_mean( tf.math.square(residual) )
            
            loss_dict = {
                "equation loss": equation_loss,
                "residual gradient loss": dr_dx_loss
            }
        
        self.optimizer.minimize(loss_dict, self.trainable_variables, tape=weights_update_tape)
        # grads = weights_update_tape.gradient(target=loss_dict, sources=self.trainable_variables)
        # self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss_dict


class Poisson_problem_1D_pointwise_hard_dirichlet_boundary(keras.Sequential):
    """
    .
    """
    def __init__(self, RHS_function, left_boundary_point, right_boundary_point, *args, equation_loss_coef=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.RHS_function = RHS_function
        self.equation_loss_coef = equation_loss_coef
        self.left_boundary_point = left_boundary_point
        self.right_boundary_point = right_boundary_point

    def call(self, inputs, training=None, mask=None):
        inputs = tf.expand_dims(inputs, axis=-1)
        x = super().call(inputs=inputs, training=training, mask=mask)

        return (inputs - self.left_boundary_point) * (self.right_boundary_point - inputs) * x

    # @tf.function
    def train_step(self, data):
        x, y_true = data

        input_type = x.dtype
        left_boundary_point = tf.constant([[self.left_boundary_point]], dtype=input_type) # needs shape (1, 1) so it can be fed into a dense layer
        right_boundary_point = tf.constant([[self.right_boundary_point]], dtype=input_type) # needs shape (1, 1) so it can be fed into a dense layer

        with tf.GradientTape() as weights_update_tape:
            with tf.GradientTape(watch_accessed_variables=False) as second_der_tape:
                second_der_tape.watch(x)    # shape of x: (batch_size, 1)

                with tf.GradientTape(watch_accessed_variables=False) as first_der_tape:
                    first_der_tape.watch(x) # shape of x: (batch_size, 1)
                    prediction = self(x, training=True)   # shape: (batch_size, 1)

                # CAREFULL: derivatives wouldn't work correctly like this if I had multiple outpust, there would be some
                # modifications needed: taking batch_jacobian for first derivative and extracting the diagonal derivatives.
                # # gradients for multiple outputs would just sum the derivatives of individial outpust for fixed x
                # # (it would just act as gradient does regurarly for multiple targets, see docs: 
                # # https://www.tensorflow.org/guide/autodiff#gradients_of_non-scalar_targets)

                u_x = first_der_tape.gradient(target=prediction, sources=x)   # shape: (batch_of_points, 1)

            u_xx = second_der_tape.batch_jacobian(target=u_x, source=x) # shape: (batch_of_points, 1, 1)

            u_xx = tf.squeeze(u_xx) # shape: (batch_size)
            RHS_evaluated = tf.squeeze( self.RHS_function(x) )  # shape: (batch_size)
            equation_loss = self.equation_loss_coef * tf.math.reduce_mean( tf.math.squared_difference(u_xx, RHS_evaluated) )

            loss_dict = {
                "equation loss": equation_loss
            }
        
        self.optimizer.minimize(loss_dict, self.trainable_variables, tape=weights_update_tape)
        # grads = weights_update_tape.gradient(target=loss_dict, sources=self.trainable_variables)
        # self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss_dict


class Poisson_problem_1D_pointwise_dynamic_loss_coeficients(keras.Sequential):
    """
    .
    """
    def __init__(self, RHS_function, left_boundary_point, right_boundary_point, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.RHS_function = RHS_function
        self.equation_loss_coef = self.add_weight(initializer="ones", trainable=True)
        self.boundary_loss_coef = self.add_weight(initializer="ones", trainable=True)
        self.left_boundary_point = left_boundary_point
        self.right_boundary_point = right_boundary_point

    # @tf.function
    def train_step(self, data):
        x, y_true = data

        input_type = x.dtype
        left_boundary_point = tf.constant([[self.left_boundary_point]], dtype=input_type) # needs shape (1, 1) so it can be fed into a dense layer
        # left_boundary_point = tf.Variable(initial_value=[[self.left_boundary_point]], trainable=True, dtype=input_type)

        right_boundary_point = tf.constant([[self.right_boundary_point]], dtype=input_type) # needs shape (1, 1) so it can be fed into a dense layer
        # right_boundary_point = tf.Variable(initial_value=[[self.right_boundary_point]], trainable=True, dtype=input_type)

        with tf.GradientTape() as weights_update_tape:
            with tf.GradientTape(persistent=True) as second_der_tape:
                second_der_tape.watch(x)    # shape of x: (batch_size, 1)
                second_der_tape.watch(left_boundary_point)
                second_der_tape.watch(right_boundary_point)

                with tf.GradientTape(persistent=True) as first_der_tape:
                    first_der_tape.watch(x) # shape of x: (batch_size, 1)
                    first_der_tape.watch(left_boundary_point)
                    first_der_tape.watch(right_boundary_point)
                    prediction = self(x, training=True)   # shape: (batch_size, 1)
                    prediction_left_boundary = self(left_boundary_point, training=True) # shape: (1, 1)
                    prediction_right_boundary = self(right_boundary_point, training=True)   # shape: (1, 1)

                # CAREFULL: derivatives wouldn't work correctly like this if I had multiple outpust, there would be some
                # modifications needed: taking batch_jacobian for first derivative and extracting the diagonal derivatives.
                # # gradients for multiple outputs would just sum the derivatives of individial outpust for fixed x
                # # (it would just act as gradient does regurarly for multiple targets, see docs: 
                # # https://www.tensorflow.org/guide/autodiff#gradients_of_non-scalar_targets)
                u_x = first_der_tape.gradient(target=prediction, sources=x)   # shape: (batch_of_points, 1)
                u_x_left_boundary = first_der_tape.gradient(target=prediction_left_boundary, sources=left_boundary_point)   # shape: (1, 1)
                u_x_right_boundary = first_der_tape.gradient(target=prediction_right_boundary, sources=right_boundary_point)    # shape: (1, 1)
                del first_der_tape

            u_xx = second_der_tape.batch_jacobian(target=u_x, source=x) # shape: (batch_of_points, 1, 1)
            u_xx_left_boundary = second_der_tape.batch_jacobian(target=u_x_left_boundary, source=left_boundary_point) # shape: (1, 1, 1)
            u_xx_right_boundary = second_der_tape.batch_jacobian(target=u_x_right_boundary, source=right_boundary_point) # shape: (1, 1, 1)
            del second_der_tape

            u_xx = tf.squeeze(u_xx) # shape: (batch_size)
            RHS_evaluated = tf.squeeze( self.RHS_function(x) )  # shape: (batch_size)
            # residual = u_xx - RHS_evaluated
            # equation_loss = self.equation_loss_coef * tf.math.reduce_mean( tf.math.square(residual) )
            equation_loss = self.equation_loss_coef * tf.math.reduce_mean( tf.math.squared_difference(u_xx, RHS_evaluated) )

            u_xx_left_boundary = tf.squeeze(u_xx_left_boundary)
            RHS_at_left_boundary = tf.squeeze( self.RHS_function(left_boundary_point) )
            loss_left_boundary = tf.math.squared_difference(u_xx_left_boundary, RHS_at_left_boundary)

            u_xx_right_boundary = tf.squeeze(u_xx_right_boundary)
            RHS_at_right_boundary = tf.squeeze( self.RHS_function(right_boundary_point) )
            loss_right_boundary = tf.math.squared_difference(u_xx_right_boundary, RHS_at_right_boundary)

            boundary_loss = self.boundary_loss_coef * (loss_left_boundary + loss_right_boundary) / 2

            # loss = boundary_loss + equation_loss
        loss_dict = {
            "equation loss": equation_loss,
            "boundary loss": boundary_loss
        }
        
        # self.optimizer.minimize(loss_dict, self.trainable_variables, tape=weights_update_tape)
        grads = weights_update_tape.gradient(target=loss_dict, sources=self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss_dict



class Poisson_problem_1D_pointwise_tapeless(keras.Sequential):
    """    
    .
    """
    def __init__(self, RHS_function, left_boundary_point, right_boundary_point, *args, equation_loss_coef=1, boundary_loss_coef=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.RHS_function = RHS_function
        self.equation_loss_coef = equation_loss_coef
        self.boundary_loss_coef = boundary_loss_coef
        self.left_boundary_point = left_boundary_point
        self.right_boundary_point = right_boundary_point

    @tf.function
    def train_step(self, data):
        x, y_true = data
        left_boundary_point = tf.constant([[self.left_boundary_point]])
        right_boundary_point = tf.constant([[self.right_boundary_point]])
        
        with tf.GradientTape() as tape:
            output = self(x, training=True)

            # d2u_dx2 = tf.hessians(output, x)[0] # shape: (batch size, 1, batch size, 1)
            #     # the [0] extracts the result tensor from a list
            #     # since the result of tf.hessians is a list
            # d2u_dx2 = tf.linalg.tensor_diag_part(d2u_dx2)   # shape (batch size, 1)

            # TF.HESSIANS WAS SLOWING IT DOWN A LOT, BUT WITH MY SETTING I CAN JUST CALL TF.GRADIENTS
            # TWICE IN A ROW AND CALCULATE THE DIAGONAL (because of how the points are proccessed,
            # the non-diagonal terms are zero, so doing the derivative twice with tf.gradients yields
            # normally the second derivative, as one would expect)

            du_dx = tf.gradients(output, x)[0]
            d2u_dx2 = tf.gradients(du_dx, x)[0]

            residual = d2u_dx2 - self.RHS_function(x)   # shape: (batch size, 1)
            
            equation_loss = self.equation_loss_coef * tf.math.reduce_mean(tf.math.square(residual))


            # output_left_boundary = self(left_boundary_point, training=True) # shape: (1, 1)
            # d2u_dx2_left_boundary = tf.hessians(output_left_boundary, left_boundary_point)[0] # shape: (1, 1, 1, 1)
            # d2u_dx2_left_boundary = tf.linalg.tensor_diag_part(d2u_dx2_left_boundary) # shape: (1, 1)
            # residual_left_boundary = d2u_dx2_left_boundary - self.RHS_function(left_boundary_point) # shape: (1, 1)

            # output_right_boundary = self(right_boundary_point, training=True) # shape: (1, 1)
            # d2u_dx2_right_boundary = tf.hessians(output_right_boundary, right_boundary_point)[0] # shape: (1, 1, 1, 1)
            # d2u_dx2_right_boundary = tf.linalg.tensor_diag_part(d2u_dx2_right_boundary) # shape: (1, 1)
            # residual_right_boundary = d2u_dx2_right_boundary - self.RHS_function(right_boundary_point) # shape: (1, 1)

            # residual_boundary = tf.stack([residual_left_boundary, residual_right_boundary], axis=0)
            # boundary_loss = self.boundary_loss_coef * tf.math.reduce_mean(tf.math.square(residual_boundary))

            # total_loss = equation_loss + boundary_loss

            total_loss = equation_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # tf.print("d2u_dx2 shape")
        # tf.print(d2u_dx2.shape)
        # tf.print("residual shape")
        # tf.print(residual.shape)
        # tf.print("output left boundary shape")
        # tf.print(output_left_boundary.shape)
        # tf.print("rhs function on x shape")
        # tf.print(self.RHS_function(x).shape)


        loss_dict = {
            "equation loss": equation_loss,
            # "boundary loss": boundary_loss,
        }

        return loss_dict



# currently, my version using tf.gradient is very slow (could it possibly be due to using python lists and for loops?)
# and the version with dynamic coeficients didn't really work, since when the coeficients were set to be learned as trianable
#   parameters, they kept decreasing and decresing, causing the loss go into large negative values
class Poisson_problem_1D_pointwise_tapeless_old(keras.Sequential):
    """    
    .
    """
    def __init__(self, RHS_function, left_boundary_point, right_boundary_point, *args, equation_loss_coef=1, boundary_loss_coef=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.RHS_function = RHS_function
        self.equation_loss_coef = equation_loss_coef
        self.boundary_loss_coef = boundary_loss_coef
        self.left_boundary_point = left_boundary_point
        self.right_boundary_point = right_boundary_point

    @tf.function
    def train_step(self, data):
        x, y_true = data

        # x has shape (batch size, 1)
        x_list = tf.unstack(x, axis=0)
            # list with "batch size" number of shape (1,) tensors
        pred_list = [self(x, training=True) for x in x_list]
            # list with "batch size" number of shape (1, 1) tensors

        # Just testing if the for loop is converted into a tensorflow for loop
        # pred_list = []
        # for x in x_list:
        #     pred_list.append(self(x, training=True))
        #     print("normal python for loop")

        d2u_dx2_list = [tf.hessians(result, point)[0] for result, point in zip(pred_list, x_list)]
            # Returns a list of shape (1,1) tensors containing the second derivatives of output wrt input for each
            #   point in the batch.
            # tf.hessians returns a list of tensors with same shape as the "xs", so "point" in this case. However "point"
            #   is a tensor of shape (1,1) (it gets converted to this from shape (1,) probably so that it can be passed
            #   into a dense layer), so
            #   the return type of tf.hessians in this case a list with single tensor of shape (1,1). The "[0]" used on
            #   tf.hessians is used to extract the tensor, so in the end d2u_dx2_list is not a list of lists but a list
            #   of tensors.
        # tf.print(d2u_dx2_list[0])
        # tf.print(d2u_dx2_list[0].shape)
        # tf.print(type(d2u_dx2_list[0]))
        equation_residual_list = [tf.squeeze(second_derivative, axis=1) - self.RHS_function(point) for second_derivative, point in zip(d2u_dx2_list, x_list)]
            # list of shape (1,) tensors
            # the squeeze is there in order to make the second derivative into shape (1,) tensor from shape (1,1), so substraction can be
            #   performed between the 2 tensors, since RHS_funcion output is shape (1,)
        # tf.print(equation_residual_list[0])
        # tf.print(type(equation_residual_list[0]))
        equation_residual = tf.stack(equation_residual_list)
            # turns the list into a shape (batch size, 1) tensor
        # tf.print(equation_residual.shape)
        equation_residual_squared = tf.math.square(equation_residual)
        equation_residual_mean = tf.math.reduce_mean(equation_residual_squared)
        # equation_loss = self.equation_loss_coef * tf.math.reduce_mean(tf.math.square(equation_residual))
        equation_loss = self.equation_loss_coef * equation_residual_mean
            # equation loss is the mean square of the residual
        # tf.print(equation_loss.shape)
        # tf.print(equation_residual)
        # tf.print(equation_residual_squared)
        # tf.print(equation_residual_mean)
        # tf.print(self.equation_loss_coef)

        left_boundary_point = tf.constant([[self.left_boundary_point]])
        pred_left_boundary = self(left_boundary_point, training=True)
        # tf.print(pred_left_boundary)
        d2u_dx2_left_boundary = tf.hessians(pred_left_boundary, left_boundary_point)[0]
            # shape (1,1) tensor
            # tf.hessians returns a list with single shape (1,1) tensor, the "[0]" extracts that tensor
        left_boundary_residual = d2u_dx2_left_boundary - self.RHS_function(left_boundary_point)
            # shape (1,1) tensor

        right_boundary_point = tf.constant([[self.right_boundary_point]])
        pred_right_boundary = self(right_boundary_point, training=True)
        d2u_dx2_right_boundary = tf.hessians(pred_right_boundary, right_boundary_point)[0]
            # shape (1,1) tensor
            # tf.hessians returns a list with single shape (1,1) tensor, the "[0]" extracts that tensor
        right_boundary_residual = d2u_dx2_right_boundary - self.RHS_function(right_boundary_point)
            # shape (1,1) tensor

        mean_square_boundary_residual = (right_boundary_residual**2 + left_boundary_residual**2) / 2
        boundary_loss = self.boundary_loss_coef * tf.squeeze(mean_square_boundary_residual)

        total_loss = equation_loss + boundary_loss
        grads = tf.gradients(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # tf.print(equation_loss.shape)
        # tf.print(boundary_loss.shape)

        return {
            "equation loss": equation_loss,
            "boundary loss": boundary_loss
        }   


# @tf.function
def mean_squared_residual_1D(input, y_true, y_pred, number_of_grid_points, step_size, boundary_value):
    d2u_dx2 = tf.zeros_like(input=input)
    last_grid_index = number_of_grid_points - 1

    number_of_samples_in_batch = input.shape[0]
    for input_sample_index in range(0, number_of_samples_in_batch):
        total_residual = 0
        # # # MSE (mean squared error) na vnitřku intervalu, pro fixni input pravou stranu f:
        for xi in range(1, last_grid_index):
            y_neg_1 =   y_pred[input_sample_index, xi - 1, 0]
            y_0 =       y_pred[input_sample_index, xi,     0]
            y_1 =       y_pred[input_sample_index, xi + 1, 0]
            update_value = ( y_1 - 2*y_0 + y_neg_1 ) / ( step_size ** 2 )
            update_value_tensor = tf.reshape(tensor=update_value, shape=(1,))

            indicies = tf.constant([ [input_sample_index, xi, 0] ])
            d2u_dx2 = tf.tensor_scatter_nd_add(tensor=d2u_dx2, indices=indicies, updates=update_value_tensor)
        # x = d2u_dx2 - input
        # squared_residual_interior = tf.reshape(shape=[-1], tensor=tf.math.sqrt(tf.math.abs(x=x)))
        squared_residual_interior = tf.reshape(shape=[-1], tensor=tf.math.squared_difference(x=d2u_dx2, y=input))
        mean_squared_residual_interior = tf.math.reduce_mean(squared_residual_interior)

        # squared_residual_boundary = tf.math.sqrt(tf.math.abs((input[input_sample_index][0][0] - boundary_value)) + tf.math.abs((input[input_sample_index][last_grid_index][0] - boundary_value)))
        squared_residual_boundary = (input[input_sample_index, 0, 0] - boundary_value)**2 + (input[input_sample_index, last_grid_index, 0] - boundary_value)**2
        mean_squared_residual_boundary = squared_residual_boundary / 2

        total_residual += mean_squared_residual_boundary + mean_squared_residual_interior

    return total_residual
