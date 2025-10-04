import tensorflow as tf
import keras
import sys
from ..utils import *
# Můžeš ignorovat utils, to jsem měl jěnom nějaké výpisy vektorů při debugování. Kdybys náhodou dostal error a našel nějakou funkci podobnou print,
# tak to stačí zakomentovat a mělo by to fungovat.
import time

class HE_1D_PINN_custom_fit(keras.Sequential):
    """
    Class implementing the heat equation u_t - u_xx = RHS_function with Dirichlet boundary conditions and
    initial condition given by init_cond_function.

    Args:
    RHS_function:
        - a function made with tensorflow operations
        - input (x, t), where x and t are tensors with points in which the right hand side is to be evaluated
        - returns the right hand side of the equation

    init_cond_function:
        - a function made with tensorflow operations
        - input (x, t), where x and t are tensors with points in which the right hand side is to be evaluated
        - returns the initial condition
    .
    """
    def __init__(
        self,
        RHS_function,
        init_cond_function,
        spacial_start_point,
        spacial_end_point,
        num_of_spacial_points_for_equation,
        num_of_spacial_points_for_init_cond,
        time_start_point,
        time_end_point,
        num_of_temporal_points_for_equation,
        *args,
        equation_loss_coef=1.0,
        boundary_loss_coef=1.0,
        init_cond_loss_coef=1.0,
        left_boundary_value=0.0,
        right_boundary_value=0.0,
        heat_coeficient=1.0,
        **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.RHS_function = RHS_function
        self.init_cond_function = init_cond_function

        self.spacial_start_point = spacial_start_point
        self.spacial_end_point = spacial_end_point
        self.num_of_spacial_points_for_equation = num_of_spacial_points_for_equation
        self.num_of_spacial_points_for_init_cond = num_of_spacial_points_for_init_cond

        self.time_start_point = time_start_point
        self.time_end_point = time_end_point
        self.num_of_temporal_points_for_equation = num_of_temporal_points_for_equation

        self.equation_loss_coef = equation_loss_coef
        self.boundary_loss_coef = boundary_loss_coef
        self.init_cond_loss_coef = init_cond_loss_coef
        self.heat_coeficient = heat_coeficient

        self.left_boundary_value = left_boundary_value
        self.right_boundary_value = right_boundary_value


    def _print_epoch_results(self, epoch_index, logs):
        tf.print(f"Epoch {epoch_index}:")
        for key, value in logs.items():
            tf.print(f"{key}: {value}", end=", ")
        # for metric in self.metrics:
        #     tf.print(f"{metric.result()}", end=", ")
        tf.print("")


    def fit(
        self,
        epochs,
        write_every=None,
        callbacks=None,
        initial_epoch=0,
        run_as_graph=True,
        jit_compile=True,
    ):
        """
        Keras fit, just do not pass in any x, y and batch_size arguments, they will be generated.
        Pass the rest of the arguments as keyword arguments.
        .
        """
        x = tf.linspace(self.spacial_start_point, self.spacial_end_point, self.num_of_spacial_points_for_equation)
        t = tf.linspace(self.time_start_point, self.time_end_point, self.num_of_temporal_points_for_equation)

        X, T = tf.meshgrid(x[1:-1], t[1:])
        # X, T = tf.meshgrid(x, t)
        x_eq = tf.reshape(X, [-1])
        t_eq = tf.reshape(T, [-1])
        
        x_lb = tf.constant(x[0], shape=t.shape)
        x_rb = tf.constant(x[-1], shape=t.shape)
        lb_target_value = tf.constant(self.left_boundary_value, shape=x_lb.shape)
        rb_target_value = tf.constant(self.right_boundary_value, shape=x_rb.shape)
        x_boundary = tf.concat([x_lb, x_rb], axis=0)
        t_boundary = tf.concat([t, t], axis=0)
        boundary_target_value = tf.concat([lb_target_value, rb_target_value], axis=0)

        x_ini = tf.linspace(self.spacial_start_point, self.spacial_end_point, self.num_of_spacial_points_for_init_cond)
        t_ini = tf.constant(self.time_start_point, shape=x_ini.shape)

        # tf.print("x_eq size: ")
        # tf.print(tf.size(x_eq).numpy())

        # Dataset is needed in order to be able to pass multiple tensors (as a tuple) into the keras fit function
        dataset = tf.data.Dataset.from_tensors( (x_eq, t_eq, x_boundary, t_boundary, boundary_target_value, x_ini, t_ini) )
        del x, t, X, T, x_lb, x_rb, lb_target_value, rb_target_value
        
        
        if run_as_graph:
            print("tracing train step")
            self.train_step = tf.function(self.train_step,jit_compile=jit_compile,reduce_retracing=True)

        loss_history = dict()
        if callbacks is not None:
            for callback in callbacks:
                callback.on_train_begin()

        train_start_time = time.time()
        for epoch_index in range(initial_epoch, epochs):

            if callbacks is not None:
                for callback in callbacks:
                    callback.on_epoch_begin()

            with tf.profiler.experimental.Trace("train_step", step_num=epoch_index, _r=1): # This enables the use of profiling and Tensorboard.
                if callbacks is not None:
                    for callback in callbacks:
                        callback.on_train_batch_begin(epoch_index)

                loss_dict = self.train_step((x_eq, t_eq, x_boundary, t_boundary, boundary_target_value, x_ini, t_ini))

                if callbacks is not None:
                    for callback in callbacks:
                        callback.on_train_batch_end(epoch_index, loss_dict)


            if callbacks is not None:
                for callback in callbacks:
                    callback.on_epoch_end(epoch_index, loss_dict)

            for loss_name, loss_value in loss_dict.items():
                if loss_name not in loss_history:
                    loss_history[loss_name] = []
                loss_history[loss_name].append(loss_value)

            if epoch_index == 0:
                self._print_epoch_results(0, loss_dict)

            elif epoch_index == epochs:
                self._print_epoch_results(epochs, loss_dict)

            elif epoch_index % write_every == 0:
                self._print_epoch_results(epoch_index, loss_dict)

        train_end_time = time.time()
        tf.print(f"\nTraining took {train_end_time - train_start_time}s.")

        if callbacks is not None:
            for callback in callbacks:
                callback.on_train_end()

        return loss_history


    def train_step(self, data):
        print("Tracing train step.")
        x_eq, t_eq, x_boundary, t_boundary, boundary_target_value, x_ini, t_ini = data

        with tf.GradientTape() as weights_update_tape:
            with tf.GradientTape(watch_accessed_variables=False, persistent=True) as derivation_tape:
                derivation_tape.watch(t_eq)
                derivation_tape.watch(x_eq)
                tx_eq = tf.stack([t_eq, x_eq], axis=1)
                u_pred = self(tx_eq, training=True)

                u_x = derivation_tape.gradient(u_pred, x_eq)
            u_xx = derivation_tape.gradient(u_x, x_eq)
            u_t = derivation_tape.gradient(u_pred, t_eq)
            del derivation_tape

            # tf.print(tx_eq.shape)
            # tf.print(u_pred.shape)
            # tf.print(u_x.shape)
            # tf.print(u_xx.shape)
            # tf.print(u_t.shape)

            # print_tensor(tf.reshape(u_x, shape=[-1]))
            # print_tensor(tf.reshape(u_xx, shape=[-1]))
            # print_tensor(tf.reshape(u_t, shape=[-1]))

            RHS = self.RHS_function(t_eq, x_eq, u_pred, self.heat_coeficient)
            equation_residual = u_t - self.heat_coeficient * u_xx - RHS
            weighted_equation_loss = self.equation_loss_coef * self.compute_loss(tx_eq, tf.zeros_like(equation_residual), equation_residual)
            del tx_eq


            # if tf.rank(RHS) == 1 and RHS.shape[0] == xt.shape[0]:
            #     RHS = RHS[:, tf.newaxis]
            #     # just a check, in case the RHS_function returs a shape (num of points,) tensor
            #     # instead of shape (num of points, 1) tensor for some reason


            tx_ini = tf.stack([t_ini, x_ini], axis=-1)
            u_ini = self(tx_ini, training=True)
            u_ini_target = self.init_cond_function(x_ini, u_pred, self.heat_coeficient)
            weighted_init_cond_loss = self.init_cond_loss_coef * self.compute_loss(tx_ini, u_ini_target, u_ini)
            del tx_ini


            tx_boundary = tf.stack([t_boundary, x_boundary], axis=1)
            u_boundary = self(tx_boundary, training=True)
            weighted_boundary_loss = self.boundary_loss_coef * self.compute_loss(tx_boundary, boundary_target_value, u_boundary)
            del tx_boundary

            loss_dict = {
                "weighted equation loss": weighted_equation_loss,
                "weighted boundary condition loss": weighted_boundary_loss,
                "weighted initial condition loss": weighted_init_cond_loss,
            }

        grads = weights_update_tape.gradient(loss_dict, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss_dict



class HE_1D_PINN(keras.Sequential):
    """
    Class implementing the heat equation u_t - u_xx = RHS_function with Dirichlet boundary conditions and
    initial condition given by init_cond_function.

    Args:
    RHS_function:
        - a function made with tensorflow operations
        - input (x, t), where x and t are tensors with points in which the right hand side is to be evaluated
        - returns the right hand side of the equation

    init_cond_function:
        - a function made with tensorflow operations
        - input (x, t), where x and t are tensors with points in which the right hand side is to be evaluated
        - returns the initial condition
    .
    """
    def __init__(
        self,
        RHS_function,
        init_cond_function,
        spacial_start_point,
        spacial_end_point,
        num_of_spacial_points_for_equation,
        num_of_spacial_points_for_init_cond,
        time_start_point,
        time_end_point,
        num_of_temporal_points_for_equation,
        *args,
        equation_loss_coef=1.0,
        boundary_loss_coef=1.0,
        init_cond_loss_coef=1.0,
        left_boundary_value=0.0,
        right_boundary_value=0.0,
        heat_coeficient=1.0,
        **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.RHS_function = RHS_function
        self.init_cond_function = init_cond_function

        self.spacial_start_point = spacial_start_point
        self.spacial_end_point = spacial_end_point
        self.num_of_spacial_points_for_equation = num_of_spacial_points_for_equation
        self.num_of_spacial_points_for_init_cond = num_of_spacial_points_for_init_cond

        self.time_start_point = time_start_point
        self.time_end_point = time_end_point
        self.num_of_temporal_points_for_equation = num_of_temporal_points_for_equation

        self.equation_loss_coef = equation_loss_coef
        self.boundary_loss_coef = boundary_loss_coef
        self.init_cond_loss_coef = init_cond_loss_coef
        self.heat_coeficient = heat_coeficient

        self.left_boundary_value = left_boundary_value
        self.right_boundary_value = right_boundary_value


    def fit(self, **kwargs):
        """
        Keras fit, just do not pass in any x, y and batch_size arguments, they will be generated.
        Pass the rest of the arguments as keyword arguments.
        .
        """
        x = tf.linspace(self.spacial_start_point, self.spacial_end_point, self.num_of_spacial_points_for_equation)
        t = tf.linspace(self.time_start_point, self.time_end_point, self.num_of_temporal_points_for_equation)

        X, T = tf.meshgrid(x[1:-1], t[1:])
        # X, T = tf.meshgrid(x, t)
        x_eq = tf.reshape(X, [-1])
        t_eq = tf.reshape(T, [-1])
        
        x_lb = tf.constant(x[0], shape=t.shape)
        x_rb = tf.constant(x[-1], shape=t.shape)
        lb_target_value = tf.constant(self.left_boundary_value, shape=x_lb.shape)
        rb_target_value = tf.constant(self.right_boundary_value, shape=x_rb.shape)
        x_boundary = tf.concat([x_lb, x_rb], axis=0)
        t_boundary = tf.concat([t, t], axis=0)
        boundary_target_value = tf.concat([lb_target_value, rb_target_value], axis=0)

        x_ini = tf.linspace(self.spacial_start_point, self.spacial_end_point, self.num_of_spacial_points_for_init_cond)
        t_ini = tf.constant(self.time_start_point, shape=x_ini.shape)

        tf.print("x_eq size: ")
        tf.print(tf.size(x_eq).numpy())

        # Dataset is needed in order to be able to pass multiple tensors (as a tuple) into the keras fit function
        dataset = tf.data.Dataset.from_tensors( (x_eq, t_eq, x_boundary, t_boundary, boundary_target_value, x_ini, t_ini) )
        del x, t, X, T, x_lb, x_rb, lb_target_value, rb_target_value
        
        # tf.print(x_eq.shape)
        # tf.print(t_eq.shape)
        # tf.print(x_boundary.shape)
        # tf.print(t_boundary.shape)
        # tf.print(boundary_target_value.shape)
        # tf.print(x_ini.shape)
        # tf.print(t_ini.shape)

        # TODO: asi udělat vlastní training loop, aby se daly input body batchovat

        return super().fit(dataset, **kwargs)

    def train_step(self, data):
        print("Tracing train step.")
        x_eq, t_eq, x_boundary, t_boundary, boundary_target_value, x_ini, t_ini = data

        with tf.GradientTape() as weights_update_tape:
            with tf.GradientTape(watch_accessed_variables=False, persistent=True) as derivation_tape:
                derivation_tape.watch(t_eq)
                derivation_tape.watch(x_eq)
                tx_eq = tf.stack([t_eq, x_eq], axis=1)
                u_pred = self(tx_eq, training=True)

                u_x = derivation_tape.gradient(u_pred, x_eq)
            u_xx = derivation_tape.gradient(u_x, x_eq)
            u_t = derivation_tape.gradient(u_pred, t_eq)
            del derivation_tape

            # tf.print(tx_eq.shape)
            # tf.print(u_pred.shape)
            # tf.print(u_x.shape)
            # tf.print(u_xx.shape)
            # tf.print(u_t.shape)

            # print_tensor(tf.reshape(u_x, shape=[-1]))
            # print_tensor(tf.reshape(u_xx, shape=[-1]))
            # print_tensor(tf.reshape(u_t, shape=[-1]))

            RHS = self.RHS_function(t_eq, x_eq, u_pred, self.heat_coeficient)
            equation_residual = u_t - self.heat_coeficient * u_xx - RHS
            weighted_equation_loss = self.equation_loss_coef * self.compute_loss(tx_eq, tf.zeros_like(equation_residual), equation_residual)
            del tx_eq


            # if tf.rank(RHS) == 1 and RHS.shape[0] == xt.shape[0]:
            #     RHS = RHS[:, tf.newaxis]
            #     # just a check, in case the RHS_function returs a shape (num of points,) tensor
            #     # instead of shape (num of points, 1) tensor for some reason


            tx_ini = tf.stack([t_ini, x_ini], axis=-1)
            u_ini = self(tx_ini, training=True)
            u_ini_target = self.init_cond_function(x_ini, u_pred, self.heat_coeficient)
            weighted_init_cond_loss = self.init_cond_loss_coef * self.compute_loss(tx_ini, u_ini_target, u_ini)
            del tx_ini


            tx_boundary = tf.stack([t_boundary, x_boundary], axis=1)
            u_boundary = self(tx_boundary, training=True)
            weighted_boundary_loss = self.boundary_loss_coef * self.compute_loss(tx_boundary, boundary_target_value, u_boundary)
            del tx_boundary

            loss_dict = {
                "weighted equation loss": weighted_equation_loss,
                "weighted boundary condition loss": weighted_boundary_loss,
                "weighted initial condition loss": weighted_init_cond_loss,
            }

        grads = weights_update_tape.gradient(loss_dict, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss_dict



class HE_1D_PINN_old(keras.Sequential):
    """
    Class implementing the heat equation u_t - u_xx = RHS_function with Dirichlet boundary conditions and
    initial condition given by init_cond_function.

    Args:
    RHS_function:
        - a function made with tensorflow operations
        - input (x, t), where x and t are tensors with points in which the right hand side is to be evaluated
        - returns the right hand side of the equation

    init_cond_function:
        - a function made with tensorflow operations
        - input (x, t), where x and t are tensors with points in which the right hand side is to be evaluated
        - returns the initial condition
    .
    """
    def __init__(
        self,
        RHS_function,
        init_cond_function,
        spacial_start_point,
        spacial_end_point,
        num_of_spacial_points_for_equation,
        num_of_spacial_points_for_init_cond,
        time_start_point,
        time_end_point,
        num_of_temporal_points_for_equation,
        *args,
        equation_loss_coef=1.0,
        boundary_loss_coef=1.0,
        init_cond_loss_coef=1.0,
        left_boundary_value=0.0,
        right_boundary_value=0.0,
        heat_coeficient=1.0,
        **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.RHS_function = RHS_function
        self.init_cond_function = init_cond_function

        self.spacial_start_point = spacial_start_point
        self.spacial_end_point = spacial_end_point
        self.num_of_spacial_points_for_equation = num_of_spacial_points_for_equation
        self.num_of_spacial_points_for_init_cond = num_of_spacial_points_for_init_cond

        self.time_start_point = time_start_point
        self.time_end_point = time_end_point
        self.num_of_temporal_points_for_equation = num_of_temporal_points_for_equation

        self.equation_loss_coef = equation_loss_coef
        self.boundary_loss_coef = boundary_loss_coef
        self.init_cond_loss_coef = init_cond_loss_coef
        self.heat_coeficient = heat_coeficient

        self.left_boundary_value = left_boundary_value
        self.right_boundary_value = right_boundary_value


    def fit(self, **kwargs):
        """
        Keras fit, just do not pass in any x, y and batch_size arguments, they will be generated.
        Pass the rest of the arguments as keyword arguments.
        .
        """
        dtype = tf.float64
        x = tf.linspace(self.spacial_start_point, self.spacial_end_point, self.num_of_spacial_points_for_equation)
        t = tf.linspace(self.time_start_point, self.time_end_point, self.num_of_temporal_points_for_equation)
        x_ini = tf.linspace(self.spacial_start_point, self.spacial_end_point, self.num_of_spacial_points_for_init_cond)
        t_ini = tf.constant(self.time_start_point, shape=self.num_of_spacial_points_for_init_cond)

        # Dataset is needed in order to be able to pass 4 tensors (as a tuple) into the keras fit function
        dataset = tf.data.Dataset.from_tensors( (x, t, x_ini, t_ini) )

        # TODO: asi udělat vlastní training loop, aby se daly input body batchovat

        return super().fit(dataset, **kwargs)

    def train_step(self, data):
        x, t, x_ini, t_ini = data
        
        X, T = tf.meshgrid(x, t)
        x_eq = tf.reshape(X, (-1, 1))
        t_eq = tf.reshape(T, (-1, 1))
        del X, T
            # delete these temporary matricies to free up memory

        xt_ini = tf.stack([x_ini, t_ini], axis=-1)

        # tf.print("#########################################")
        # print_tensor(xt_ini)
        # # xt_ini seems correct

        # creating boundary points
        num_of_time_points = t.shape[0]

        x_lb = tf.constant(self.spacial_start_point, shape=(num_of_time_points,))
        left_boundary = tf.stack([x_lb, t], axis=-1) # shape (num of boundary points, 1)

        x_rb = tf.constant(self.spacial_end_point, shape=(num_of_time_points,))
        right_boundary = tf.stack([x_rb, t], axis=-1) # shape (num of boundary points, 1)

        left_boundary_target_value = tf.constant(self.left_boundary_value, shape=left_boundary.shape)
        right_boundary_target_value = tf.constant(self.right_boundary_value, shape=right_boundary.shape)

        # tf.print("#########################################")
        # print_tensor(left_boundary)
        # print_tensor(right_boundary)
        # print_tensor(left_boundary_target_value)
        # print_tensor(right_boundary_target_value)
        # # Boundary input and target values seems correct

        with tf.GradientTape() as tape:
            with tf.GradientTape(persistent=True) as der_tape:
                der_tape.watch(x_eq)
                der_tape.watch(t_eq)
                xt_eq = tf.concat([x_eq, t_eq], axis=-1)
                u_pred = self(xt_eq, training=True) # shape (num of points, 2)

                u_x = der_tape.gradient(u_pred, x_eq)
            u_xx = der_tape.gradient(u_x, x_eq)
            u_t = der_tape.gradient(u_pred, t_eq)

            # tf.print(xt_eq.shape)
            # tf.print(u_pred.shape)
            # tf.print(u_x.shape)
            # tf.print(u_xx.shape)
            # tf.print(u_t.shape)

            # print_tensor(tf.reshape(u_x, shape=[-1]))
            # print_tensor(tf.reshape(u_xx, shape=[-1]))
            # print_tensor(tf.reshape(u_t, shape=[-1]))

            RHS = self.RHS_function(x_eq, t_eq)

            # if tf.rank(RHS) == 1 and RHS.shape[0] == xt.shape[0]:
            #     RHS = RHS[:, tf.newaxis]
            #     # just a check, in case the RHS_function returs a shape (num of points,) tensor
            #     # instead of shape (num of points, 1) tensor for some reason


            equation_residual = u_t - self.heat_coeficient * u_xx - RHS
            equation_loss = self.equation_loss_coef * self.compute_loss(xt_eq, tf.zeros_like(equation_residual), equation_residual)


            u_ini = self(xt_ini, training=True)
            u_ini_target = self.init_cond_function(xt_ini[:, 0:1], xt_ini[:, 1:2])

            init_cond_loss = self.init_cond_loss_coef * self.compute_loss(xt_ini, u_ini_target, u_ini)


            error_left_boundary = self(left_boundary, training=True) - left_boundary_target_value
            error_right_boundary = self(right_boundary, training=True) - right_boundary_target_value
                # shapes (num of time points, 1)
            error_boundary = tf.concat([error_left_boundary, error_right_boundary], axis=-1) # shape (num of time points, 2)

            boundary_loss = self.boundary_loss_coef * self.compute_loss(None, tf.zeros_like(error_boundary), error_boundary)

            # print_tensor(error_boundary)

            total_loss = equation_loss + init_cond_loss + boundary_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        loss_dict = {
            "equation loss": equation_loss,
            "initial condition loss": init_cond_loss,
            "boundary loss": boundary_loss,
        }

        del der_tape

        return loss_dict



class HE_1D_PINN_tapeless(keras.Sequential):
    """
    Class implementing the heat equation u_t - u_xx = RHS_function with Dirichlet boundary conditions and
    initial condition given by init_cond_function.

    Args:
    RHS_function:
        - a function made with tensorflow operations
        - input (x, t), where x and t are tensors with points in which the right hand side is to be evaluated
        - returns the right hand side of the equation

    init_cond_function:
        - a function made with tensorflow operations
        - input (x, t), where x and t are tensors with points in which the right hand side is to be evaluated
        - returns the initial condition
    .
    """
    def __init__(
        self,
        RHS_function,
        init_cond_function,
        spacial_start_point,
        spacial_end_point,
        num_of_spacial_points_for_equation,
        num_of_spacial_points_for_init_cond,
        time_start_point,
        time_end_point,
        num_of_temporal_points_for_equation,
        *args,
        equation_loss_coef=1.0,
        boundary_loss_coef=1.0,
        init_cond_loss_coef=1.0,
        left_boundary_value=0.0,
        right_boundary_value=0.0,
        # TODO: přidat heat coeficient i sem do téhle varianty
        **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.RHS_function = RHS_function
        self.init_cond_function = init_cond_function

        self.spacial_start_point = spacial_start_point
        self.spacial_end_point = spacial_end_point
        self.num_of_spacial_points_for_equation = num_of_spacial_points_for_equation
        self.num_of_spacial_points_for_init_cond = num_of_spacial_points_for_init_cond

        self.time_start_point = time_start_point
        self.time_end_point = time_end_point
        self.num_of_temporal_points_for_equation = num_of_temporal_points_for_equation

        self.equation_loss_coef = equation_loss_coef
        self.boundary_loss_coef = boundary_loss_coef
        self.init_cond_loss_coef = init_cond_loss_coef

        self.left_boundary_value = left_boundary_value
        self.right_boundary_value = right_boundary_value


    def fit(self, **kwargs):
        """
        Keras fit, just do not pass in any x, y and batch_size arguments, they will be generated.
        Pass the rest of the arguments as positional arguments.
        .
        """
        dtype = tf.float64
        x = tf.linspace(self.spacial_start_point, self.spacial_end_point, self.num_of_spacial_points_for_equation)
        t = tf.linspace(self.time_start_point, self.time_end_point, self.num_of_temporal_points_for_equation)
        x_ini = tf.linspace(self.spacial_start_point, self.spacial_end_point, self.num_of_spacial_points_for_init_cond)
        t_ini = tf.constant(self.time_start_point, shape=self.num_of_spacial_points_for_init_cond)

        # Dataset is needed in order to be able to pass 4 tensors (as a tuple) into the keras fit function
        dataset = tf.data.Dataset.from_tensors( (x, t, x_ini, t_ini) )

        # TODO: asi udělat vlastní training loop, aby se daly input body batchovat

        return super().fit(dataset, **kwargs)

    def train_step(self, data):
        x, t, x_ini, t_ini = data
        
        # TODO: U tapeless varianty zkusit tohle přesunout do fit, abych ten vektor vytvořil jednou na začátku a pak
        # ho předal sem, než abych ho tu vytvářel vždycky znovu. Mohlo by to o něco zrychlit to trénování.
        X, T = tf.meshgrid(x, t)
        x_eq = tf.reshape(X, (-1, 1))
        t_eq = tf.reshape(T, (-1, 1))
        del X, T
            # delete these temporary matricies to free up memory

        xt_eq = tf.concat([x_eq, t_eq], axis=-1)

        xt_ini = tf.stack([x_ini, t_ini], axis=-1)

        # creating boundary points
        num_of_time_points = t.shape[0]

        x_lb = tf.constant(self.spacial_start_point, shape=(num_of_time_points,))
        left_boundary = tf.stack([x_lb, t], axis=-1) # shape (num of boundary points, 2)

        x_rb = tf.constant(self.spacial_end_point, shape=(num_of_time_points,))
        right_boundary = tf.stack([x_rb, t], axis=-1) # shape (num of boundary points, 2)

        left_boundary_target_value = tf.constant(self.left_boundary_value, shape=left_boundary.shape)
        right_boundary_target_value = tf.constant(self.right_boundary_value, shape=right_boundary.shape)

        with tf.GradientTape() as tape:
            u_pred = self(xt_eq, training=True) # shape (num of points, 2)

            # TODO: změřit operace přímo zde pomocí něčeho, pak je změřit v deepxde a porovnat, jestli mi to počítá
            #   derivace stejně rychle, nebo ne
            u_x = tf.gradients(u_pred, x_eq)[0]
            u_xx = tf.gradients(u_x, x_eq)[0]
            u_t = tf.gradients(u_pred, t_eq)[0]

            RHS = self.RHS_function(x_eq, t_eq)

            # if tf.rank(RHS) == 1 and RHS.shape[0] == xt.shape[0]:
            #     RHS = RHS[:, tf.newaxis]
            #     # just a check, in case the RHS_function returs a shape (num of points,) tensor
            #     # instead of shape (num of points, 1) tensor for some reason


            equation_residual = u_t - u_xx - RHS
            equation_loss = self.equation_loss_coef * self.compute_loss(xt_eq, tf.zeros_like(equation_residual), equation_residual)


            u_ini = self(xt_ini, training=True)
            u_ini_target = self.init_cond_function(xt_ini[:, 0], xt_ini[:, 1])

            init_cond_loss = self.init_cond_loss_coef * self.compute_loss(xt_ini, u_ini_target, u_ini)


            error_left_boundary = self(left_boundary, training=True) - left_boundary_target_value
            error_right_boundary = self(right_boundary, training=True) - right_boundary_target_value
                # shapes (num of time points, 1)
            error_boundary = tf.concat([error_left_boundary, error_right_boundary], axis=-1) # shape (num of time points, 2)

            boundary_loss = self.boundary_loss_coef * self.compute_loss(None, tf.zeros_like(error_boundary), error_boundary)


            total_loss = equation_loss + init_cond_loss + boundary_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        loss_dict = {
            "equation loss": equation_loss,
            "initial condition loss": init_cond_loss,
            "boundary loss": boundary_loss,
        }

        # string = f"x_eq shape: {x_eq.shape}"
        # tf.print(string)
        # string = f"t_eq shape: {t_eq.shape}"
        # tf.print(string)
        # string = f"xt_eq shape: {xt_eq.shape}"
        # tf.print(string)
        # string = f"xt_ini shape: {xt_ini.shape}"
        # tf.print(string)
        # string = f"left_boundary shape: {left_boundary.shape}"
        # tf.print(string)
        # string = f"right_boundary shape: {right_boundary.shape}"
        # tf.print(string)
        # string = f"u_pred shape: {u_pred.shape}"
        # tf.print(string)
        # string = f"u_x shape: {u_x.shape}"
        # tf.print(string)
        # string = f"u_xx shape: {u_xx.shape}"
        # tf.print(string)
        # string = f"u_t shape: {u_t.shape}"
        # tf.print(string)
        # tf.print("\n\n\n")

        return loss_dict
