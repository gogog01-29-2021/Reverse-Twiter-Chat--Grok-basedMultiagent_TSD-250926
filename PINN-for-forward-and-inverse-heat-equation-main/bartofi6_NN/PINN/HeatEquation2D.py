import tensorflow as tf
import keras
import time


class HE_2D_PINN(keras.Sequential):
    """
    Class implementing the heat equation u_t - u_xx - u_yy = RHS_function with Dirichlet boundary conditions and
    initial condition given by init_cond_function.

    Args:
    RHS_function:
        - a function made with tensorflow operations
        - input (x, y, t), where x, y and t are tensors with points in which the right hand side is to be evaluated
        - returns the right hand side of the equation

    init_cond_function:
        - a function made with tensorflow operations
        - input (x, y, t), where x, y and t are tensors with points in which the right hand side is to be evaluated
        - returns the initial condition
    .
    """
    def __init__(
        self,
        RHS_function,
        init_cond_function,
        boun_cond_funtion,
        x_start_point,
        x_end_point,
        num_of_x_points,
        # num_of_x_points_for_equation,
        # num_of_x_points_for_bound_cond,
        # num_of_x_points_for_init_cond,
        y_start_point,
        y_end_point,
        num_of_y_points,
        # num_of_y_points_for_equation,
        # num_of_y_points_for_bound_cond,
        # num_of_y_points_for_init_cond,
        t_start_point,
        t_end_point,
        num_of_t_points,
        # num_of_t_points_for_equation,
        # num_of_t_points_for_bound_cond,
        *args,
        equation_loss_coef=1.0,
        boundary_loss_coef=1.0,
        init_cond_loss_coef=1.0,
        heat_coeficient=1.0,
        **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.RHS_function = RHS_function
        self.init_cond_function = init_cond_function
        self.bound_cond_function = boun_cond_funtion

        self.x_start_point = x_start_point
        self.x_end_point = x_end_point
        self.num_of_x_points = num_of_x_points
        # self.num_of_x_points_for_equation = num_of_x_points_for_equation
        # self.num_of_x_points_for_init_cond = num_of_x_points_for_init_cond

        self.y_start_point = y_start_point
        self.y_end_point = y_end_point
        self.num_of_y_points = num_of_y_points
        # self.num_of_y_points_for_equation = num_of_y_points_for_equation
        # self.num_of_y_points_for_init_cond = num_of_y_points_for_init_cond
        self.num_of_t_points = num_of_t_points
        self.t_start_point = t_start_point
        self.t_end_point = t_end_point

        # self.num_of_t_points_for_equation = num_of_t_points_for_equation

        self.equation_loss_coef = equation_loss_coef
        self.boundary_loss_coef = boundary_loss_coef
        self.init_cond_loss_coef = init_cond_loss_coef
        self.heat_coeficient = heat_coeficient


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
        .
        """
        x = tf.linspace(self.x_start_point, self.x_end_point, self.num_of_x_points)
        y = tf.linspace(self.y_start_point, self.y_end_point, self.num_of_y_points)
        t = tf.linspace(self.t_start_point, self.t_end_point, self.num_of_t_points)

        X, Y, T = tf.meshgrid(x, y, t, indexing='ij')
        # Indexování 'ij' zařídí, že bod na indexu [i,j,k] odpovídá bodu s hodnotami x[i], y[j], t[k].
        # Tedy prostě standardní kartészké indexování: z hodnot x, y a t se vygeneruje 3D mřížka a X, Y a T určují
        # hodnoty x, y a t v daných bodech. Např. hodnota x v bodě s indexy i,j,k je X[i,j,k], atd.

        # Delete unneeded tensors to free up memory.
        del x, y, t

        x_eq = tf.reshape(X[1:-1, 1:-1, 1:], [-1])
        y_eq = tf.reshape(Y[1:-1, 1:-1, 1:], [-1])
        t_eq = tf.reshape(T[1:-1, 1:-1, 1:], [-1])
        
        x_boundary_for_fixed_y0 = tf.reshape(X[:, 0, :], [-1])
        y_boundary_for_fixed_y0 = tf.reshape(Y[:, 0, :], [-1])
        t_boundary_for_fixed_y0 = tf.reshape(T[:, 0, :], [-1])
        
        x_boundary_for_fixed_y_last = tf.reshape(X[:, -1, :], [-1])
        y_boundary_for_fixed_y_last = tf.reshape(Y[:, -1, :], [-1])
        t_boundary_for_fixed_y_last = tf.reshape(T[:, -1, :], [-1])
        
        # Zde je y složka bodů od indexu 1 do předposledního (včetně), jelikož ty body s okrajovými hodnotami
        # jsme již započetli výše.
        x_boundary_for_fixed_x0 = tf.reshape(X[0, 1:-1, :], [-1])
        y_boundary_for_fixed_x0 = tf.reshape(Y[0, 1:-1, :], [-1])
        t_boundary_for_fixed_x0 = tf.reshape(T[0, 1:-1, :], [-1])
        
        x_boundary_for_fixed_x_last = tf.reshape(X[-1, 1:-1, :], [-1])
        y_boundary_for_fixed_x_last = tf.reshape(Y[-1, 1:-1, :], [-1])
        t_boundary_for_fixed_x_last = tf.reshape(T[-1, 1:-1, :], [-1])
        
        x_boundary = tf.concat([x_boundary_for_fixed_y0, x_boundary_for_fixed_y_last, x_boundary_for_fixed_x0, x_boundary_for_fixed_x_last], 0)
        y_boundary = tf.concat([y_boundary_for_fixed_y0, y_boundary_for_fixed_y_last, y_boundary_for_fixed_x0, y_boundary_for_fixed_x_last], 0)
        t_boundary = tf.concat([t_boundary_for_fixed_y0, t_boundary_for_fixed_y_last, t_boundary_for_fixed_x0, t_boundary_for_fixed_x_last], 0)

        # Delete unneeded tensors to free up memory.
        del x_boundary_for_fixed_y0, y_boundary_for_fixed_y0, t_boundary_for_fixed_y0,
        del x_boundary_for_fixed_y_last, y_boundary_for_fixed_y_last, t_boundary_for_fixed_y_last,
        del x_boundary_for_fixed_x0, y_boundary_for_fixed_x0, t_boundary_for_fixed_x0,
        del x_boundary_for_fixed_x_last, y_boundary_for_fixed_x_last, t_boundary_for_fixed_x_last

        x_ini = tf.reshape(X[:, :, 0], [-1])
        y_ini = tf.reshape(Y[:, :, 0], [-1])
        t_ini = tf.reshape(T[:, :, 0], [-1])

        del X, Y, T
        
        
        if run_as_graph:
            print("tracing train step from fit method")
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

                loss_dict = self.train_step((x_eq, y_eq, t_eq, x_boundary, y_boundary, t_boundary, x_ini, y_ini, t_ini))

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
        x_eq, y_eq, t_eq, x_boundary, y_boundary, t_boundary, x_ini, y_ini, t_ini = data

        with tf.GradientTape() as weights_update_tape:
            with tf.GradientTape(watch_accessed_variables=False, persistent=True) as derivation_tape:
                derivation_tape.watch(t_eq)
                derivation_tape.watch(x_eq)
                derivation_tape.watch(y_eq)
                txy_eq = tf.stack([t_eq, x_eq, y_eq], axis=1)
                u_pred = self(txy_eq, training=True)

                u_x = derivation_tape.gradient(u_pred, x_eq)
                u_y = derivation_tape.gradient(u_pred, y_eq)
            u_xx = derivation_tape.gradient(u_x, x_eq)
            u_yy = derivation_tape.gradient(u_y, y_eq)
            u_t = derivation_tape.gradient(u_pred, t_eq)
            del derivation_tape

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

            RHS = self.RHS_function(t_eq, x_eq, y_eq, u_pred, self.heat_coeficient)
            equation_residual = u_t - self.heat_coeficient * u_xx - self.heat_coeficient * u_yy - RHS
            weighted_equation_loss = self.equation_loss_coef * self.compute_loss(txy_eq, tf.zeros_like(equation_residual), equation_residual)
            del txy_eq, u_t, u_xx, u_yy


            # if tf.rank(RHS) == 1 and RHS.shape[0] == xt.shape[0]:
            #     RHS = RHS[:, tf.newaxis]
            #     # just a check, in case the RHS_function returs a shape (num of points,) tensor
            #     # instead of shape (num of points, 1) tensor for some reason


            txy_ini = tf.stack([t_ini, x_ini, y_ini], axis=-1)
            u_ini_pred = self(txy_ini, training=True)
            u_ini_target = self.init_cond_function(x_ini, y_ini, u_pred, self.heat_coeficient)
            weighted_init_cond_loss = self.init_cond_loss_coef * self.compute_loss(txy_ini, u_ini_target, u_ini_pred)
            del txy_ini, u_ini_pred, u_ini_target


            txy_boundary = tf.stack([t_boundary, x_boundary, y_boundary], axis=1)
            u_boundary_pred = self(txy_boundary, training=True)
            u_boundary_target = self.bound_cond_function(t_boundary, x_boundary, y_boundary, u_pred, self.heat_coeficient)
            weighted_boundary_loss = self.boundary_loss_coef * self.compute_loss(txy_boundary, u_boundary_target, u_boundary_pred)
            del txy_boundary, u_boundary_pred, u_boundary_target

            loss_dict = {
                "weighted equation loss": weighted_equation_loss,
                "weighted boundary condition loss": weighted_boundary_loss,
                "weighted initial condition loss": weighted_init_cond_loss,
            }

        grads = weights_update_tape.gradient(loss_dict, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss_dict



class HE_2D_PINN_old_without_custom_fit(keras.Sequential):
    """
    Class implementing the heat equation u_t - u_xx - u_yy = RHS_function with Dirichlet boundary conditions and
    initial condition given by init_cond_function.

    Args:
    RHS_function:
        - a function made with tensorflow operations
        - input (x, y, t), where x, y and t are tensors with points in which the right hand side is to be evaluated
        - returns the right hand side of the equation

    init_cond_function:
        - a function made with tensorflow operations
        - input (x, y, t), where x, y and t are tensors with points in which the right hand side is to be evaluated
        - returns the initial condition
    .
    """
    def __init__(
        self,
        RHS_function,
        init_cond_function,
        boun_cond_funtion,
        x_start_point,
        x_end_point,
        num_of_x_points,
        # num_of_x_points_for_equation,
        # num_of_x_points_for_bound_cond,
        # num_of_x_points_for_init_cond,
        y_start_point,
        y_end_point,
        num_of_y_points,
        # num_of_y_points_for_equation,
        # num_of_y_points_for_bound_cond,
        # num_of_y_points_for_init_cond,
        t_start_point,
        t_end_point,
        num_of_t_points,
        # num_of_t_points_for_equation,
        # num_of_t_points_for_bound_cond,
        *args,
        equation_loss_coef=1.0,
        boundary_loss_coef=1.0,
        init_cond_loss_coef=1.0,
        heat_coeficient=1.0,
        **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.RHS_function = RHS_function
        self.init_cond_function = init_cond_function
        self.bound_cond_function = boun_cond_funtion

        self.x_start_point = x_start_point
        self.x_end_point = x_end_point
        self.num_of_x_points = num_of_x_points
        # self.num_of_x_points_for_equation = num_of_x_points_for_equation
        # self.num_of_x_points_for_init_cond = num_of_x_points_for_init_cond

        self.y_start_point = y_start_point
        self.y_end_point = y_end_point
        self.num_of_y_points = num_of_y_points
        # self.num_of_y_points_for_equation = num_of_y_points_for_equation
        # self.num_of_y_points_for_init_cond = num_of_y_points_for_init_cond
        self.num_of_t_points = num_of_t_points
        self.t_start_point = t_start_point
        self.t_end_point = t_end_point

        # self.num_of_t_points_for_equation = num_of_t_points_for_equation

        self.equation_loss_coef = equation_loss_coef
        self.boundary_loss_coef = boundary_loss_coef
        self.init_cond_loss_coef = init_cond_loss_coef
        self.heat_coeficient = heat_coeficient


    def fit(self, **kwargs):
        """
        Keras fit, just do not pass in any x, y and batch_size arguments, they will be generated.
        Pass the rest of the arguments as keyword arguments.
        .
        """
        x = tf.linspace(self.x_start_point, self.x_end_point, self.num_of_x_points)
        y = tf.linspace(self.y_start_point, self.y_end_point, self.num_of_y_points)
        t = tf.linspace(self.t_start_point, self.t_end_point, self.num_of_t_points)

        X, Y, T = tf.meshgrid(x, y, t, indexing='ij')
        # Indexování 'ij' zařídí, že bod na indexu [i,j,k] odpovídá bodu s hodnotami x[i], y[j], t[k].
        # Tedy prostě standardní kartészké indexování: z hodnot x, y a t se vygeneruje 3D mřížka a X, Y a T určují
        # hodnoty x, y a t v daných bodech. Např. hodnota x v bodě s indexy i,j,k je X[i,j,k], atd.

        # Delete unneeded tensors to free up memory.
        del x, y, t

        x_eq = tf.reshape(X[1:-1, 1:-1, 1:], [-1])
        y_eq = tf.reshape(Y[1:-1, 1:-1, 1:], [-1])
        t_eq = tf.reshape(T[1:-1, 1:-1, 1:], [-1])
        
        x_boundary_for_fixed_y0 = tf.reshape(X[:, 0, :], [-1])
        y_boundary_for_fixed_y0 = tf.reshape(Y[:, 0, :], [-1])
        t_boundary_for_fixed_y0 = tf.reshape(T[:, 0, :], [-1])
        
        x_boundary_for_fixed_y_last = tf.reshape(X[:, -1, :], [-1])
        y_boundary_for_fixed_y_last = tf.reshape(Y[:, -1, :], [-1])
        t_boundary_for_fixed_y_last = tf.reshape(T[:, -1, :], [-1])
        
        # Zde je y složka bodů od indexu 1 do předposledního (včetně), jelikož ty body s okrajovými hodnotami
        # jsme již započetli výše.
        x_boundary_for_fixed_x0 = tf.reshape(X[0, 1:-1, :], [-1])
        y_boundary_for_fixed_x0 = tf.reshape(Y[0, 1:-1, :], [-1])
        t_boundary_for_fixed_x0 = tf.reshape(T[0, 1:-1, :], [-1])
        
        x_boundary_for_fixed_x_last = tf.reshape(X[-1, 1:-1, :], [-1])
        y_boundary_for_fixed_x_last = tf.reshape(Y[-1, 1:-1, :], [-1])
        t_boundary_for_fixed_x_last = tf.reshape(T[-1, 1:-1, :], [-1])
        
        x_boundary = tf.concat([x_boundary_for_fixed_y0, x_boundary_for_fixed_y_last, x_boundary_for_fixed_x0, x_boundary_for_fixed_x_last], 0)
        y_boundary = tf.concat([y_boundary_for_fixed_y0, y_boundary_for_fixed_y_last, y_boundary_for_fixed_x0, y_boundary_for_fixed_x_last], 0)
        t_boundary = tf.concat([t_boundary_for_fixed_y0, t_boundary_for_fixed_y_last, t_boundary_for_fixed_x0, t_boundary_for_fixed_x_last], 0)

        # Delete unneeded tensors to free up memory.
        del x_boundary_for_fixed_y0, y_boundary_for_fixed_y0, t_boundary_for_fixed_y0,
        del x_boundary_for_fixed_y_last, y_boundary_for_fixed_y_last, t_boundary_for_fixed_y_last,
        del x_boundary_for_fixed_x0, y_boundary_for_fixed_x0, t_boundary_for_fixed_x0,
        del x_boundary_for_fixed_x_last, y_boundary_for_fixed_x_last, t_boundary_for_fixed_x_last

        x_ini = tf.reshape(X[:, :, 0], [-1])
        y_ini = tf.reshape(Y[:, :, 0], [-1])
        t_ini = tf.reshape(T[:, :, 0], [-1])

        del X, Y, T

        # Dataset is needed in order to be able to pass multiple tensors (as a tuple) into the keras fit function
        dataset = tf.data.Dataset.from_tensors( (x_eq, y_eq, t_eq, x_boundary, y_boundary, t_boundary, x_ini, y_ini, t_ini) )
        
        # tf.print(x_eq.shape)
        # tf.print(y_eq.shape)
        # tf.print(t_eq.shape)
        # tf.print(x_boundary.shape)
        # tf.print(y_boundary.shape)
        # tf.print(t_boundary.shape)
        # tf.print(x_ini.shape)
        # tf.print(y_ini.shape)
        # tf.print(t_ini.shape)

        # TODO: Asi udělat vlastní training loop, aby se daly input body batchovat.

        return super().fit(dataset, **kwargs)

    def train_step(self, data):
        # print("Tracing train step.")
        x_eq, y_eq, t_eq, x_boundary, y_boundary, t_boundary, x_ini, y_ini, t_ini = data

        with tf.GradientTape() as weights_update_tape:
            with tf.GradientTape(watch_accessed_variables=False, persistent=True) as derivation_tape:
                derivation_tape.watch(t_eq)
                derivation_tape.watch(x_eq)
                derivation_tape.watch(y_eq)
                txy_eq = tf.stack([t_eq, x_eq, y_eq], axis=1)
                u_pred = self(txy_eq, training=True)

                u_x = derivation_tape.gradient(u_pred, x_eq)
                u_y = derivation_tape.gradient(u_pred, y_eq)
            u_xx = derivation_tape.gradient(u_x, x_eq)
            u_yy = derivation_tape.gradient(u_y, y_eq)
            u_t = derivation_tape.gradient(u_pred, t_eq)
            del derivation_tape

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

            RHS = self.RHS_function(t_eq, x_eq, y_eq, u_pred, self.heat_coeficient)
            equation_residual = u_t - self.heat_coeficient * u_xx - self.heat_coeficient * u_yy - RHS
            weighted_equation_loss = self.equation_loss_coef * self.compute_loss(txy_eq, tf.zeros_like(equation_residual), equation_residual)
            del txy_eq, u_t, u_xx, u_yy


            # if tf.rank(RHS) == 1 and RHS.shape[0] == xt.shape[0]:
            #     RHS = RHS[:, tf.newaxis]
            #     # just a check, in case the RHS_function returs a shape (num of points,) tensor
            #     # instead of shape (num of points, 1) tensor for some reason


            txy_ini = tf.stack([t_ini, x_ini, y_ini], axis=-1)
            u_ini_pred = self(txy_ini, training=True)
            u_ini_target = self.init_cond_function(x_ini, y_ini, u_pred, self.heat_coeficient)
            weighted_init_cond_loss = self.init_cond_loss_coef * self.compute_loss(txy_ini, u_ini_target, u_ini_pred)
            del txy_ini, u_ini_pred, u_ini_target


            txy_boundary = tf.stack([t_boundary, x_boundary, y_boundary], axis=1)
            u_boundary_pred = self(txy_boundary, training=True)
            u_boundary_target = self.bound_cond_function(t_boundary, x_boundary, y_boundary, u_pred, self.heat_coeficient)
            weighted_boundary_loss = self.boundary_loss_coef * self.compute_loss(txy_boundary, u_boundary_target, u_boundary_pred)
            del txy_boundary, u_boundary_pred, u_boundary_target

            loss_dict = {
                "weighted equation loss": weighted_equation_loss,
                "weighted boundary condition loss": weighted_boundary_loss,
                "weighted initial condition loss": weighted_init_cond_loss,
            }

        grads = weights_update_tape.gradient(loss_dict, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss_dict
