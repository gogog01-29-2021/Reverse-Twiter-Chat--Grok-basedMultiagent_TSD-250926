from keras.layers import Layer, Dense
from tensorflow import concat

class MultiActivationDense(Layer):
    def __init__(self, units: int | list, activations: list, **dense_layer_kwargs):
        super().__init__()

        self.dense_layer_kwargs = dense_layer_kwargs
        self.activations = activations
        self.units_list = units if type(units) == list else [units]*len(activations)

        if type(units) == list and len(activations) != len(units):
            raise ValueError(f"The number of activation functions (currently {len(activations)}) must match the number of unit groups (currently {len(units)}).")

        self.dense_layers = [Dense(units=units, activation=activation, **self.dense_layer_kwargs)
                                                for activation, units in zip(self.activations, self.units_list)]

    
    def call(self, inputs):
        outputs = []
        for dense_layer in self.dense_layers:
            outputs.append(dense_layer(inputs))
        
        return concat(outputs, axis=-1)
