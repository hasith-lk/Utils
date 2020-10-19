from keras.layers import Dense, Input

inputs = Input((8,))
layer = Dense(8)

x = layer(inputs)

layer.name
layer.get_weights()
layer.input, layer.output, layer.input_shape, layer.output_shape

x = layer(x)
layer.input, layer.output, layer.input_shape, layer.output_shape
layer.get_weights()


config = layer.get_config()
config