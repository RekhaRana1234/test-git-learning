#train coding values

import tensorflow as tf
import numpy as np


input_d = np.array([-2, -1, 0, 1, 2, 3, 4 ], dtype=float)
output = np.array([-5, -1, 3, 7, 11, 15, 19], dtype=float)


# defining layer
epochs_value = 500

layer = tf.keras.layers.Dense(units=1, input_shape=[1])

model = tf.keras.Sequential([layer])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

model.fit(input_d, output, epochs=epochs_value, verbose=False)

print(model.predict([400.0]))

print("layer varibales is : {}".format(layer.get_weights()))
