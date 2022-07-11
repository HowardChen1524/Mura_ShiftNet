import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
import glob
from tensorflow.keras.losses import MeanSquaredError
import tensorflow as tf
y_true = np.array([
    [[[0., 1., 1., 1.], [2., 0. , 1., 1.], [0., 1., 1., 1.], [2., 0., 1., 1.]]],
    [[[0., 1., 1., 1.], [2., 0. , 1., 1.], [0., 1., 1., 1.], [2., 0., 1., 1.]]],
    [[[0., 1., 1., 1.], [2., 0. , 1., 1.], [0., 1., 1., 1.], [2., 0., 1., 1.]]],
    [[[0., 1., 1., 1.], [2., 0. , 1., 1.], [0., 1., 1., 1.], [2., 0., 1., 1.]]]
    ])
y_pred = np.array([
    [[[0., 1., 0., 0.], [2., 0. , 1., 1.], [1., 1., 1., 1.], [3., 0., 1., 1.]]],
    [[[0., 1., 0., 0.], [2., 0. , 1., 1.], [1., 1., 1., 1.], [3., 0., 1., 1.]]],
    [[[0., 1., 0., 0.], [2., 0. , 1., 1.], [1., 1., 1., 1.], [3., 0., 1., 1.]]],
    [[[0., 1., 0., 0.], [2., 0. , 1., 1.], [1., 1., 1., 1.], [3., 0., 1., 1.]]]
    ])
# Using 'auto'/'sum_over_batch_size' reduction type.
mse = tf.keras.losses.MeanSquaredError()
print(y_true.shape)
print(y_pred.shape)
print(mse(y_true, y_pred).numpy())
