import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class SimpleModel(Model):

    def __init__(self):
        super().__init__()
        self.dense0 = Dense(2)
        self.dense1 = Dense(1)

    def call(self, inputs):
        z = self.dense0(inputs)
        z = self.dense1(z)  # Breakpoint in IDE here. =====
        return z

x = tf.convert_to_tensor([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)

model0 = SimpleModel()
y0 = model0.call(x)  # Values of z shown at breakpoint. =====

# model1 = SimpleModel()
# model1.run_eagerly = True
# model1.compile(optimizer=Adam(), loss=BinaryCrossentropy())
# y1 = model1.predict(x)  # Values of z *not* shown at breakpoint. =====

model2 = SimpleModel()
model2.compile(optimizer=Adam(), loss=BinaryCrossentropy())
model2.run_eagerly = True
y2 = model2.predict(x)  # Values of z shown at breakpoint. =====