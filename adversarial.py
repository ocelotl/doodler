import IPython
# from ipdb import set_trace
import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display

# This is done to avoid the
# "could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR" error that is
# apparently caused by insufficient pre-allocated memory. A different
# CUDA_ERROR_OUT_OF_MEMORY shows up now but the code runs still (shrugs).

# This example is taken from here:
# https://github.com/tensorflow/docs/blob/
# master/site/en/r2/tutorials/generative/dcgan.ipynb
tf.config.gpu.set_per_process_memory_growth(True)

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

# train_images.shape == (60000, 28, 28)
# train_labels.shape == (60000,)

# train_images[0].shape ==(28, 28)

# train_images is 60000 28 x 28 matrices of values between 0 and 255

train_images = train_images.reshape(
    train_images.shape[0], 28, 28, 1
).astype('float32')

# Now every element of the 60000 matrices is a list of a single float.

# Normalize the images to [-1, 1]
train_images = (train_images - 127.5) / 127.5

# Not sure about last step, previous values that are equal to 0 will end up
# being -1 but previous values smaller than 127.5 will end up being negative
# too but not -1. The rest of the values will end up being positive. This means
# that there will be values different than -1 and 1.

BUFFER_SIZE = 60000
BATCH_SIZE = 256

# A Dataset-type object is created here by taking train_images, shuffling its
# elements and groupin them into batches.
# Batch and shuffle the data
train_dataset = (
    tf.data.Dataset.from_tensor_slices(train_images).
    shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
)

generator = tf.keras.Sequential()

# Dense implements the operation output = activation(dot(input, kernel) + bias)
# where actiavtion is the element-wise activation function, kernel is a weights
# matrix created by the layer and bias is a bias vector created by the layer if
# use_bias is True
# No activation function is the default: activation(x) = x
# 7 * 7 * 256 is the amount of units in the layer, the dimensionality of the
# output space.
# More about dot products used in machine learning here:
# https://medium.com/data-science-bootcamp/understand-dot-products-matrix-
# multiplications-usage-in-deep-learning-in-minutes-beginner-95edf2e66155
# The model will take as input arrays of shape (*, 100)

# Seems like input shape is this one because the input is a [1, 100]-shaped
# tensor of random noise.
# FIXME why are the units that?

generator.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))

# Normalize the activations of the previous layer at each batch, i.e. applies a
# transformation that maintains the mean activation close to 0 and the
# activation standard deviation close to 1.
# FIXME understand this
# This seems to be an optimization that speeds up the training. The
# distribution of each layer's inputs change during training as the parameters
# of the previous layers change, and this slows down the trainin process, more
# information here: https://arxiv.org/abs/1502.03167
generator.add(layers.BatchNormalization())

# Rectifier Linear Unit is an activation function that returns x if x is
# positive and returns 0 if x is negative. A LeakyReLU is the same thing but
# returns 0.01 * x if the x is negative. More information here:
# https://medium.com/tinymind/a-practical-guide-to-relu-b83ca804f1f7
generator.add(layers.LeakyReLU())

# FIXME why this shape?
generator.add(layers.Reshape((7, 7, 256)))
# Note: None is the batch size
assert generator.output_shape == (None, 7, 7, 256)

# Seems to be the same as Conv2D but transposed (duh). The documentation shows
# that this is useful when there is a need to use a transformation that goes in
# the opposite direction of a normal convolution, from something that has the
# shape of the output of some convolution to something that has the shape of
# its input while maintaining a connectivity pattern that is compatible with
# said convolution.
# filters: the number of output filters in the convolution (the dimensionalty
# of the output space). This layer seems to return the "results" of applying
# 128 different convolution filters to the images.
# kernel_size: height and width of the 2D convolution window
# More information for the remaining arguments:
# https://stackoverflow.com/questions/37674306/what-is-the-difference-between-
# same-and-valid-padding-in-tf-nn-max-pool-of-t
# https://github.com/vdumoulin/conv_arithmetic/tree/master/gif
generator.add(
    layers.Conv2DTranspose(
        128,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding='same',
        use_bias=False
    )
)
assert generator.output_shape == (None, 7, 7, 128)

# Notice how this pair of layers are added between each other one.
generator.add(layers.BatchNormalization())
generator.add(layers.LeakyReLU())

generator.add(
    layers.Conv2DTranspose(
        64,
        (5, 5),
        strides=(2, 2),
        padding='same',
        use_bias=False
    )
)
assert generator.output_shape == (None, 14, 14, 64)
generator.add(layers.BatchNormalization())
generator.add(layers.LeakyReLU())

# Seems like tanh is the hyperbolic tangent function
generator.add(
    layers.Conv2DTranspose(
        1,
        (5, 5),
        strides=(2, 2),
        padding='same',
        use_bias=False,
        activation='tanh'
    )
)

# Here is the shape of an image that is generated by the generator, 28 by 28
# pixels.
assert generator.output_shape == (None, 28, 28, 1)

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

# The shape of generated_image is [1, 28, 28, 1].
# The "slicing" operation made here over generated_image returns a
# [28, 28]-shaped tensor.
plt.imshow(generated_image[0, :, :, 0], cmap='gray')

discriminator = tf.keras.Sequential()
discriminator.add(
    layers.Conv2D(
        64,
        (5, 5),
        strides=(2, 2),
        padding='same',
        input_shape=[28, 28, 1]
    )
)
discriminator.add(layers.LeakyReLU())

# Dropout sets a specified fraction of input units to 0 at each update during
# training time, which helps prevent overfitting.
discriminator.add(layers.Dropout(0.3))

discriminator.add(
    layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')
)
discriminator.add(layers.LeakyReLU())
discriminator.add(layers.Dropout(0.3))

discriminator.add(layers.Flatten())
# Notice how the last layer of the discriminator is simply a single neuron and
# its ouptut is a [1, 1] tensor whose only value is a number.
discriminator.add(layers.Dense(1))

decision = discriminator(generated_image)
print(decision)

# Apparently a logit is a function that maps probability values that go from
# 0 to 1 to -infinite to infinite. A probablity of 0.5 will be mapped to 0.
# from_logits = True means that the input of probabilities will be a logit
# tensor.
# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Adam optimization is a stochastic gradient descent method. Its parameter is
# the learning rate.
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator
)

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
# This decorator does a lot of things:
# https://www.tensorflow.org/alpha/tutorials/eager/tf_function
# @tf.function
def train_step(images):
    # This train_step function will be called as many times as batches of
    # images there are. Funny thing is that the generator will always start
    # with random noise.
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    # GradientTape objects record operations for automatic differentiation
    # Operations are recorded if they are executed within this context manager
    # and at least one of their inputs is being "watched".
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        # generated_images has a shape of [256, 28, 28, 1]

        # FIXME what does training do?

        # real_output and fake_output are logit tensors, their values are real
        # numbers. A value of 0 means a probability of 0.5, an absolutely large
        # negative value means a low probability and an absolutely large
        # positive value means a high probability.

        # The discriminator starts without knowing anything about the real
        # images, at the beginning, real_output and fake_output are just the
        # results of running the real and fake images through the discriminator
        # and these results should be kind of similar since the discriminator
        # has not had its weights adjusted yet.
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        # real_output and fake_output have shapes of [256, 1]

        # This cross_entropy object has a __call__ method that has these
        # parameters:
        # y_true: ground truth values
        # y_pred: predicted values

        # Here is where we "communicate" one neural network with the other. We
        # calculate the loss of the generator by telling it the results that
        # the discriminator produced for the fake images that the generator
        # produced. So, basically, we tell the generator "this image that you
        # generated was rejected by the discriminator and this image that you
        # generated was accepted by the generator". This will then make the
        # generator favor the images that the generator accepted, the ones that
        # were able to "cheat" the generator. This introduces a tendency in the
        # generator to generate images similar to the ones that "cheated" the
        # discriminator.
        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

        # Here we basically do this:
        # real_loss is the loss when we use ones with real_output. In this way
        # we train the generator to get better at detecting real images by
        # "telling" its loss function that the real_output results (the ones
        # obtained by running real images) are to be considered as "true"
        # results because we use a vector of ones against it. Viceversa for
        # the results gotten by running the fake images.
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

        # So, basically, we train the generator to get better at clasifying
        # real images as real and fake images as fake.
        disc_loss = real_loss + fake_loss

    print('generator loss: {}'.format(gen_loss))
    print('real loss: {}'.format(real_loss))
    print('fake loss: {}'.format(fake_loss))

    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator.trainable_variables
    )
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, discriminator.trainable_variables)
    )


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(
            epoch + 1, time.time() - start)
        )

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))

    # This is done to make the image being shown not stop the whole program
    # execution.
    plt.show(block=False)
    plt.pause(1)
    plt.close()


train(train_dataset, EPOCHS)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


# Display a single image using the epoch number
def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


display_image(EPOCHS)

anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('image*.png')
    filenames = sorted(filenames)
    last = -1
    for i, filename in enumerate(filenames):
        frame = 2 * (i ** 0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)

if IPython.version_info > (6, 2, 0, ''):
    display.Image(filename=anim_file)
