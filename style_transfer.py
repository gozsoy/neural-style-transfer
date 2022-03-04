import ssl
import matplotlib
import numpy as np
import tensorflow as tf
from utils import CustomAdam
ssl._create_default_https_context = ssl._create_unverified_context

print(tf.config.list_physical_devices())


# preprocessing and loss functions


def load_img(path_to_img):
    max_dim = 512

    # loading
    img = tf.io.read_file(path_to_img)
    img = tf.io.decode_jpeg(img)  # H,W,C
    img = tf.image.convert_image_dtype(img, tf.float32)

    # reshaping
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)

    # preprocessing for vgg
    img = img[tf.newaxis, :]  # adding batch axis

    return img


# compute content_loss for 1 layer
def content_loss(content_repr, current_content_repr):
    channel_size = content_repr.shape[-1]
    f = tf.reshape(content_repr, [1, -1, channel_size])
    p = tf.reshape(current_content_repr, [1, -1, channel_size])

    return 0.5 * tf.reduce_sum(tf.pow(f - p, 2))


# compute style loss for 1 layer
def style_loss_layer(temp_style_repr, temp_current_style_repr):
    channel_size = temp_style_repr.shape[-1]
    temp_style_repr = tf.reshape(temp_style_repr, [1, -1, channel_size])
    _, ml, nl = temp_style_repr.shape
    temp_current_style_repr = tf.reshape(temp_current_style_repr, [1, -1, channel_size])

    g = tf.transpose(temp_style_repr, perm=[0, 2, 1]) @ temp_style_repr
    a = tf.transpose(temp_current_style_repr, perm=[0, 2, 1]) @ temp_current_style_repr

    return (1./(4 * ml**2 * nl**2)) * tf.reduce_sum(tf.pow(g - a, 2))


# compute style loss for all layers
def style_loss(style_repr, current_style_repr, style_layer_weights=None):

    # convert to list if only using 1 style repr
    if not isinstance(style_repr, list):
        style_repr = [style_repr]
        current_style_repr = [current_style_repr]

    if style_layer_weights is None:  # assign equal weight
        style_layer_weights = list(map(lambda x: (1./len(style_repr)), style_repr))

    total = 0
    for idx in range(len(style_repr)):
        total += style_layer_weights[idx] * style_loss_layer(style_repr[idx], current_style_repr[idx])

    return total


# compute variation loss for image (to reduce high frequency artifacts)
# borrowed from https://www.tensorflow.org/tutorials/generative/style_transfer
def variation_loss(img):
    x_deltas = img[:, :, 1:, :] - img[:, :, :-1, :]
    y_deltas = img[:, 1:, :, :] - img[:, :-1, :, :]
    return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))


def main():

    # load content and style images
    content_img = load_img('images/tubingen.jpg')
    style_img = load_img('images/kandinsky.jpg')

    # load pretrained vgg19 model
    base = tf.keras.applications.VGG19(include_top=False)

    # change maxpooling with average pooling layers
    x = base.layers[0].output
    for l_idx in range(1, len(base.layers)):
        current_layer = base.layers[l_idx]

        if isinstance(current_layer, tf.keras.layers.MaxPool2D):
            x = tf.keras.layers.AveragePooling2D(**current_layer.get_config())(x)
        else:
            x = current_layer(x)

    vgg19 = tf.keras.Model(inputs=base.layers[0].input, outputs=x)
    # make the base model non trainable
    vgg19.trainable = False

    # decide content and style layers
    content_layers = ['block4_conv2']
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

    content_outputs = []
    style_outputs = []
    for layer in vgg19.layers:
        if layer.name in content_layers:
            content_outputs.append(layer.output)
        if layer.name in style_layers:
            style_outputs.append(layer.output)

    # create models which only extract content and style outputs
    model_content = tf.keras.Model(inputs=vgg19.input, outputs=content_outputs)
    model_style = tf.keras.Model(inputs=vgg19.input, outputs=style_outputs)

    # obtain content and style representations once
    content_repr = model_content(tf.keras.applications.vgg19.preprocess_input(content_img * 255.))
    style_repr = model_style(tf.keras.applications.vgg19.preprocess_input(style_img * 255.))


    # training loop
    optimizer = CustomAdam(learning_rate=0.02, beta1=0.99, epsilon=1e-1)  # use tf.keras.optimizers.Adam if have NVIDIA GPU

    # initialize with trainable noise
    # noise_img = tf.Variable(initial_value=tf.random.normal(shape=content_img.shape,mean = np.mean(content_img),stddev = np.std(content_img)))
    # OR
    # initialize with content image
    noise_img = tf.Variable(initial_value=content_img)

    model_noise = tf.keras.Model(inputs=vgg19.input, outputs=style_outputs + content_outputs)

    beta = 1  # style
    alpha = 1e-4  # content
    var_weigth = 30  # variation weight

    EPOCHS = 1000

    for e in range(EPOCHS):

        # training loop
        with tf.GradientTape() as tape:
            
            noise_repr = model_noise(tf.keras.applications.vgg19.preprocess_input(noise_img * 255.))
            current_content_repr = noise_repr[-1]
            current_style_repr = noise_repr[:-1]

            loss_c = alpha * content_loss(content_repr, current_content_repr)
            loss_s = beta * style_loss(style_repr, current_style_repr)
            loss_v = var_weigth * variation_loss(noise_img)
            loss = loss_s + loss_c + loss_v
        
        gradients = tape.gradient(loss, [noise_img])
        optimizer.apply_gradients(zip(gradients, [noise_img]))
        noise_img.assign(tf.clip_by_value(noise_img, clip_value_min=0., clip_value_max=1.))
        
        if e % 100 == 0:
            print(f'epoch: {e}, content_loss: {loss_c.numpy()}, style_loss: {loss_s.numpy()}, variation_loss: {loss_v.numpy()}')
        matplotlib.image.imsave('images/created.jpg', np.array(noise_img[0]*255., dtype=np.uint8))


if __name__ == '__main__':
    main()
