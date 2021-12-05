import tensorflow as tf


def res_blk_down_up(x, filters=32, is_bn=False, is_train=True, stride=2):
    bn1 = tf.layers.batch_normalization(x, training=is_train, name='bn1') if is_bn else x
    relu1 = tf.nn.relu(bn1)
    # down-sample stride = 2
    conv1 = tf.layers.conv2d(
        relu1, filters, (3, 3), (stride, stride), padding='same',
        kernel_initializer=tf.initializers.random_normal(stddev=tf.sqrt(2. / 3 / 3 / filters)),
        name='conv1'
    )
    bn2 = tf.layers.batch_normalization(conv1, training=is_train, name='bn2') if is_bn else conv1
    relu2 = tf.nn.relu(bn2)
    # up-sample stride = 2
    conv2 = tf.layers.conv2d_transpose(
        relu2, filters, (3, 3), (stride, stride), padding='same',
        kernel_initializer=tf.initializers.random_normal(stddev=tf.sqrt(2. / 3 / 3 / filters)),
        name='conv2'
    )
    return x + conv2


def res_blk(x, filters=32, is_bn=False, is_train=True, f_size=3, mid_filters=-1):
    bn1 = tf.layers.batch_normalization(x, training=is_train, name='bn1') if is_bn else x
    relu1 = tf.nn.relu(bn1)
    if mid_filters < 0:
        mid_filters = filters
    conv1 = tf.layers.conv2d(
        relu1, mid_filters, (f_size, f_size), (1, 1), padding='same',
        kernel_initializer=tf.initializers.random_normal(stddev=tf.sqrt(2. / 3 / 3 / mid_filters)),
        name='conv1'
    )
    bn2 = tf.layers.batch_normalization(conv1, training=is_train, name='bn2') if is_bn else conv1
    relu2 = tf.nn.relu(bn2)
    conv2 = tf.layers.conv2d(
        relu2, filters, (f_size, f_size), (1, 1), padding='same',
        kernel_initializer=tf.initializers.random_normal(stddev=tf.sqrt(2. / 3 / 3 / filters)),
        name='conv2'
    )
    return x[:, :, :, :filters] + conv2  # in case concat kernel


def res_blk_single_layer(x, filters=32, is_bn=False, is_train=True, f_size=3):
    bn1 = tf.layers.batch_normalization(x, training=is_train, name='bn1') if is_bn else x
    relu1 = tf.nn.relu(bn1)
    conv1 = tf.layers.conv2d(
        relu1, filters, (f_size, f_size), (1, 1), padding='same',
        kernel_initializer=tf.initializers.random_normal(stddev=tf.sqrt(2. / 3 / 3 / filters)),
        name='conv1'
    )
    return x + conv1


def res_blk_with_param(x, params):
    """
    :param x: input tensor
    :param params: a dictionary of parameters, with keys 'w1', 'w2', 'b1', 'b2'
    :return: output tensor
    """
    keys = ['w1', 'w2', 'b1', 'b2']
    try:
        w1, w2, b1, b2 = [params[k] for k in keys]
    except:
        raise ValueError('Invalid parameters!')
    relu1 = tf.nn.relu(x)
    conv1 = tf.nn.conv2d(relu1, filter=w1, strides=[1, 1, 1, 1], padding='SAME')
    out1 = tf.nn.bias_add(conv1, b1, data_format='NHWC')
    relu2 = tf.nn.relu(out1)
    conv2 = tf.nn.conv2d(relu2, filter=w2, strides=[1, 1, 1, 1], padding='SAME')
    out2 = tf.nn.bias_add(conv2, b2)
    return x + out2


def path_res_blk(params):
    """
    :param params: a dictionary of parameters, with keys 'w1', 'w2', 'b1', 'b2'
    :return: a function that represents the path with the given parameters
    """
    return lambda x: res_blk_with_param(x, params)


def res_blks_with_param(x, params, num_blk):
    # multiple residual blocks with parameters
    out = x
    param_key = ['w1', 'w2', 'b1', 'b2']
    for k in range(1, num_blk+1):
        str_k = str(k)
        keys = ['w'+str_k+'1', 'w'+str_k+'2', 'b'+str_k+'1', 'b'+str_k+'2']
        try:
            out = res_blk_with_param(out, dict(zip(param_key, [params[key] for key in keys])))
        except:
            raise ValueError('Invalid parameters!')
    return out


def path_res_blks(params, num_blk):
    return lambda x: res_blks_with_param(x, params, num_blk)


# used for building router
def conv_relu_with_param(x, w, b, stride):
    """ conv + ReLU
    :param x: input
    :param w: weight
    :param b: bias
    :param stride: stride of convolution
    :return: output
    """
    conv = tf.nn.conv2d(x, filter=w, strides=[1, stride, stride, 1], padding='SAME')
    conv_bias = tf.nn.bias_add(conv, b)
    out = tf.nn.relu(conv_bias)
    return out


def conv_with_param(x, w, b, stride):
    """ conv + ReLU
    :param x: input
    :param w: weight
    :param b: bias
    :param stride: stride of convolution
    :return: output
    """
    conv = tf.nn.conv2d(x, filter=w, strides=[1, stride, stride, 1], padding='SAME')
    out = tf.nn.bias_add(conv, b)
    return out


def linear_with_param(x, w, b):
    """ fc
    :param x: input
    :param w: weight matrix
    :param b: bias
    :return: output
    """
    linear = tf.matmul(x, w)
    out = tf.nn.bias_add(linear, b)
    return out


def linear_relu_with_param(x, w, b):
    """ fc + ReLU """
    out = tf.nn.relu(linear_with_param(x, w, b))
    return out


# sub-pixel shuffle
def subpixel_shuffle(x, scale=4):
    shape = tf.shape(x)
    batch = shape[0]; H = shape[1]; W = shape[2]; C = shape[3]
    out = tf.reshape(x, [batch, H, W, scale, scale, -1])
    out = tf.transpose(out, [0, 1, 3, 2, 4, 5])
    out = tf.reshape(out, [batch, H*scale, W*scale, -1])
    return out


# get shape after pooling / strided conv
def get_shape_after_pool(input_size, scales):
    """
    :param input_size: a tuple (H, W), note H and W are tf tensors
    :param scales: a list, down-sampling scale for each strided conv / pooling
    :return: output size, a tuple (h, w)
    """
    h, w = input_size
    for cur_scale in scales:
        h = tf.cast(tf.math.floor((h - 1) / cur_scale) + 1, tf.int32)
        w = tf.cast(tf.math.floor((w - 1) / cur_scale) + 1, tf.int32)
    return h, w


# ConvLSTM, change all Hadamard products to convolutions
def my_conv_lstm(x, input_state, p_in, p_for, p_cell, p_out):
    """
    :param x: input
    :param input_state: input state (h_in, c_in)
    :param p_in: input gate parameters (3 conv, 1 bias)
    :param p_for: forget gate parameters (3 conv, 1 bias)
    :param p_cell: update cell parameters (2 conv, 1 bias)
    :param p_out: output gate parameters (3 conv, 1 bias)
    :return: output state h_out, output cell c_out
    """
    h_in, c_in = input_state
    # input gate
    wi_x, wi_h, wi_c, bi = p_in
    conv_in = tf.nn.conv2d(x, filter=wi_x, strides=[1, 1, 1, 1], padding='SAME') + \
              tf.nn.conv2d(h_in, filter=wi_h, strides=[1, 1, 1, 1], padding='SAME') + \
              tf.nn.conv2d(c_in, filter=wi_c, strides=[1, 1, 1, 1], padding='SAME')
    gate_in = tf.nn.sigmoid(tf.nn.bias_add(conv_in, bi))

    # forget gate
    wf_x, wf_h, wf_c, bf = p_for
    conv_for = tf.nn.conv2d(x, filter=wf_x, strides=[1, 1, 1, 1], padding='SAME') + \
               tf.nn.conv2d(h_in, filter=wf_h, strides=[1, 1, 1, 1], padding='SAME') + \
               tf.nn.conv2d(c_in, filter=wf_c, strides=[1, 1, 1, 1], padding='SAME')
    gate_for = tf.nn.sigmoid(tf.nn.bias_add(conv_for, bf))

    # update cell
    wc_x, wc_h, bc = p_cell
    conv_cell = tf.nn.conv2d(x, filter=wc_x, strides=[1, 1, 1, 1], padding='SAME') + \
                tf.nn.conv2d(h_in, filter=wc_h, strides=[1, 1, 1, 1], padding='SAME')
    act_cell = tf.nn.tanh(tf.nn.bias_add(conv_cell, bc))  # tanh activation
    c_out = gate_for * c_in + gate_in * act_cell

    # output gate
    wo_x, wo_h, wo_c, bo = p_out
    conv_out = tf.nn.conv2d(x, filter=wo_x, strides=[1, 1, 1, 1], padding='SAME') + \
               tf.nn.conv2d(h_in, filter=wo_h, strides=[1, 1, 1, 1], padding='SAME') + \
               tf.nn.conv2d(c_in, filter=wo_c, strides=[1, 1, 1, 1], padding='SAME')
    gate_out = tf.nn.sigmoid(tf.nn.bias_add(conv_out, bo))
    h_out = gate_out * tf.nn.tanh(c_out)

    return h_out, c_out
