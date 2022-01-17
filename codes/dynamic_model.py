from block import (
    res_blk, path_res_blks, conv_relu_with_param,
    linear_relu_with_param, linear_with_param,
    res_blk_down_up, res_blk_single_layer, subpixel_shuffle
)
from dataset import MyData
import math
import tensorflow as tf


class DynamicModel(object):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

        # define dataset
        with tf.device('/cpu:0'):
            self.dataset = MyData(**kwargs)

        # build graph
        self.graph_dyn = tf.Graph()
        config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )  # config gpu memory usage
        config.gpu_options.allow_growth = True
        self.sess_dyn = tf.Session(graph=self.graph_dyn, config=config)
        self.path_funcs = []
        data_size = kwargs['data_size']
        up_scale = kwargs['up_scale']

        with self.graph_dyn.as_default(), tf.device('/cpu:0'):
            # define placeholders
            self.input_ph_dyn = tf.placeholder(
                dtype=tf.float32,
                shape=[None, data_size, data_size, 3]
            )
            self.gt_ph_dyn = tf.placeholder(
                dtype=tf.float32,
                shape=[None, data_size*up_scale, data_size*up_scale, 3]
            )
            self.temperature_ph = tf.placeholder(tf.float32, shape=[])
            self.is_train_ph = tf.placeholder(tf.bool, shape=[])

            self.router = None
            self.rnn_state = None

            # build model
            self.output_dyn, self.saver_test = self.build_dynamic_model()

    def build_dynamic_model(self):
        # parameters
        out_channel = self.kwargs['out_channel']
        filters = self.kwargs['filters']
        is_train = self.kwargs['is_train']
        rb_num = self.kwargs['rb_num']
        up_scale = self.kwargs['up_scale']

        # sub-graph input
        with tf.name_scope('sg_in'):
            # first conv
            conv1 = tf.layers.conv2d(
                inputs=self.input_ph_dyn,
                filters=filters,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                name='conv1',
                kernel_initializer=tf.initializers.random_normal(
                    stddev=tf.sqrt(2. / 3 / 3 / filters)
                )
            )

        # sub-graph middle (dynamic routing block)
        pi_list = []
        act_list = []
        out_list = []
        with tf.name_scope('sg_mid_dynamic'):
            res_out = conv1
            for k in range(rb_num):
                res_out, pi, act = self.build_routing_block(res_out, k)
                pi_list.append(pi)
                act_list.append(act)
                out_list.append(res_out)

        # sub-graph output
        with tf.name_scope('sg_out'):
            # last layer
            bn_last = res_out
            relu_last = tf.nn.relu(bn_last)
            output = tf.layers.conv2d(
                inputs=relu_last,
                filters=out_channel*(up_scale**2),
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                name='conv_last',
                kernel_initializer=tf.initializers.random_normal(
                    stddev=tf.sqrt(2. / 3 / 3 / filters)
                )
            )

            # up-sampling
            if up_scale > 1:
                print('Up-sampling scale: {:d}'.format(up_scale))
                output = subpixel_shuffle(output, up_scale)

        # optimizer
        with tf.name_scope('optimizer'):
            # TODO: implement loss functions
            pass

        # for testing
        psnr_out = tf.image.psnr(output, self.gt_ph_dyn, max_val=1.0)
        if not is_train:
            # standard test information
            test_info = act_list
            test_info.append(psnr_out)
            setattr(self, 'test_info', test_info)

        # summary
        with tf.name_scope('summary'):
            # TODO: implement training and validation summary writer
            pass

        # saver
        ban_list = ['beta1_power:0', 'beta2_power:0']
        with tf.name_scope('saver'):
            var_list_test = [
                v for v in tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES
                ) if v.name not in ban_list and 'Adam' not in v.name
            ]
            saver_test = tf.train.Saver(var_list=var_list_test)

        return (output, saver_test)

    def build_routing_block(self, x, block_id):
        """
        :param x: input features
        :param block_id: index of dynamic routing block
        :return: the output of dynamic routing block,
                 router out (pi), cond (a_t)
        """
        # get parameters
        num_share = self.kwargs['rb_num_share']
        filters = self.kwargs['filters']
        is_train = self.kwargs['is_train']
        num_path = self.kwargs['num_path']
        share_filter_size = self.kwargs['share_filter_size']
        share_layers = self.kwargs['share_layers']
        shared_mid_filters = self.kwargs['shared_mid_filters']
        is_router_action = self.kwargs['is_router_action']
        is_share_down_up = self.kwargs['is_share_down_up']
        is_router_parallel = self.kwargs['is_router_parallel']
        is_bn = False  # hard code, do not use BN

        # build routing block
        with tf.variable_scope('rb' + str(block_id + 1)):
            with tf.name_scope('shared'):
                s_out = x
                for k in range(num_share):
                    with tf.variable_scope('res' + str(k + 1)):
                        if share_layers == 2:
                            if is_share_down_up:
                                print('Share two layers!')
                                s_out = res_blk_down_up(
                                    s_out,
                                    filters,
                                    is_bn,
                                    is_train,
                                    stride=3
                                )
                            else:
                                s_out = res_blk(
                                    s_out,
                                    filters,
                                    is_bn,
                                    is_train,
                                    f_size=share_filter_size,
                                    mid_filters=shared_mid_filters
                                )
                        elif share_layers == 1:
                            print('Share single layer!')
                            s_out = res_blk_single_layer(
                                s_out,
                                filters,
                                is_bn,
                                is_train,
                                f_size=share_filter_size
                            )
                        else:
                            raise ValueError(
                                'Invalid shared layers {:d}'.format(
                                    share_layers
                                )
                            )

            batch_size = tf.shape(s_out)[0]
            with tf.variable_scope('router'):  # router (pathfinder)
                # fake router
                if num_path <= 2:
                    # fake router (random actions)
                    uniform_probs = tf.convert_to_tensor(
                        [1./num_path] * num_path, dtype=tf.float32
                    )
                    cond_fake = tf.distributions.Categorical(
                        probs=uniform_probs
                    ).sample(batch_size)
                else:
                    # uniform sampling
                    uniform_probs = tf.convert_to_tensor(
                        [1. / num_path] * num_path, dtype=tf.float32
                    )
                    cond_fake = tf.distributions.Categorical(
                        probs=uniform_probs
                    ).sample(batch_size)

                # true router (s_out --> cond)
                if block_id == 0:
                    # build router and initialize RNN state
                    hidden_unit = 32
                    self.router, rnn_cell = self.build_router(hidden_unit)
                    self.rnn_state = rnn_cell.zero_state(
                        batch_size,
                        dtype=tf.float32
                    )

                # whether router is parallel to shared block
                if is_router_parallel:
                    router_out, self.rnn_state = \
                        self.router(x, self.rnn_state)
                else:
                    router_out, self.rnn_state = \
                        self.router(s_out, self.rnn_state)
                cond_router = tf.cond(
                    self.is_train_ph,
                    tf.distributions.Categorical(probs=router_out).sample,
                    lambda: tf.argmax(router_out, axis=1, output_type=tf.int32)
                )

                # use router or fake distribution
                if not is_router_action:
                    cond = cond_fake
                else:
                    cond = cond_router

            with tf.name_scope('dynamic'):
                feats = []
                path_funcs = self.build_paths_binary(num_path, name='')
                for k in range(num_path):
                    mask = tf.equal(
                        cond,
                        tf.ones_like(cond, dtype=tf.int32) * k
                    )
                    if k == 0:
                        shuffle_idx = tf.where(mask)  # indices with shape k*1
                    else:
                        shuffle_idx = tf.concat(
                            [shuffle_idx, tf.where(mask)],
                            axis=0
                        )
                    feats.append(tf.boolean_mask(s_out, mask))
                    # forward feature
                    feats[-1] = path_funcs[k](feats[-1])
                # concat all features (with shuffled shape)
                rb_shuffle = tf.concat(feats, axis=0)
                # get back the ordered shape
                shuffle_idx = tf.cast(shuffle_idx, dtype=tf.int32)
                shuffle_back = tf.scatter_nd(
                    shuffle_idx,
                    tf.range(batch_size, dtype=tf.int32),
                    shape=[batch_size]
                )
                # gather is inverse operation of scatter
                rb_out = tf.gather(rb_shuffle, shuffle_back)

        return rb_out, router_out, cond

    def build_router(self, hidden_unit=32):
        """
        :param hidden_unit: hidden unit of RNN (LSTM)
        :return: a router (input: x, input state; output: output, output state)
                 and an RNN cell (in the router)
        """
        # get parameters
        filters = self.kwargs['filters']
        num_path = self.kwargs['num_path']
        data_size = self.kwargs['data_size']
        is_train = self.kwargs['is_train']
        router_conv_filters = self.kwargs['router_conv_filters']

        # define variables for convolution layers
        weights = []
        biases = []
        conv_filters = [filters] + router_conv_filters

        if len(conv_filters) == 5:  # stride=2
            for k in range(len(conv_filters) - 1):
                pre_filters = conv_filters[k]
                cur_filters = conv_filters[k+1]
                w = tf.get_variable(
                    'w' + str(k+1),
                    shape=[3, 3, pre_filters, cur_filters],
                    dtype=tf.float32,
                    initializer=tf.initializers.random_normal(
                        stddev=tf.sqrt(2. / 3 / 3 / cur_filters)
                    )
                )
                b = tf.get_variable(
                    'b' + str(k+1),
                    shape=[cur_filters]
                )
                weights.append(w)
                biases.append(b)
        elif len(conv_filters) == 3:  # stride=4
            for k in range(len(conv_filters) - 1):
                pre_filters = conv_filters[k]
                cur_filters = conv_filters[k+1]
                w = tf.get_variable(
                    'w' + str(k+1),
                    shape=[5, 5, pre_filters, cur_filters],
                    dtype=tf.float32,
                    initializer=tf.initializers.random_normal(
                        stddev=tf.sqrt(2. / 5 / 5 / cur_filters)
                    )
                )
                b = tf.get_variable(
                    'b' + str(k+1),
                    shape=[cur_filters]
                )
                weights.append(w)
                biases.append(b)
        else:
            raise ValueError('Invalid conv filters {}.'.format(conv_filters))

        # define variables for fc layers
        # hard code, down-sample 16x
        feat_size = math.ceil(data_size / 16.)
        linear_input = feat_size * feat_size * conv_filters[-1]
        hidden_unit = min(linear_input, hidden_unit)
        linear_size = [linear_input, hidden_unit]
        linear_ws = []
        linear_bs = []
        for k in range(len(linear_size) - 1):
            in_size = linear_size[k]
            out_size = linear_size[k+1]
            w = tf.get_variable(
                'fc_w' + str(k+1),
                shape=[in_size, out_size],
                dtype=tf.float32,
                initializer=tf.initializers.random_normal(stddev=0.001)
            )
            b = tf.get_variable(
                'fc_b' + str(k+1),
                shape=[out_size]
            )
            linear_ws.append(w)
            linear_bs.append(b)

        # define RNN cell
        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(
            num_units=hidden_unit,
            state_is_tuple=True,
            reuse=tf.AUTO_REUSE
        )

        # output fc layer
        out_w = tf.get_variable(
            'out_w',
            shape=[hidden_unit, num_path],
            dtype=tf.float32,
            initializer=tf.initializers.random_normal(stddev=0.001)
        )
        out_b = tf.get_variable(
            'out_b',
            shape=[num_path]
        )

        # define the function of router
        def router_func(x, input_state):
            # convolution
            conv_out = x
            for k in range(len(weights)):
                w, b = weights[k], biases[k]
                if len(conv_filters) == 5:
                    conv_out = conv_relu_with_param(conv_out, w, b, 2)
                elif len(conv_filters) == 3:
                    conv_out = conv_relu_with_param(conv_out, w, b, 4)
                else:
                    raise ValueError('Invalid conv filters.')

            # fc
            conv_flat = tf.reshape(conv_out, [-1, linear_size[0]])
            fc_out = conv_flat
            for k in range(len(linear_ws)):
                w, b = linear_ws[k], linear_bs[k]
                fc_out = linear_relu_with_param(fc_out, w, b)

            # RNN
            # batch size * step length * input dimension
            rnn_input = tf.reshape(fc_out, [-1, 1, linear_size[-1]])
            rnn_output, output_state = tf.nn.dynamic_rnn(
                cell=rnn_cell,
                inputs=rnn_input,
                initial_state=input_state
            )

            # output fc
            rnn_out = tf.reshape(rnn_output, [-1, hidden_unit])
            output = linear_with_param(rnn_out, out_w, out_b)

            # categorical distribution
            if is_train:
                output = tf.nn.softmax(output / self.temperature_ph)
            else:
                output = tf.nn.softmax(output)

            return output, output_state

        return router_func, rnn_cell

    def build_paths_binary(self, num_path, name):
        """
        :param num_path: No. of paths
        :param name: prefix of path name
        :return: self.path_funcs = a list of path functions
                 (capacity = [0, 1, 1, 1 ...])
        """
        path_funcs = []
        filters = self.kwargs['filters']
        share_layers = self.kwargs['share_layers']
        num_blk = share_layers // 2 if share_layers > 2 else 1
        for k in range(num_path):
            key_list = []
            var_list = []
            for m in range(1, min(k+1, 2)):  # at most 1 residual block
                with tf.variable_scope(
                    name + 'path' + str(k) + '_blk' + str(m)
                ):
                    # define parameters for each path, each block
                    w1 = tf.get_variable(
                        'w1',
                        shape=[3, 3, filters, filters],
                        dtype=tf.float32,
                        initializer=tf.initializers.random_normal(
                            stddev=tf.sqrt(2. / 3 / 3 / filters)
                        )
                    )
                    w2 = tf.get_variable(
                        'w2',
                        shape=[3, 3, filters, filters],
                        dtype=tf.float32,
                        initializer=tf.initializers.random_normal(
                            stddev=tf.sqrt(2. / 3 / 3 / filters)
                        )
                    )
                    b1 = tf.get_variable('b1', shape=[filters])
                    b2 = tf.get_variable('b2', shape=[filters])
                    str_m = str(m)
                    key_list += [
                        'w'+str_m+'1',
                        'w'+str_m+'2',
                        'b'+str_m+'1',
                        'b'+str_m+'2'
                    ]
                    var_list += [w1, w2, b1, b2]

            params = dict(zip(key_list, var_list))
            path_funcs.append(
                path_res_blks(params, num_blk=min(k, 1) * num_blk)
            )
        return path_funcs

    def initialize_dynamic(self):
        # initialize dynamic model
        load_dir_dyn = self.kwargs['load_dir_dyn']
        load_which = self.kwargs['load_which']
        if load_which == 'restorer':
            self.saver_init_dyn.restore(self.sess_dyn, load_dir_dyn)
        elif load_which == 'router':
            self.saver_init_router.restore(self.sess_dyn, load_dir_dyn)
        elif load_which == 'all':
            self.saver_init_all.restore(self.sess_dyn, load_dir_dyn)
        else:
            raise ValueError('Invalid load_which: {}'.format(load_which))

    def train_dynamic(self):
        """ TODO: implement training """
        pass
