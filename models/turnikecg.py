import tensorflow.keras as keras
from tensorflow.keras.models import Model
# from keras.models import Sequential, load_model
# from keras.optimizers import SGD, Adam
import tensorflow.keras.layers as layers


def _residual_block(x, kernel_size, layer_name, conv_filts, res_filts, skip_filts, dilation_rate, res=True, skip=True):
    # Outputs dictionary
    outputs = dict()

    # Convolution tanh
    conv_filt = layers.Conv1D(filters=conv_filts,
                       kernel_size=kernel_size,
                       padding='same',
                       strides=1,
                       dilation_rate=dilation_rate,
                       activation='tanh',
                       use_bias=False,
                       name=layer_name + '_conv_filt')(x)

    # Convolution sigmoid
    conv_gate = layers.Conv1D(filters=conv_filts,
                       kernel_size=kernel_size,
                       padding='same',
                       strides=1,
                       dilation_rate=dilation_rate,
                       activation='sigmoid',
                       use_bias=False,
                       name=layer_name + '_conv_gate')(x)

    activation = layers.Multiply()([conv_filt, conv_gate])

    # Residual
    if res:
        # Convolution
        outputs['res'] = layers.Conv1D(
            kernel_size=1,
            strides=1,
            dilation_rate=1,
            filters=res_filts,
            padding='same',
            name=layer_name + '_conv_res',
        )(activation)

        outputs['res'] = layers.Add(name=layer_name + '_add_identity')([outputs['res'], x])

    # Skip
    if skip:
        # Convolution
        outputs['skip'] = layers.Conv1D(
            kernel_size=1,
            strides=1,
            dilation_rate=1,
            filters=skip_filts,
            padding='same',
            name=layer_name + '_conv_skip',
        )(activation)

    return outputs


def Turnikv7( input_shape, n_classes ):

    kernel_size = 3
    conv_filts = 256
    res_filts = 256
    skip_filts = 256
    skips = list()

    input_layer = layers.Input(input_shape)

    """BatchNorm"""
    bn = layers.BatchNormalization()(input_layer)

    """Block Series 1"""
    # --- Layer 1 (Convolution) ------------------------------------------------------------------------------ #
    # Set name
    layer_name = 'layer_1'

    net = layers.Conv1D(filters=res_filts,
                    kernel_size=kernel_size,
                    padding='same',
                    strides=1,
                    dilation_rate=1,
                    use_bias=False,
                    name=layer_name + '_conv'
                    )(bn)

    # --- Layer 2 (Convolution) ------------------------------------------------------------------------------ #

    # Set name
    layer_name = 'layer_2'

    # Compute block
    outputs = _residual_block(x=net, kernel_size=kernel_size, layer_name=layer_name,
                                   conv_filts=conv_filts, res_filts=res_filts, skip_filts=skip_filts,
                                   dilation_rate=2, res=True, skip=True)

    # Collect skip
    skips.append(outputs['skip'])

    # --- Layer 3 (Convolution) ------------------------------------------------------------------------------ #

    # Set name
    layer_name = 'layer_3'

    # Compute block
    outputs = _residual_block(x=outputs['res'], kernel_size=kernel_size, layer_name=layer_name,
                                   conv_filts=conv_filts, res_filts=res_filts, skip_filts=skip_filts,
                                   dilation_rate=4, res=True, skip=True)

    # Collect skip
    skips.append(outputs['skip'])

    # --- Layer 4 (Convolution) ------------------------------------------------------------------------------ #

    # Set name
    layer_name = 'layer_4'

    # Compute block
    outputs = _residual_block(x=outputs['res'], kernel_size=kernel_size, layer_name=layer_name,
                                   conv_filts=conv_filts, res_filts=res_filts, skip_filts=skip_filts,
                                   dilation_rate=8, res=True, skip=True)

    # Collect skip
    skips.append(outputs['skip'])

    #--- Layer 5 (Convolution) ------------------------------------------------------------------------------ #

    # Set name
    layer_name = 'layer_5'

    # Compute block
    outputs = _residual_block(x=outputs['res'], kernel_size=kernel_size, layer_name=layer_name,
                                   conv_filts=conv_filts, res_filts=res_filts, skip_filts=skip_filts,
                                   dilation_rate=16, res=True, skip=True)

    # Collect skip
    skips.append(outputs['skip'])

    #--- Layer 6 (Convolution) ------------------------------------------------------------------------------ #

    # Set name
    layer_name = 'layer_6'

    # Compute block
    outputs = _residual_block(x=outputs['res'], kernel_size=kernel_size, layer_name=layer_name,
                                   conv_filts=conv_filts, res_filts=res_filts, skip_filts=skip_filts,
                                   dilation_rate=32, res=True, skip=True)

    # Collect skip
    skips.append(outputs['skip'])

    # --- Layer 7 (Convolution) ------------------------------------------------------------------------------ #

    # Set name
    layer_name = 'layer_7'

    # Compute block
    outputs = _residual_block(x=outputs['res'], kernel_size=kernel_size, layer_name=layer_name,
                                   conv_filts=conv_filts, res_filts=res_filts, skip_filts=skip_filts,
                                   dilation_rate=64, res=True, skip=True)

    # Collect skip
    skips.append(outputs['skip'])

    # --- Layer 8 (Convolution) ------------------------------------------------------------------------------ #

    # Set name
    layer_name = 'layer_8'

    # Compute block
    outputs = _residual_block(x=outputs['res'], kernel_size=kernel_size, layer_name=layer_name,
                                   conv_filts=conv_filts, res_filts=res_filts, skip_filts=skip_filts,
                                   dilation_rate=128, res=True, skip=True)

    # Collect skip
    skips.append(outputs['skip'])

    # --- Layer 9 (Convolution) ------------------------------------------------------------------------------ #

    # Set name
    layer_name = 'layer_9'

    # Compute block
    outputs = _residual_block(x=outputs['res'], kernel_size=kernel_size, layer_name=layer_name,
                                   conv_filts=conv_filts, res_filts=res_filts, skip_filts=skip_filts,
                                   dilation_rate=256, res=True, skip=True)

    # Collect skip
    skips.append(outputs['skip'])

    # --- Layer 10 (Convolution) ----------------------------------------------------------------------------- #

    # Set name
    layer_name = 'layer_10'

    # Compute block
    outputs = _residual_block(x=outputs['res'], kernel_size=kernel_size, layer_name=layer_name,
                                   conv_filts=conv_filts, res_filts=res_filts, skip_filts=skip_filts,
                                   dilation_rate=512, res=False, skip=True)

    # Collect skip
    skips.append(outputs['skip'])

    output = layers.Add()(skips)

    # Activation
    #output = tf.nn.relu(output)
    output = layers.ReLU()(output)


    # Dropout
    output = layers.Dropout(rate=0.3, name='dropout1')(output)

    # Convolution
    output = layers.Conv1D(kernel_size=kernel_size,
                                 strides=1,
                                 dilation_rate=1,
                                 filters=256,
                                 padding='same',
                                 activation='relu',
                                 name='conv1')(output)

    # Dropout
    output = layers.Dropout(rate=0.3, name='dropout2')(output)

    # Convolution
    output = layers.Conv1D(kernel_size=kernel_size,
                                 strides=1,
                                 dilation_rate=1,
                                 filters=512,
                                 padding='same',
                                 activation='relu',
                                 name='conv2')(output)

    # Dropout
    output = layers.Dropout(rate=0.3, name='dropout3')(output)

    """Network Output"""
    # --- Global Average Pooling Layer ----------------------------------------------------------------------- #

    # Set name
    layer_name = 'gap'

    gap_layer = layers.GlobalAveragePooling1D(name=layer_name)(output)


    output_layer = layers.Dense(n_classes, activation='softmax')(gap_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])


    # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)
    #
    # file_path = self.output_directory + 'best_model.hdf5'
    #
    # model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
    #                                                    save_best_only=True)

    # self.callbacks = [reduce_lr, model_checkpoint]

    return model
