from keras.layers import Input, Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten,Lambda,Activation
from keras.layers.core import Dropout
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras import backend as K

def build_custom_cnn(n_freq_cnn=128, n_frames_cnn=50, n_filters_cnn=64,
                     filter_size_cnn=(5, 5), pool_size_cnn=(2,2),
                     n_classes=10, large_cnn=False, n_dense_cnn=64):

    """
    Function that returns the S-CNN [1] model.
    
    [1] J. Salamon, D. MacConnell, M. Cartwright, P. Li and J. P. Bello 
    "Scaper: A Library for Soundscape Synthesis and Augmentation", 
    IEEE Workshop on Applications of Signal Processing to Audio and Acoustics.
    
    ----------
    n_freq_cnn : int
        number of frecuency bins of the input
    
    n_frames_cnn : int
        number of time steps (hops) of the input
    
    n_filters_cnn : int
        number of filter in each conv layer
    
    filter_size_cnn : tuple of int
        kernel size of each conv filter
        
    pool_size_cnn : tuple of int
        kernel size of the pool operations
        
    n_classes : int
        number of classes for the classification taks 
        (size of the last layer)
        
    large_cnn : bool
        If true, the model has one more dense layer
        
    n_dense_cnn : int
        Size of middle dense layers
    Returns
    -------
    m : Model
        Keras class Model with the designed S-CNN.

    Notes
    -----
    Code based on Salamon's implementation 
    https://github.com/justinsalamon/scaper_waspaa2017
    
    """

    if large_cnn:
        n_filters_cnn = 128
        n_dense_cnn = 128

    # INPUT
    x = Input(shape=(n_frames_cnn,n_freq_cnn), dtype='float32')
    
    # Add Batchnorm to do the feature normalization
    y = BatchNormalization()(x)
    
    y = Lambda(lambda x: K.expand_dims(x,-1))(y) 
    
    # CONV 1
    y = Conv2D(n_filters_cnn, filter_size_cnn, padding='valid',
               activation='relu')(y)
    y = MaxPooling2D(pool_size=pool_size_cnn, strides=None, padding='valid')(y)
    y = BatchNormalization()(y)

    # CONV 2
    y = Conv2D(n_filters_cnn, filter_size_cnn, padding='valid',
               activation='relu')(y)
    y = MaxPooling2D(pool_size=pool_size_cnn, strides=None, padding='valid')(y)
    y = BatchNormalization()(y)

    # CONV 3
    y = Conv2D(n_filters_cnn, filter_size_cnn, padding='valid',
               activation='relu')(y)
    # y = MaxPooling2D(pool_size=pool_size_cnn, strides=None, padding='valid')(y)
    y = BatchNormalization()(y)

    # Flatten for dense layers
    y = Flatten()(y)
    y = Dropout(0.5)(y)
    y = Dense(n_dense_cnn, activation='relu')(y)
    if large_cnn:
        y = Dropout(0.5)(y)
        y = Dense(n_dense_cnn, activation='relu')(y)
    y = Dropout(0.5)(y)
    y = Dense(n_classes, activation='sigmoid')(y)

    m = Model(inputs=x, outputs=y)

    return m

def cnn_fine_tuned(n_freq_cnn=128, n_frames_cnn=50, large_cnn=False,
                    n_classes1=2, n_classes2=2, n_classes3=2,
                    model_baseline=None):

    """
    Function that returns the S-CNN [1] fine tuned model.
    
    [1] J. Salamon, D. MacConnell, M. Cartwright, P. Li and J. P. Bello 
    "Scaper: A Library for Soundscape Synthesis and Augmentation", 
    IEEE Workshop on Applications of Signal Processing to Audio and Acoustics.
    
    ----------
    n_freq_cnn : int
        number of frecuency bins of the input
    
    n_frames_cnn : int
        number of time steps (hops) of the input
    
    large_cnn : bool
        If true, the model has one more dense layer
        
    n_classes1 : int
        Number of classes in level 1

    n_classes2 : int
        Number of classes in level 3
        
    n_classes3 : int
        Number of classes in level 3
        
    model_baseline : string
        Path to the file weights of the baseline model.
        If None, no weights are loaded

    Returns
    -------
    m : Model
        Keras class Model with the fine tuned S-CNN.
    
    """

    model = build_custom_cnn(n_freq_cnn=n_freq_cnn, n_frames_cnn=n_frames_cnn,large_cnn=large_cnn)

    if model_baseline is not None:
        model.load_weights(model_baseline)
    
    x = Dense(32, activation='relu')(model.layers[-2].output)

    o1 = Dense(n_classes1, activation='sigmoid')(x)
    o2 = Dense(n_classes2, activation='sigmoid')(o1)
    o3 = Dense(n_classes3, activation='sigmoid')(o2)

    model = Model(inputs=model.input, outputs=[o1,o2,o3])
    
    return model