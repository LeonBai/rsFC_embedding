## This code snippet contains the core code in training CVAE with embedded clustering concept
## The dataset is provided explicitly, see ReadMe for more info.

from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow.keras.layers import Conv2D, Flatten, Lambda
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse, log_cosh,mean_squared_logarithmic_error
from tensorflow.keras import backend as K
from tensorflow import keras
import tensorflow as tf
import tensorflow as tf
import tensorflow_addons as tfa

###################  Defined Functions   ###################

## Reparameterisation trick, follows conventional VAE; computing z;
## this part is directly adopted from the KERAS official example


### Gaussian manifold case
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.shape(z_mean)[1]

    epsilon = K.random_normal(shape = (batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


### vMF manifold case (see SI S 1.1.2)

def sampling(mu):

    dims = K.int_shape(mu)[-1]
    epsilon = 1e-7
    x = np.arange(-1 + epsilon, 1, epsilon)
    y = kappa * x + np.log(1 - x**2) * (dims - 3) / 2
    y = np.cumsum(np.exp(y - y.max()))
    y = y / y[-1]
    W = K.constant(np.interp(np.random.random(10**6), y, x))
    idx = K.random_uniform(K.shape(mu[:, :1]), 0, 10**6, dtype='int32')
    w = K.gather(W, idx)
    eps = K.random_normal(K.shape(mu))
    nu = eps - K.sum(eps * mu, axis=1, keepdims=True) * mu
    nu = K.l2_normalize(nu, axis=-1)
    return w * mu + (1 - w**2)**0.5 * nu


## Defined gaussian layer to get the z-\mu_c in the main text
## Least to Mediam amount of dianostic information

class Gaussian(Layer):

    def __init__(self, num_classes, **kwargs):
        self.num_classes = num_classes
        super(Gaussian, self).__init__(**kwargs)
    def build(self, input_shape):
        latent_dim = input_shape[-1]
        self.mean = self.add_weight(name='mean',
                                    shape=(self.num_classes, latent_dim),
                                    initializer='zeros')
    def call(self, inputs):
        z = inputs # z.shape=(batch_size, latent_dim)
        z = K.expand_dims(z, 1)
        return z - K.expand_dims(self.mean, 0)
    def compute_output_shape(self, input_shape):
        return (None, self.num_classes, input_shape[-1])

## explicit contrastive learning, computing \lambda_clr in our main text
### Most diagnostic information
class SupervisedContrastiveLoss(keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)

### Normalise the encoded z to be Normal distributed between [0, 1]
class Scaler(Layer):


    def __init__(self, tau=0.5, **kwargs):
        super(Scaler, self).__init__(**kwargs)
        self.tau = tau

    def build(self, input_shape):
        super(Scaler, self).build(input_shape)
        self.scale = self.add_weight(
            name='scale', shape=(input_shape[-1],), initializer='zeros'
        )

    def call(self, inputs, mode='positive'):
        if mode == 'positive':
            scale = self.tau + (1 - self.tau) * K.sigmoid(self.scale)
        else:
            scale = (1 - self.tau) * K.sigmoid(-self.scale)
        return inputs * K.sqrt(scale)

    def get_config(self):
        config = {'tau': self.tau}
        base_config = super(Scaler, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

######################################

###################  Networks; see Table 1 in Supplementary Material ######################

### Network-1 ####

'''
This network-1 is a encoder with scaling the encoded z to be either Normal distributed or vMF distributed

'''

input_shape = (9730, )
batch_size = 128
#kernel_size = 3
#filters = 16
latent_dim = 2

x_inputs = Input(input_shape)

xx_1 = x_inputs
x_1 = Dense(2000, activation='relu')(x_1)
x_1 = Dense(200, activation='relu')(x_1)
#x_1 = Dense(100, activation='relu')x_1)

scaler = Scaler()
z_mean_1 = Dense(latent_dim, name='z_mean_1')(x_1)
z_mean_1 = BatchNormalization(scale=False, center=False, epsilon=1e-8)(z_mean_1)
z_mean_1 = scaler(z_mean_1, mode='positive')

z_log_var_1 = Dense(latent_dim, name='z_log_var_1')(x_1)
z_log_var_1 = BatchNormalization(scale=False, center=False, epsilon=1e-8)(z_log_var_1)
z_log_var_1 = scaler(z_log_var_1, mode='negative')


# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
#"""
### Gaussian manifold; Comment this if you choose vMF manifold
z_1 = Lambda(sampling, output_shape=(latent_dim,), name='z_1')([z_mean_1, z_log_var_1])
#"""


"""
### Uncomment this section for vMF distribution and comment the previous Gaussian section

mu = Dense(latent_dim)(x_1)
mu = Lambda(lambda x: K.l2_normalize(x, axis=-1))(mu)
z_mean_1 = Lambda(sampling)(z_mean_1)
z_log_var_1 = Lambda(sampling)(z_log_var_1)

"""
encoder_1 = Model(x_inputs, [z_mean_1, z_log_var_1, z_1], name='encoder_1')

### Network-2 ####


'''
This network-2 is to compute the p(c) and q(c|z)

'''

z = Input(shape=(2,1))
y = Dense(40, activation = 'relu')(z)
y = Dense(3, activation = 'softmax')(y)

classifier = Model(z, y)
y = classifier(z_1)

gaussian = Gaussian(3)
z_prior_mean = gaussian(z_1)


### Network-3 ####

'''
This network-3 is a contrastive learning component to embed the diagnostic difference,
see 1.2.1 and 1.2.2 in the SI.

'''

encoder_with_projection_head = Input(shape=(2,))
encoder_with_projection_head_outputs = Dense(40, activation = 'relu')(encoder_with_projection_head)
encoder_with_projection_model = Model(encoder_with_projection_head,
                                     encoder_with_projection_head_outputs,
                                     name = 'Con')


### Common decoder ####
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(100, activation='relu')(latent_inputs)
x = Dense(200, activation='relu')(x)
x = Dense(2000, activation='relu')(x)

recon = Dense(9730, activation='sigmoid')(x)
# instantiate decoder model
decoder = Model(latent_inputs, recon, name='decoder')
x_recon = decoder(z_1)
generator = decoder

######################################

inputs = [x_inputs, y_true]

outputs = [x_recon, z_prior_mean, y, yh_true, encoder_with_projection_model(encoder_1(x_inputs)[0])]

model = Model(inputs, outputs)

###################  Defined Loss functions ######################
z_mean = K.expand_dims(z_mean_1, 1)
z_log_var = K.expand_dims(z_log_var_1, 1)
yh_true_ = K.expand_dims(yh_true, 1)

temperature = 1

### Reconstruction error
xent_loss = 0.5 * K.mean((x_inputs - x_recon)**2, 0)

### Pairwise difference between the original and embeddings D = (dxij - ||zi - zj||)^2

na = tf.reduce_sum(tf.square(x_inputs), 1)
nb = tf.reduce_sum(tf.square(x_recon), 1)
na = tf.reshape(na, [-1, 1])
nb = tf.reshape(nb, [1, -1])

# return pairwise euclidead difference matrix
D = K.sum(tf.sqrt(tf.maximum(na - 2*tf.matmul(x_inputs,x_recon, False, True) + nb, 0.0)),axis = -1)


### Least diagnostic information related KL loss
kl_loss = - 0.5 * (z_log_var - K.square(z_prior_mean) )
kl_loss = K.mean(K.batch_dot(K.expand_dims(y, 1), kl_loss), 0)

# Median diagnostic information related KL loss
kl_loss_2 = - 0.5 * K.sum(1 + z_log_var_1 - K.square(z_mean_1 - yh_true) - K.exp(z_log_var_1), axis=-1)
cat_loss = K.mean(y * K.log(y + K.epsilon()), 0)
#cat_loss = 0

#### Total loss can be self defined for options on amount of diagnostic information {least, median, most}
####  + whether or not include pairwise difference D,
####  + reconstruction error (the same for both Gaussian and vMF embeddings)

### Here is the example of the loss function for deriving Gaussian embeddings, with pairwise difference and the most dignostic embedding option

total_loss = 0.1 * K.mean((1*K.sum(xent_loss) +
                           K.sum(kl_loss) + D
                        )/100.0)#K.mean(kl_loss)

model.add_loss(total_loss)

### Most diagnostic information loss, i.e., the added contrastive functions
### Comment this for least and median diagnostic options.
model.compile(optimizer='adam', loss = {'Con':SupervisedContrastiveLoss()})

### For least and median dignoistic information options, use and uncomment the following loss terms
#vae.compile(optimizer='adam', loss = {'Con':SupervisedContrastiveLoss()})
######################################
###################  Training configurations see Table 2 in SI ######################

model.fit(X_train,{'Con':y_train},
                       epochs=20,
                       batch_size=20,shuffle=True,
                      verbose=2,validation_split=0.1)

#########################################
