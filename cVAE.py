##import numpy as np
##from keras.layers import Input, Dense, Lambda
##from keras.layers.merge import concatenate
##from keras.models import Model
##from keras import backend as K
##from keras.datasets import mnist
##from keras.utils import to_categorical
##from keras.callbacks import EarlyStopping
##from scipy.misc import imsave
##import os
##import matplotlib.pyplot as plt
##
##def load_data_from_file(file_dir):
##    # returns: train_images , train_labels 
##    data = np.load(file_dir)
##    return data['arr_0'], data['arr_1']
##
##
##
##print('### my_modified_EMNIST ###')
##path = './font_EMNIST'
##print(os.listdir(path)[0])
##x_train_test, y_train_test = load_data_from_file(path+'/'+os.listdir(path)[0])
##
##test_data_size = x_train_test.shape[0]//6 # 1:6 ratio
##X_train = []
##y_train = []
##X_test = []
##y_test = []
##for idx in range(x_train_test.shape[0]):
##    if (idx <= test_data_size):
##        X_test.append(x_train_test[idx])
##        y_test.append(y_train_test[idx])
##    else:
##        X_train.append(x_train_test[idx])
##        y_train.append(y_train_test[idx])
##
##X_train = np.asarray(X_train, dtype=np.float32)
##y_train = np.asarray(y_train, dtype=np.float32)
##X_test = np.asarray(X_test, dtype=np.float32)
##y_test = np.asarray(y_test, dtype=np.float32)
##
##
####print(X_train)
####print(y_train.shape)
####print(X_test.shape)
####print(y_test.shape)
####
####plt.imshow(X_train[97]*-1., cmap='gray')
####print(y_train[97])
####plt.show()
####
####plt.imshow(X_test[37]*-1., cmap='gray')
####print(y_test[37])
####plt.show()
##
##
##
### convert y to one-hot, reshape x
##y_train = to_categorical(y_train)
##y_test = to_categorical(y_test)
##X_train = X_train.astype('float32') / 255.
##X_test = X_test.astype('float32') / 255.
##X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
##X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
##
##print(X_train.shape)
##print(y_train.shape)
##print(X_test.shape)
##print(y_test.shape)
##
### select optimizer
##optim = 'adam'
##
### dimension of latent space (batch size by latent dim)
##m = 50
##n_z = 50
##
### dimension of input (and label)
##n_x = X_train.shape[1]
##n_y = y_train.shape[1]
##
### nubmer of epochs
##n_epoch = 10
##
####  ENCODER ##
##
### encoder inputs
##X = Input(shape=(784, ))
##cond = Input(shape=(n_y, ))
##
### merge pixel representation and label
##inputs = concatenate([X, cond])
##
### dense ReLU layer to mu and sigma
##h_q = Dense(512, activation='relu')(inputs)
##mu = Dense(n_z, activation='linear')(h_q)
##log_sigma = Dense(n_z, activation='linear')(h_q)
##
##def sample_z(args):
##    mu, log_sigma = args
##    eps = K.random_normal(shape=(m, n_z), mean=0., stddev=1.)
##    return mu + K.exp(log_sigma / 2) * eps
##
##
### Sampling latent space
##z = Lambda(sample_z, output_shape = (n_z, ))([mu, log_sigma])
##
### merge latent space with label
##z_cond = concatenate([z, cond])
##
####  DECODER  ##
##
### dense ReLU to sigmoid layers
##decoder_hidden = Dense(512, activation='relu')
##decoder_out = Dense(784, activation='sigmoid')
##h_p = decoder_hidden(z_cond)
##outputs = decoder_out(h_p)
##
### define cvae and encoder models
##cvae = Model([X, cond], outputs)
##encoder = Model([X, cond], mu)
##
### reuse decoder layers to define decoder separately
##d_in = Input(shape=(n_z+n_y,))
##d_h = decoder_hidden(d_in)
##d_out = decoder_out(d_h)
##decoder = Model(d_in, d_out)
##
### define loss (sum of reconstruction and KL divergence)
##def vae_loss(y_true, y_pred):
##    # E[log P(X|z)]
##    recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
##    # D_KL(Q(z|X) || P(z|X))
##    kl = 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)
##    return recon + kl
##
##def KL_loss(y_true, y_pred):
##    return(0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1))
##
##def recon_loss(y_true, y_pred):
##    return(K.sum(K.binary_crossentropy(y_pred, y_true), axis=1))
##
##
### compile and fit
##cvae.compile(optimizer=optim, loss=vae_loss, metrics = [KL_loss, recon_loss])
##cvae_hist = cvae.fit([X_train, y_train], X_train, batch_size=m, epochs=n_epoch,
##							validation_data = ([X_test, y_test], X_test),
##							callbacks = [EarlyStopping(patience = 5)])
##
### this loop prints the one-hot decodings
##
###for i in range(n_z+n_y):
###	tmp = np.zeros((1,n_z+n_y))
###	tmp[0,i] = 1
###	generated = decoder.predict(tmp)
###	file_name = './img' + str(i) + '.jpg'
###	print(generated)
###	imsave(file_name, generated.reshape((28,28)))
###	sleep(0.5)
##
### this loop prints a transition through the number line
##
##pic_num = 0
##variations = 30 # rate of change; higher is slower
##for j in range(n_z, n_z + n_y - 1):
##    for k in range(variations):
##        v = np.zeros((1, n_z+n_y))
##        v[0, j] = 1 - (k/variations)
##        v[0, j+1] = (k/variations)
##        generated = decoder.predict(v)
##        pic_idx = j - n_z + (k/variations)
##        file_name = './transition_50/img{0:.3f}.jpg'.format(pic_idx)
##        imsave(file_name, generated.reshape((28,28)))
##        pic_num += 1
##



import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data
from numpy import array
from numpy import argmax
from keras.utils import to_categorical


class DataSet(object):

    def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

    @property
    def images(self):
    return self._images

    @property
    def labels(self):
    return self._labels

    @property
    def num_examples(self):
    return self._num_examples

    @property
    def epochs_completed(self):
    return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
      if fake_data:
        fake_image = [1] * 784
        if self.one_hot:
          fake_label = [1] + [0] * 9
        else:
          fake_label = 0
        return [fake_image for _ in xrange(batch_size)], [
            fake_label for _ in xrange(batch_size)
        ]
      start = self._index_in_epoch
      self._index_in_epoch += batch_size
      if self._index_in_epoch > self._num_examples:
        # Finished epoch
        self._epochs_completed += 1
        # Shuffle the data
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self._images[perm]
        self._labels = self._labels[perm]
        # Start next epoch
        start = 0
        self._index_in_epoch = batch_size
        assert batch_size <= self._num_examples
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]

    def load_data_from_file(self,file_dir):
        # returns: train_images , train_labels 
        data = np.load(file_dir)
        return data['arr_0'], data['arr_1']
    
    def my_dataset(self):
        print('### my_modified_EMNIST ###')
        path = './font_EMNIST'
        print(os.listdir(path)[0])
        x_train_test, y_train_test = self.load_data_from_file(path+'/'+os.listdir(path)[0])

        test_data_size = x_train_test.shape[0]//6 # 1:6 ratio
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        for idx in range(x_train_test.shape[0]):
         if (idx <= test_data_size):
             X_test.append(x_train_test[idx])
             y_test.append(y_train_test[idx])
         else:
             X_train.append(x_train_test[idx])
             y_train.append(y_train_test[idx])

        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32)
        y_test = np.asarray(y_test, dtype=np.float32)

        return (X_train,y_train),(X_test,y_test)


    def read_data_sets(train_dir,
                       fake_data=False,
                       one_hot=False,
                       dtype=dtypes.float32,
                       reshape=True,
                       validation_size=5000):
      if fake_data:

        def fake():
          return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)

        train = fake()
        validation = fake()
        test = fake()
        return base.Datasets(train=train, validation=validation, test=test)

      (train_images, train_labels), (test_images, test_labels) = self.my_dataset()
      train_labels = to_categorical(train_labels)
      test_labels = to_categorical(test_labels)

      if not 0 <= validation_size <= len(train_images):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'
            .format(len(train_images), validation_size))

      validation_images = train_images[:validation_size]
      validation_labels = train_labels[:validation_size]
      train_images = train_images[validation_size:]
      train_labels = train_labels[validation_size:]

      train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
      validation = DataSet(validation_images,
                           validation_labels,
                           dtype=dtype,
                           reshape=reshape)
      test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape)

      return base.Datasets(train=train, validation=validation, test=test)


# =================================== main code ========================================

def load_data_from_file(file_dir):
    # returns: train_images , train_labels 
    data = np.load(file_dir)
    return data['arr_0'], data['arr_1']

print('### my_modified_EMNIST ###')
path = './font_EMNIST'
print(os.listdir(path)[0])
x_train_test, y_train_test = load_data_from_file(path+'/'+os.listdir(path)[0])
y_train = to_categorical(y_train_test)
x_train = x_train_test.reshape(*x_train_test.shape[:1], -1) / 255

##mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 64
z_dim = 100
##X_dim = mnist.train.images.shape[1]
##y_dim = mnist.train.labels.shape[1]
X_dim = x_train.shape[1]
y_dim = y_train.shape[1]
h_dim = 128
c = 0
lr = 1e-3




##print(x_train[0])
##print(y_train[0])
##
##print(mnist.train.images[0])
##print(mnist.train.labels[0])




def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# =============================== Q(z|X) ======================================

X = tf.placeholder(tf.float32, shape=[None, X_dim])
c = tf.placeholder(tf.float32, shape=[None, y_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

Q_W1 = tf.Variable(xavier_init([X_dim + y_dim, h_dim]))
Q_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

Q_W2_mu = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2_mu = tf.Variable(tf.zeros(shape=[z_dim]))

Q_W2_sigma = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2_sigma = tf.Variable(tf.zeros(shape=[z_dim]))


def Q(X, c):
    inputs = tf.concat(axis=1, values=[X, c])
    h = tf.nn.relu(tf.matmul(inputs, Q_W1) + Q_b1)
    z_mu = tf.matmul(h, Q_W2_mu) + Q_b2_mu
    z_logvar = tf.matmul(h, Q_W2_sigma) + Q_b2_sigma
    return z_mu, z_logvar


def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * eps


# =============================== P(X|z) ======================================

P_W1 = tf.Variable(xavier_init([z_dim + y_dim, h_dim]))
P_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

P_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
P_b2 = tf.Variable(tf.zeros(shape=[X_dim]))


def P(z, c):
    inputs = tf.concat(axis=1, values=[z, c])
    h = tf.nn.relu(tf.matmul(inputs, P_W1) + P_b1)
    logits = tf.matmul(h, P_W2) + P_b2
    prob = tf.nn.sigmoid(logits)
    return prob, logits


# =============================== TRAINING ====================================

z_mu, z_logvar = Q(X, c)
z_sample = sample_z(z_mu, z_logvar)
_, logits = P(z_sample, c)

# Sampling from random z
X_samples, _ = P(z, c)

# E[log P(X|z)]
recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X), 1)
# D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
# VAE loss
vae_loss = tf.reduce_mean(recon_loss + kl_loss)

solver = tf.train.AdamOptimizer().minimize(vae_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

for it in range(1000000):
    X_mb, y_mb = mnist.train.next_batch(mb_size)

    _, loss = sess.run([solver, vae_loss], feed_dict={X: X_mb, c: y_mb})

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('Loss: {:.4}'. format(loss))
        print()

        y = np.zeros(shape=[16, y_dim])
        y[:, np.random.randint(0, y_dim)] = 1.

        samples = sess.run(X_samples,
                           feed_dict={z: np.random.randn(16, z_dim), c: y})

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

