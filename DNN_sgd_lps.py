import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import scipy.io as scio
import h5py
import time

LOG = open("/home/hyli/Data/InternData/log_full_sgd_lr0002.txt", "w")

rng = np.random.RandomState(1234)
random_state = 42

par = scio.loadmat('/home/hyli/Data/InternData/mvn_store.mat')

mean_noisy = np.array(par['global_mean'], dtype='float32')
std_noisy = np.array(par['global_std'], dtype='float32')

mean_noisy = mean_noisy[0, :]
std_noisy = std_noisy[0, :]


def make_window_buffer(x, neighbor=3):
    m, n = x.shape

    tmp = np.zeros(m * n * (neighbor * 2 + 1), dtype='float32').reshape(m, -1)
    for i in range(2 * neighbor + 1):
        if (i <= neighbor):

            shift = neighbor - i
            tmp[shift:m, i * n: (i + 1) * n] = x[:m - shift]
            for j in range(shift):
                tmp[j, i * n: (i + 1) * n] = x[0, :]

        else:

            shift = i - neighbor
            tmp[:m-shift, i * n: (i+1) * n] = x[shift:m]
            for j in range(shift):
                tmp[m-(j + 1), i * n: (i + 1) * n] = x[m-1, :]

    return tmp


def Normalize_data(x, mean_noisy, std_noisy):
    mean_noisy_10 = np.tile(mean_noisy, [1, 8])[0, :]
    std_noisy_10 = np.tile(std_noisy, [1, 8])[0, :]
    tmp = (x-mean_noisy_10)/std_noisy_10[np.newaxis, :]
    return np.array(tmp, dtype='float32')


def Normalize_label(x, mean_noisy, std_noisy):
    mean_noisy_2 = np.tile(mean_noisy, [1, 2])[0, :]
    std_noisy_2 = np.tile(std_noisy, [1, 2])[0, :]
    tmp = (x-mean_noisy_2)/std_noisy_2[np.newaxis, :]
    return np.array(tmp, dtype='float32')   


def gen_context(x, nat, sentence_id, neighbor, global_mean, global_std):
    m = x.shape[0]
    data = np.zeros([m, 257*8])
    #sentence_id = np.r_[np.zeros([1,1]),sentence_id]
    for ind in range(len(sentence_id)-1):
        tmp_data = make_window_buffer(
            x[sentence_id[ind]:sentence_id[ind+1], :], neighbor)
        tmp_data = np.c_[tmp_data, nat[sentence_id[ind]:sentence_id[ind+1]]]
        tmp_data = Normalize_data(tmp_data, global_mean, global_std)
        data[sentence_id[ind]:sentence_id[ind+1]] = tmp_data

    return data


class Autoencoder:

    def __init__(self, vis_dim, hid_dim, W, function=lambda x: x):
        self.W = W
        self.a = tf.Variable(np.zeros(vis_dim).astype('float32'), name='a')
        self.b = tf.Variable(np.zeros(hid_dim).astype('float32'), name='b')
        self.function = function
        self.params = [self.W, self.a, self.b]

    def encode(self, x):
        u = tf.matmul(x, self.W) + self.b
        return self.function(u)

    def decode(self, x):
        u = tf.matmul(x, tf.transpose(self.W)) + self.a
        return self.function(u)

    def f_prop(self, x):
        y = self.encode(x)
        return self.decode(y)

    def reconst_error(self, x, noise):
        tilde_x = x * noise
        reconst_x = self.f_prop(tilde_x)
        error = tf.reduce_mean(tf.reduce_sum((x - reconst_x)**2, 1))
        return error, reconst_x


class Dense:

    def __init__(self, in_dim, out_dim, function=lambda x: x):
        self.W = tf.Variable(rng.uniform(low = -0.1,
                                         high = 0.1, size=(in_dim, out_dim)).astype('float32'), name='W')
        self.b = tf.Variable(np.zeros([out_dim]).astype('float32'))
        self.function = function
        self.params = [self.W, self.b]

        self.ae = Autoencoder(in_dim, out_dim, self.W, self.function)

    def f_prop(self, x):
        u = tf.matmul(x, self.W) + self.b
        self.z = self.function(u)
        return self.z

    def pretrain(self, x, noise):
        cost, reconst_x = self.ae.reconst_error(x, noise)
        return cost, reconst_x


layers = [
    Dense(257*8, 2048, tf.nn.sigmoid),
    Dense(2048, 2048, tf.nn.sigmoid),
    Dense(2048, 2048, tf.nn.sigmoid),
    Dense(2048, 257)
]

keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, [None, 257*8])
t = tf.placeholder(tf.float32, [None, 257])


def f_props(layers, x):
    for i, layer in enumerate(layers):
        x = layer.f_prop(x)
        if(i != len(layers)-1):
            x = tf.nn.dropout(x, keep_prob)
    return x

y = f_props(layers, x)

cost_fine = tf.reduce_mean(tf.reduce_sum((y - t)**2, 1))


lrate_p = tf.placeholder(tf.float32)
mt_p = tf.placeholder(tf.float32)

train_fine = tf.train.MomentumOptimizer(
    learning_rate=lrate_p, momentum=mt_p).minimize(cost_fine)

saver = tf.train.Saver()

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

n_epochs = 50
batch_size = 128

part_num_total = 230

data_file = h5py.File(
    '/home/hyli/Data/InternData/trainDB_lps/RawData_Part888.mat')

data_valid = np.array(data_file['data'], dtype='float32').transpose()
nat = np.array(data_file['nat'], dtype='float32').transpose()
sentence_id = np.array(data_file['sentence_id'], dtype='int32').transpose()
sentence_id = sentence_id[:,0]


data_valid = gen_context(data_valid, nat, sentence_id,
                         3, mean_noisy, std_noisy)


label_valid = np.array(data_file['label'], dtype='float32').transpose()

label_valid = Normalize_label(label_valid, mean_noisy, std_noisy)

label_valid = label_valid[:,:257]

del data_file
del nat
del sentence_id

#saver.restore(sess,'/home/hyli/Data/InterData/DNN_full_sgd_lr0002_model')

print("FineTuning begin")

Cost_validation = sess.run(cost_fine,
                           feed_dict={x: data_valid, t: label_valid, keep_prob: 1.0})

print('EPOCH: 0, Validation cost: %.3f ' % (Cost_validation))

cost_valid_best = 1000000

for epoch in range(n_epochs):

    lrate = 0.001

    #if(epoch>3):
    #    lrate = 0.0005
    #if(epoch>10):
    #    lrate = 0.0002
   # if(epoch>20):
  #      lrate = 0.0001
    if(epoch>10):
        lrate = 0.0005



    mt = 0.9

    time_start = time.time()
    part_num_list = shuffle(range(part_num_total))
    for part_num in part_num_list:
        try:
            del data_part
            del _data
            del _label
            del _nat
            del sentence_id
        except:
            pass

        data_part = scio.loadmat(
            '/home/hyli/Data/InternData/trainDB_lps_shuffle/NormContextData_Part'+str(part_num+1)+'.mat')
        _data = np.array(data_part['data'], dtype='float32')
        _label = np.array(data_part['label'], dtype='float32')
        del data_part
        # doing normalization
        _label = _label[:,:257]

        _data, _label = shuffle(_data, _label)
        n_batches = _data.shape[0] // batch_size

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            sess.run(train_fine,
                     feed_dict={x: _data[start:end],
                                t: _label[start:end],
                                keep_prob: 0.8,
                                lrate_p : lrate,
                                mt_p: mt})
        #print('part %i finished'%(part_num+1))



    Cost_validation = sess.run(cost_fine,
                               feed_dict={x: data_valid, t: label_valid, keep_prob: 1.0})
    time_end = time.time()
    print('EPOCH: %i, Validation cost: %.3f ' % (epoch + 1, Cost_validation))
    print('Elapsed time for one epoch is %.3f' % (time_end-time_start))
    LOG.write('EPOCH: %i, Validation cost: %.3f \n' %
              (epoch + 1, Cost_validation))
    LOG.flush()

    if(Cost_validation < cost_valid_best):
        save_dict = {}
        save_dict['W1'] = sess.run(layers[0].W)
        save_dict['b1'] = sess.run(layers[0].b)
        save_dict['W2'] = sess.run(layers[1].W)
        save_dict['b2'] = sess.run(layers[1].b)
        save_dict['W3'] = sess.run(layers[2].W)
        save_dict['b3'] = sess.run(layers[2].b)
        save_dict['W4'] = sess.run(layers[3].W)
        save_dict['b4'] = sess.run(layers[3].b)

        MATFILE = '/home/hyli/Data/InternData/DNN_full_sgd_lr0002.mat'
        scio.savemat(MATFILE, save_dict)
        cost_valid_best = Cost_validation
        print('Model in EPOCH:%d is saved' % (epoch+1))
        LOG.write('Model in EPOCH:%d is saved' % (epoch+1))
    saver.save(sess,'/home/hyli/Data/InterData/DNN_full_sgd_lr0002_next_model')


LOG.close()
del data_valid
del label_valid
del _data
del _label


sess.close()