############################################
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import scipy.io as scio
import h5py
import time

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
    # sentence_id = np.r_[np.zeros([1,1]),sentence_id]
    for ind in range(len(sentence_id)-1):
        tmp_data = make_window_buffer(x[sentence_id[ind]:sentence_id[ind+1], :], neighbor)
        tmp_data = np.c_[tmp_data, nat[sentence_id[ind]:sentence_id[ind+1]]]
        tmp_data = Normalize_data(tmp_data, global_mean, global_std)
        data[sentence_id[ind]:sentence_id[ind+1]] = tmp_data

    return data

###############################################


part_num_total = 230

part_num_list = range(part_num_total)

for part_num in part_num_list:
	data_part = h5py.File('/home/hyli/Data/InternData/trainDB_lps/RawData_Part'+str(part_num+1)+'.mat')
	data = np.array(data_part['data'], dtype='float32').transpose()
	label = np.array(data_part['label'], dtype='float32').transpose()
	nat = np.array(data_part['nat'], dtype='float32').transpose()
	sentence_id = np.array(data_part['sentence_id'], dtype='int32').transpose()
	sentence_id = sentence_id[:,0]

	del data_part

	data = gen_context(data, nat, sentence_id, 3, mean_noisy, std_noisy)
	label = Normalize_label(label, mean_noisy, std_noisy) 
	del nat
	save_dict={}
	save_dict['data'] = data
	save_dict['label'] = label
	del data
	del label
	MATFILE = '/home/hyli/Data/InternData/trainDB_lps_shuffle/NormContextData_Part'+str(part_num+1)+'.mat'
	scio.savemat(MATFILE, save_dict)
	print 'Gen NormContextPart'+str(part_num+1)+'finished'
