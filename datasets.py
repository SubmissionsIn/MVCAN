import numpy as np
import scipy.io as sio
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
from ClusteringTest import test


def load_data(config):
    """
       Load multi-view data, different datasets may need different data pre-processing methods,
       e.g., normalization, regularization, cleaning labels to [0, 1, 2 ... K-1], etc.
    """
    data_name = config['dataset']
    X_list = []
    Y_list = []

    if data_name in ['DIGIT']:
        mat = sio.loadmat("./data/DIGIT2V_N.mat")
        X_list.append(mat['X1'].astype('float32'))
        X_list.append(mat['X2'].astype('float32'))
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y)
    elif data_name in ['NoisyDIGIT']:
        mat = sio.loadmat("./data/DIGIT2V_N.mat")
        X_list.append(mat['X1'].astype('float32'))
        X_list.append(mat['X2'].astype('float32'))
        X_list.append(mat['X3'].astype('float32'))
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y)
    elif data_name in ['Amazon']:
        mat = sio.loadmat("./data/Amazon3V_N.mat")
        X_list.append(mat['X1'].astype('float32'))
        X_list.append(mat['X2'].astype('float32'))
        X_list.append(mat['X3'].astype('float32'))
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y)
    elif data_name in ['NoisyAmazon']:
        mat = sio.loadmat("./data/Amazon3V_N.mat")
        X_list.append(mat['X1'].astype('float32'))
        X_list.append(mat['X2'].astype('float32'))
        X_list.append(mat['X3'].astype('float32'))
        X_list.append(mat['X4'].astype('float32'))
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y)
    elif data_name in ['COIL']:
        mat = sio.loadmat("./data/COIL3V_N.mat")
        X_list.append(mat['X1'].astype('float32'))
        X_list.append(mat['X2'].astype('float32'))
        X_list.append(mat['X3'].astype('float32'))
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y)
    elif data_name in ['NoisyCOIL']:
        mat = sio.loadmat("./data/COIL3V_N.mat")
        X_list.append(mat['X1'].astype('float32'))
        X_list.append(mat['X2'].astype('float32'))
        X_list.append(mat['X3'].astype('float32'))
        X_list.append(mat['X4'].astype('float32'))
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y)
    elif data_name in ['BDGP']:
        mat = sio.loadmat("./data/BDGP2V_N.mat")
        X_list.append(mat['X1'].astype('float32'))
        X_list.append(mat['X2'].astype('float32'))
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y)
    elif data_name in ['NoisyBDGP']:
        mat = sio.loadmat("./data/BDGP2V_N.mat")
        X_list.append(mat['X1'].astype('float32'))
        X_list.append(mat['X2'].astype('float32'))
        X_list.append(mat['X3'].astype('float32'))
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y)
    elif data_name in ['DHA']:
        mat = sio.loadmat("./data/DHA.mat")
        X_list.append(min_max_scaler.fit_transform(mat['X1'].astype('float32')))
        X_list.append(min_max_scaler.fit_transform(mat['X2'].astype('float32')))
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y)
    elif data_name in ['Caltech-6V']:
        mat = sio.loadmat("./data/Caltech.mat")
        X_list.append(mat['X1'].astype('float32'))
        X_list.append(mat['X2'].astype('float32'))
        X_list.append(min_max_scaler.fit_transform(mat['X3'].astype('float32')))
        X_list.append(mat['X4'].astype('float32'))
        X_list.append(mat['X5'].astype('float32'))
        X_list.append(mat['X6'].astype('float32'))
        y = np.squeeze(mat['Y']).astype('int')
        # print(y[0:100])       # 7 class labels are [6  2  3  3  4  1  4  95  6  2  4  2  4  1  4  6  0 ...]
        for i in range(len(y)):
            if y[i] == 95:
                y[i] = 5        # cleaning labels to [0, 1, 2 ... K-1] for visualization
        Y_list.append(y)
        # print(y[0:100])
    elif data_name in ['RGB-D']:
        mat = sio.loadmat("./data/RGB-D.mat")
        X_list.append(mat['X1'].astype('float32'))
        X_list.append(mat['X2'].astype('float32'))
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y)
    elif data_name in ['YoutubeVideo']:
        mat = sio.loadmat("./data/Video-3V.mat")
        X_list.append(mat['X1'].astype('float32'))
        X_list.append(mat['X2'].astype('float32'))
        X_list.append(mat['X3'].astype('float32'))
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y - 1)    # cleaning labels to [0, 1, 2 ... K-1] for visualization

    return X_list, Y_list
