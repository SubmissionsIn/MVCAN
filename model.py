import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from ClusteringTest import test
import ClusteringTest
import numpy as np
import random
from torch.nn.parameter import Parameter
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
from numpy import hstack
import time
from TSNE import TSNE_PLOT


class Autoencoder(nn.Module):
    """AutoEncoder module"""

    def __init__(self,
                 encoder_dim,
                 activation='relu',
                 batchnorm=True,
                 n_clusters=10,
                 FCN=True,
                 channal=1):
        """Constructor.
        """
        super(Autoencoder, self).__init__()
        self.FCN = FCN
        encoder_layers = []
        self._activation = activation
        self._batchnorm = batchnorm
        if FCN:
            self._dim = len(encoder_dim) - 1
            for i in range(self._dim):
                encoder_layers.append(
                    nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
                if i < self._dim - 1:
                    if self._batchnorm:
                        encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                    if self._activation == 'sigmoid':
                        encoder_layers.append(nn.Sigmoid())
                    elif self._activation == 'leakyrelu':
                        encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                    elif self._activation == 'tanh':
                        encoder_layers.append(nn.Tanh())
                    elif self._activation == 'relu':
                        if i < self._dim - 1:
                            encoder_layers.append(nn.ReLU())
                    else:
                        raise ValueError('Unknown activation type %s' % self._activation)
        else:
            encoder_layers = [
                nn.Conv2d(channal[0], 32, (4, 4), stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, (4, 4), stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, (4, 4), stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(start_dim=1),
                nn.Linear(64 * 4 * 4, encoder_dim[-1]),
            ]
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        if FCN:
            for i in range(self._dim):
                decoder_layers.append(
                    nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
                if self._batchnorm:
                    decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
                if self._activation == 'sigmoid':
                    decoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    decoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    if i < self._dim - 1:
                        decoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        else:
            decoder_layers = [
                nn.Linear(encoder_dim[-1], 64 * 4 * 4),
                nn.ReLU(),
                nn.Unflatten(dim=1, unflattened_size=(64, 4, 4)),
                nn.ConvTranspose2d(64, 32, (4, 4), stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 32, (4, 4), stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, channal[0], (4, 4), stride=2, padding=1),
            ]
        self._decoder = nn.Sequential(*decoder_layers)

        # cluster layer
        self.alpha = 1.0
        self._cluster_layer = Parameter(torch.Tensor(n_clusters, encoder_dim[-1]))
        torch.nn.init.xavier_normal_(self._cluster_layer.data)

    def encoder(self, x):
        """Encode sample features.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              feature Z^v.
        """
        latent = self._encoder(x)
        return latent

    def decoder(self, latent):
        """Decode sample features.

            Args:
              latent: [num, latent_dim] float tensor, feature Z^v.

            Returns:
              float tensor, reconstruction samples x_hat.
        """
        x_hat = self._decoder(latent)
        return x_hat

    def clustering(self, latent):
        z = latent
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self._cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    def forward(self, x):
        """Pass through model.
        """
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(latent.unsqueeze(1) - self._cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_hat, latent, q


class MvCAN():
    """MvCAN module."""
    def __init__(self,
                 config, view_num, view_size, n_clusters=20, seed=0, data_size=10000):
        """Constructor.

        Args:
          config: parameters defined in configure.py.
        """
        self._config = config
        self.view_num = view_num
        self._latent_dim = config['Autoencoder']['arch'][-1]
        self.autoencoders = []
        self.n_clusters = n_clusters
        for i in range(view_num):
            # Set random seeds for model initialization
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            # np.random.seed(seed)
            # random.seed(seed)

            self.autoencoders.append(
                Autoencoder([view_size[i], 500, 500, 2000, self._latent_dim],
                            config['Autoencoder']['activations'],
                            config['Autoencoder']['batchnorm'],
                            n_clusters=self.n_clusters,
                            FCN=config['Autoencoder']['FCN'],
                            channal=config['Autoencoder']['channal'],
                            )
            )
        self.data_size = data_size

    def to_device(self, device):
        """ to cuda if gpu is used """
        for i in range(self.view_num):
            self.autoencoders[i].to(device)

    def train(self, config, X_train, Y_list, optimizers, device, ROUND=1):
        """Training the model.
        """
        time_begin = time.time()
        # Get complete data for training
        Y_list = torch.tensor(Y_list).int().to(device).squeeze(dim=0).unsqueeze(dim=1)
        batch_size = config['training']['batch_size']

        if self.view_num == 1:
            dataset = Data.TensorDataset(X_train[0], Y_list)
        if self.view_num == 2:
            dataset = Data.TensorDataset(X_train[0], X_train[1], Y_list)
        if self.view_num == 3:
            dataset = Data.TensorDataset(X_train[0], X_train[1], X_train[2], Y_list)
        if self.view_num == 4:
            dataset = Data.TensorDataset(X_train[0], X_train[1], X_train[2], X_train[3], Y_list)
        if self.view_num == 5:
            dataset = Data.TensorDataset(X_train[0], X_train[1], X_train[2], X_train[3], X_train[4], Y_list)
        if self.view_num == 6:
            dataset = Data.TensorDataset(X_train[0], X_train[1], X_train[2], X_train[3], X_train[4], X_train[5], Y_list)
        if self.view_num == 7:
            dataset = Data.TensorDataset(X_train[0], X_train[1], X_train[2], X_train[3], X_train[4], X_train[5],
                                         X_train[6], Y_list)
        All_data = torch.utils.data.DataLoader(
            dataset,
            batch_size=100000000,
            shuffle=False,
        )

        print("Initialization...")
        for epoch in range(config['training']['init_epoch']):
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False
            )
            for batch_idx, (X_data) in enumerate(loader):
                Z_A = []
                for v in range(self.view_num):
                    z_a = self.autoencoders[v].encoder(X_data[v])
                    Z_A.append(z_a)

                for v in range(self.view_num):
                    REC_loss = F.mse_loss(self.autoencoders[v].decoder(Z_A[v]), X_data[v])
                    loss = REC_loss
                    optimizers[v].zero_grad()
                    loss.backward()
                    optimizers[v].step()

        kmeans = KMeans(n_clusters=self.n_clusters, n_init=100)
        if self.data_size > 10000:
            kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, n_init=100, batch_size=10000)
        for batch_idx, (X_data) in enumerate(All_data):
            Y_list = X_data[-1]
            for v in range(self.view_num):
                z_a = self.autoencoders[v].encoder(X_data[v])
                y_pred = kmeans.fit_predict(z_a.data.cpu().numpy())
                if config['training']['epoch'] > 0:
                    self.autoencoders[v]._cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

            for v in range(self.view_num):
                z_a = self.autoencoders[v].encoder(X_data[v])
                y_a = self.autoencoders[v].clustering(z_a).cpu().detach().numpy().argmax(1)
                acc, nmi, ari = test(Y_list.cpu().detach().numpy().T[0], y_a)

        if self.view_num == 1:
            return acc, nmi, ari

        iteration = config['training']['T_2']
        www = []
        for view in range(self.view_num):
            www.append(1.0)
        LOSS = []
        for v in range(self.view_num):
            LOSS.append([0.0])

        # np.random.seed(ROUND)

        for epoch in range(config['training']['epoch'] + 1):
            all_loss = 0
            rec_losses = 0
            clu_losses = 0
            if epoch % iteration == 0:
                print("Epoch: " + str(epoch))

                for updata_www in range(config['training']['T_1']):
                    Z_A = []
                    y_view = []
                    for batch_idx, (X_data) in enumerate(All_data):
                        for v in range(self.view_num):
                            x_h, z_a, q = self.autoencoders[v](X_data[v])
                            z_a = min_max_scaler.fit_transform(z_a.data.cpu().numpy())
                            Z_A.append(z_a * www[v])
                            y_l = q.cpu().detach().numpy().argmax(1)
                            y_view.append(y_l)

                    latent_fusion = hstack(Z_A)

                    y_pred = kmeans.fit_predict(latent_fusion)

                    for view in range(self.view_num):
                        nmi = np.round(ClusteringTest.nmi(y_pred, y_view[view]), 5)
                        www[view] = np.exp(nmi)

                MMM = []
                for v in range(self.view_num):
                    _, _, _, M = self.Match(y_view[v], y_pred)
                    MMM.append(M)

                Center_init = kmeans.cluster_centers_
                new_P = self.new_P(latent_fusion, Center_init)
                p = self.target_distribution(new_P)
                p = torch.from_numpy(p).float().to(device)
                acc, nmi, ari = test(Y_list.cpu().detach().numpy().T[0], y_pred)

            if self.view_num == 1:
                dataset = Data.TensorDataset(X_train[0], p)
            if self.view_num == 2:
                dataset = Data.TensorDataset(X_train[0], X_train[1], p)
            if self.view_num == 3:
                dataset = Data.TensorDataset(X_train[0], X_train[1], X_train[2], p)
            if self.view_num == 4:
                dataset = Data.TensorDataset(X_train[0], X_train[1], X_train[2], X_train[3], p)
            if self.view_num == 5:
                dataset = Data.TensorDataset(X_train[0], X_train[1], X_train[2], X_train[3], X_train[4], p)
            if self.view_num == 6:
                dataset = Data.TensorDataset(X_train[0], X_train[1], X_train[2], X_train[3], X_train[4], X_train[5],
                                             p)
            if self.view_num == 7:
                dataset = Data.TensorDataset(X_train[0], X_train[1], X_train[2], X_train[3], X_train[4], X_train[5],
                                             X_train[6], p)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False
            )
            for batch_idx, (X_data) in enumerate(loader):
                Z_A = []
                Q_A = []
                P_A = []
                X_H = []
                P = X_data[-1]
                for v in range(self.view_num):
                    x_hat, z, q = self.autoencoders[v](X_data[v])
                    Z_A.append(z)
                    X_H.append(x_hat)
                    Q_A.append(q)
                    P_A.append(torch.mm(P, torch.from_numpy(MMM[v]).float().to(device)))

                for v in range(self.view_num):
                    REC_loss = F.mse_loss(X_H[v], X_data[v])
                    CLU_loss = F.mse_loss(Q_A[v], P_A[v])

                    loss = REC_loss + config['training']['lambda1'] * CLU_loss

                    optimizers[v].zero_grad()
                    loss.backward()
                    optimizers[v].step()
                    all_loss += loss.item()
                    rec_losses += REC_loss.item()
                    clu_losses += CLU_loss.item()

                    LOSS[v][0] = LOSS[v][0] + CLU_loss

            for v in range(self.view_num):
                LOSS[v].append(LOSS[v][0])
                LOSS[v][0] = 0

            if config['training']['epoch'] == 0:
                # too large datasets may lead to out-of-memory
                # TSNE_PLOT(latent_fusion, Y_list.cpu().detach().numpy().T[0], config['dataset'] + "_K_" + str(self.n_clusters))
                TSNE_PLOT(latent_fusion[0:10000, :], Y_list.cpu().detach().numpy().T[0][0:10000], config['dataset']+"_K_"+str(self.n_clusters))
                print("Visualization of Z_(t)...")

        time_end = time.time()
        print('time:', time_end - time_begin)

        return acc, nmi, ari

    def target_distribution(self, q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def new_P(self, inputs, centers):
        alpha = 1
        q = 1.0 / (1.0 + (np.sum(np.square(np.expand_dims(inputs, axis=1) - centers), axis=2) / alpha))
        q **= (alpha + 1.0) / 2.0
        q = np.transpose(np.transpose(q) / np.sum(q, axis=1))
        return q

    def Match(self, y_true, y_pred):
        """"
            y_modified = Match(y_modified_before, y_modified_target)
        """
        y_true = y_true.astype(np.int64)
        y_pred = y_pred.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(w.max() - w)
        new_y = np.zeros(y_true.shape[0])

        matrix = np.zeros((D, D), dtype=np.int64)
        matrix[row_ind, col_ind] = 1
        for i in range(y_pred.size):
            for j in row_ind:
                if y_true[i] == col_ind[j]:
                    new_y[i] = row_ind[j]
        return new_y, row_ind, col_ind, matrix
