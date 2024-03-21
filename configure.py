def get_default_config(data_name):
    if data_name in ['DIGIT']:
        return dict(
            Autoencoder=dict(
                arch=[10],
                channal=[1],
                activations='relu',
                batchnorm=False,
                FCN=False
            ),
            training=dict(
                seed=0,
                batch_size=256,
                init_epoch=500,
                T_1=2,
                T_2=100,
                epoch=1000,
                lr=0.0001,
                lambda1=10,
            ),
        )
    elif data_name in ['NoisyDIGIT']:
        return dict(
            Autoencoder=dict(
                arch=[10],
                channal=[1],
                activations='relu',
                batchnorm=False,
                FCN=False
            ),
            training=dict(
                seed=0,
                batch_size=256,
                init_epoch=500,
                T_1=2,
                T_2=100,
                epoch=1000,
                lr=0.0001,
                lambda1=10,
            ),
        )
    elif data_name in ['COIL']:
        return dict(
            Autoencoder=dict(
                arch=[10],
                channal=[1],
                activations='relu',
                batchnorm=False,
                FCN=False
            ),
            training=dict(
                seed=3,
                batch_size=256,
                init_epoch=500,
                T_1=2,
                T_2=100,
                epoch=1000,
                lr=0.0001,
                lambda1=10,
            ),
        )
    elif data_name in ['NoisyCOIL']:
        return dict(
            Autoencoder=dict(
                arch=[10],
                channal=[1],
                activations='relu',
                batchnorm=False,
                FCN=False
            ),
            training=dict(
                seed=3,
                batch_size=256,
                init_epoch=500,
                T_1=2,
                T_2=100,
                epoch=1000,
                lr=0.0001,
                lambda1=10,
            ),
        )
    elif data_name in ['Amazon']:
        return dict(
            Autoencoder=dict(
                arch=[10],
                channal=[3],
                activations='relu',
                batchnorm=False,
                FCN=False
            ),
            training=dict(
                seed=1,
                batch_size=256,
                init_epoch=200,
                T_1=2,
                T_2=100,
                epoch=3000,
                lr=0.0001,
                lambda1=100,
            ),
        )
    elif data_name in ['NoisyAmazon']:
        return dict(
            Autoencoder=dict(
                arch=[10],
                channal=[3],
                activations='relu',
                batchnorm=False,
                FCN=False
            ),
            training=dict(
                seed=1,
                batch_size=256,
                init_epoch=200,
                T_1=2,
                T_2=100,
                epoch=2500,
                lr=0.0001,
                lambda1=100,
            ),
        )
    elif data_name in ['BDGP']:
        return dict(
            Autoencoder=dict(
                arch=[10],
                channal=[1],
                activations='relu',
                batchnorm=False,
                FCN=True
            ),
            training=dict(
                seed=1,
                batch_size=256,
                init_epoch=200,
                T_1=2,
                T_2=100,
                epoch=1000,
                lr=0.0001,
                lambda1=10,
            ),
        )
    elif data_name in ['NoisyBDGP']:
        return dict(
            Autoencoder=dict(
                arch=[10],
                channal=[1],
                activations='relu',
                batchnorm=False,
                FCN=True
            ),
            training=dict(
                seed=1,
                batch_size=256,
                init_epoch=200,
                T_1=2,
                T_2=100,
                epoch=1000,
                lr=0.0001,
                lambda1=10,
            ),
        )
    elif data_name in ['DHA']:
        return dict(
            Autoencoder=dict(
                arch=[10],
                channal=[1],
                activations='relu',
                batchnorm=False,
                FCN=True
            ),
            training=dict(
                seed=1,
                batch_size=256,
                init_epoch=200,
                T_1=2,
                T_2=100,
                epoch=1000,
                lr=0.0001,
                lambda1=0.1,
            ),
        )
    elif data_name in ['RGB-D']:
        return dict(
            Autoencoder=dict(
                arch=[10],
                channal=[1],
                activations='relu',
                batchnorm=False,
                FCN=True
            ),
            training=dict(
                seed=13,
                batch_size=256,
                init_epoch=200,
                T_1=2,
                T_2=100,
                epoch=200,
                lr=0.0001,
                lambda1=0.1,
            ),
        )
    elif data_name in ['Caltech-6V']:
        return dict(
            Autoencoder=dict(
                arch=[10],
                channal=[1],
                activations='relu',
                batchnorm=False,
                FCN=True
            ),
            training=dict(
                seed=5,
                batch_size=256,
                init_epoch=200,
                T_1=2,
                T_2=100,
                epoch=1000,
                lr=0.0001,
                lambda1=0.01,
            ),
        )
    elif data_name in ['YoutubeVideo']:
        return dict(
            Autoencoder=dict(
                arch=[10],
                channal=[1],
                activations='relu',
                batchnorm=False,
                FCN=True
            ),
            training=dict(
                seed=2,
                batch_size=256,
                init_epoch=50,
                T_1=2,
                T_2=10,
                epoch=50,
                lr=0.0001,
                lambda1=10.0,
            ),
        )
    else:
        raise Exception('Undefined data_name')
