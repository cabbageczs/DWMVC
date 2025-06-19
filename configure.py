def get_default_config(data_name):
    if data_name in ['HandWritten']:
        """The default configs."""
        return dict(
            Module=dict(
                in_dim=[240, 76, 216, 47, 64],
                trans_dim=512,
                trans_layers=1,
                trans_headers=4,
                trans_dropout=0
            ),
            Dataset=dict(
                num_sample=2000,
                num_classes=10,
                num_views=5,
                missing_rate=0.5,
                batch_size=256,
                using_tsne=True,
            ),
            training=dict(
                seed=13,
                lr=1.0e-3,
                lr_con=3.0e-4,
                # Transformer
                epoch_em=51,
                # constrast
                mse_epochs=200,
                con_epochs=50,
                loss_weight1=1,
                loss_weight2=1,
                # graph loss
                knn=10,
                lambda_graph=1,
                # self_paces 
                epoch1=21,
                epoch2=11,
            ),
        )
        
    elif data_name in ['aloideep3v']:
        """The default configs."""
        return dict(
            Module=dict(
                in_dim=[2048, 4096, 2048],
                trans_dim=512,
                trans_layers=1,
                trans_headers=4,
                trans_dropout=0
            ),
            Dataset=dict(
                num_sample=10800,
                num_classes=100,
                num_views=3,
                missing_rate=0.5,
                batch_size=256,
                using_tsne=False,
            ),
            training=dict(
                seed=123,
                lr=1.0e-3,
                lr_con=3.0e-4,
                # Transformer
                epoch_em=51,
                # constrast
                mse_epochs=200,
                con_epochs=100,
                loss_weight1=1,
                loss_weight2=1,
                # graph loss
                knn=10,
                lambda_graph=1,
                # self_paces 
                epoch1=21,
                epoch2=11,
            ),
        )
    
    elif data_name in ['Caltech101-7']:
        return dict(
            Module=dict(
                in_dim=[48, 40, 254, 1984, 512, 928],
                trans_dim=512,
                trans_layers=1,
                trans_headers=4,
                trans_dropout=0
            ),
            Dataset=dict(
                num_sample=1474,
                num_classes=7,
                num_views=6,
                missing_rate=0.5,
                batch_size=512,
                using_tsne=False,
                knn=2,
            ),
            training=dict(
                seed=13,
                lr=1.0e-3,
                lr_con=3.0e-4,
                # Transformer
                epoch_em=51,
                # constrast
                mse_epochs=100,
                con_epochs=50,
                loss_weight1=1,
                loss_weight2=1,
                # graph loss
                knn=10,
                lambda_graph=1,
                # self_paces 
                epoch1=11,
                epoch2=51,
            ),
        )
    
    elif data_name in ['Fashion']:
        """The default configs."""
        return dict(
            Module=dict(
                in_dim=[784, 784, 784],
                trans_dim=512,
                trans_layers=1,
                trans_headers=4,
                trans_dropout=0
            ),
            Dataset=dict(
                num_sample=10000,
                num_classes=10,
                batch_size=512,
                num_views=3,
                missing_rate=0.5,
                using_tsne=False,
            ),
            training=dict(
                seed=12,
                lr=1.0e-3,
                lr_con=3.0e-4,
                # Transformer
                epoch_em=51,
                # constrast
                mse_epochs=200,
                con_epochs=50,
                loss_weight1=1,
                loss_weight2=1,
                # graph loss
                knn=10,
                lambda_graph=1,
                # self_paces 
                epoch1=21,
                epoch2=11,
            ),
        )
        
    elif data_name in ['Scene-15']:
        """The default configs."""
        return dict(
            Module=dict(
                in_dim=[20, 59, 40],
                trans_dim=512,
                trans_layers=1,
                trans_headers=4,
                trans_dropout=0
            ),
            Dataset=dict(
                num_sample=4485,
                num_classes=15,
                num_views=3,
                missing_rate=0.5,
                batch_size=512,
                using_tsne=False,
            ),
            training=dict(
                seed=12,
                lr=1.0e-3,
                lr_con=3.0e-4,
                # Transformer
                epoch_em=51,
                # constrast
                mse_epochs=200,
                con_epochs=50,
                loss_weight1=1,
                loss_weight2=1,
                # graph loss
                knn=10,
                lambda_graph=1,
                # self_paces 
                epoch1=11,
                epoch2=11,
            ),
        )
        
    elif data_name in ['CIFAR10']:
        """The default configs."""
        return dict(
            Module=dict(
                in_dim=[512, 2048, 1024],
                trans_dim=512,
                trans_layers=1,
                trans_headers=4,
                trans_dropout=0
            ),
            Dataset=dict(
                num_sample=30000,
                num_classes=10,
                batch_size=256,
                num_views=3,
                missing_rate=0.5,
                using_tsne=False,
            ),
            training=dict(
                seed=123,
                lr=1.0e-3,
                lr_con=3.0e-4,
                # Transformer
                epoch_em=51,
                # constrast
                mse_epochs=200,
                con_epochs=50,
                # graph loss
                knn=10,
                lambda_graph=1,
                loss_weight1=1,
                loss_weight2=1,
                # self_paces 
                epoch1=21,
                epoch2=11,
            ),
        )
    
    else:
        raise Exception('Undefined data_name')
