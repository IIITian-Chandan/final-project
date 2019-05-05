def choose_model_param(model_id):
    #  loss, , batch_size, train_epocs,
    #optimizer, lr

    #mildnet_ablation_study
    if model_id == 'mildnet_vgg16_skip_1':
        loss="hinge_loss"
        optimizer="mo"
        train_epocs=20
        batch_size=32
        lr=0.001
    elif model_id == 'mildnet_vgg16_skip_2':
        loss="hinge_loss"
        optimizer="mo"
        train_epocs=20
        batch_size=32
        lr=0.001
    elif model_id == 'mildnet_vgg16_skip_3':
        loss="hinge_loss"
        optimizer="mo"
        train_epocs=20
        batch_size=32
        lr=0.001
    elif model_id == 'mildnet_vgg16_skip_4':
        loss="hinge_loss"
        optimizer="mo"
        train_epocs=20
        batch_size=32
        lr=0.001

    #mildnet_low_features
    elif model_id == 'mildnet_512_512':
        loss="hinge_loss"
        optimizer="mo"
        train_epocs=20
        batch_size=16
        lr=0.001

    elif model_id == 'mildnet_512_no_dropout':
        loss="hinge_loss"
        optimizer="mo"
        train_epocs=20
        batch_size=16
        lr=0.001

    elif model_id == 'mildnet_1024_512':
        loss="hinge_loss"
        optimizer="mo"
        train_epocs=20
        batch_size=16
        lr=0.001

        #mildnet_other_losses
    elif model_id == 'mildnet_vgg16_angular_loss_1':
        model_id="mildnet_vgg16"
        loss="angular_loss_1"
        optimizer="mo"
        train_epocs=6
        batch_size=16
        lr=0.001

    elif model_id == 'mildnet_vgg16_angular_loss_2':
        model_id="mildnet_vgg16"
        loss="angular_loss_2"
        optimizer="mo"
        train_epocs=6
        batch_size=16
        lr=0.001

    elif model_id == 'mildnet_vgg16_contrastive_loss':
        model_id="mildnet_vgg16"
        loss="contrastive_loss"
        optimizer="mo"
        train_epocs=8
        batch_size=16
        lr=0.001

    elif model_id == 'mildnet_vgg16_hinge_new_loss':
        model_id="mildnet_vgg16"
        loss="hinge_new_loss"
        optimizer="mo"
        train_epocs=18
        batch_size=16
        lr=0.0001

    elif model_id == 'mildnet_vgg16_lossless_loss':
        model_id="mildnet_vgg16"
        loss="lossless_loss"
        optimizer="mo"
        train_epocs=6
        batch_size=16
        lr=0.001

    #mildnet_other_variants
    elif model_id == 'mildnet_all_trainable':
        loss="hinge_loss"
        optimizer="mo"
        train_epocs=20
        batch_size=32
        lr=0.001

    elif model_id == 'mildnet_vgg16_cropped':
        model_id="mildnet_vgg16"
        loss="hinge_loss"
        optimizer="mo"
        train_epocs=20
        batch_size=16
        lr=0.0001

    elif model_id == 'mildnet_mobilenet':
        loss="hinge_loss"
        optimizer="mo"
        train_epocs=20
        batch_size=16
        lr=0.001

    elif model_id == 'mildnet_vgg16_big':
        loss="hinge_loss"
        optimizer="rms"
        train_epocs=8
        batch_size=16
        lr=0.001

    elif model_id == 'mildnet_vgg19':
        loss="contrastive_loss"
        optimizer="mo"
        train_epocs=20
        batch_size=16
        lr=0.001

    elif model_id == 'mildnet_without_skip_big':
        loss="hinge_loss"
        optimizer="mo"
        train_epocs=12
        batch_size=16
        lr=0.001

    elif model_id == 'mildnet_without_skip':
        loss="hinge_loss"
        optimizer="mo"
        train_epocs=8
        batch_size=16
        lr=0.001

    #base models
    elif model_id == 'alexnet':
        loss="hinge_loss"
        optimizer="mo"
        train_epocs=20
        batch_size=32
        lr=0.001

    elif model_id == 'mildnet':
        model_id = "mildnet_vgg16"
        loss="contrastive_loss"
        optimizer="mo"
        train_epocs=30
        batch_size=16
        lr=0.001

    elif model_id == 'ranknet_vgg19':
        loss="contrastive_loss"
        optimizer="mo"
        train_epocs=30
        batch_size=16
        lr=0.001

    elif model_id == 'ranknet_resnet':
        loss="contrastive_loss"
        optimizer="mo"
        train_epocs=30
        batch_size=16
        lr=0.001

    elif model_id == 'ranknet_inception':
        loss="contrastive_loss"
        optimizer="mo"
        train_epocs=30
        batch_size=16
        lr=0.001

    elif model_id == 'vanila_vgg16':
        loss="hinge_loss"
        optimizer="mo"
        train_epocs=8
        batch_size=16
        lr=0.001

    elif model_id == 'visnet_lrn2d_model':
        loss="hinge_loss"
        optimizer="mo"
        train_epocs=20
        batch_size=16
        lr=0.001

    elif model_id == 'visnet_model':
        loss="hinge_loss"
        optimizer="mo"
        train_epocs=20
        batch_size=16
        lr=0.001

    elif model_id == 'resnet_50_hinge_loss':
        model_id = 'resnet_50'
        loss="hinge_loss"
        optimizer="mo"
        train_epocs=30
        batch_size=16
        lr=0.001

    elif model_id == 'resnet_50_contrastive_loss':
        model_id = 'resnet_50'
        loss="contrastive_loss"
        optimizer="mo"
        train_epocs=30
        batch_size=16
        lr=0.001

    elif model_id == 'inception_v3_hinge_loss':
        model_id = 'inception_v3'
        loss="hinge_loss"
        optimizer="mo"
        train_epocs=30
        batch_size=16
        lr=0.001

    elif model_id == 'inception_v3_contrastive_loss':
        model_id = 'inception_v3'
        loss="contrastive_loss"
        optimizer="mo"
        train_epocs=30
        batch_size=16
        lr=0.001



    else:
        print("Please enter a valid choice.")
        exit()
    return model_id, loss, optimizer, train_epocs, batch_size,lr
