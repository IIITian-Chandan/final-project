from __future__ import absolute_import
from __future__ import print_function
from clint.textui import colored
from comet_ml import Experiment
from datetime import datetime

needs_reproducible = True
if needs_reproducible:
    from numpy.random import seed

    seed(1)
    from tensorflow import set_random_seed

    set_random_seed(2)

from checkpointers import *
from accuracy import *
from utils import *
from model import *
import inspect
import argparse
import pandas as pd
import dill
from hyperdash import Experiment
from tensorflow.keras.callbacks import TensorBoard
import logging
from controller import choose_model_param
import json
import os
from create_dir import add_paths
import sys

print(colored.blue("==========================================="))
print(colored.blue("==========================================="))
print(colored.blue("================ IIIT Dharwad ================"))
print(colored.blue("==========================================="))
print(colored.blue("==========================================="))
print('\n\n')

print("make sure you have added environment. ")


def main(weights_path, is_tpu, **args):
    logging.getLogger().setLevel(logging.INFO)

    print(colored.green('==============================='))
    print(colored.green('Please choose an enviornment: ='))
    print(colored.green('==============================='))
    env_dict = {'1': "testing", '2': "development"}
    for i in env_dict.keys():
        print(i + " : " + env_dict[i])
    option = input()

    try:
        os.environ['DEEPRANKING_ENV'] = env_dict[option]
    except:
        os.environ['DEEPRANKING_ENV'] = env_dict["1"]
    env = os.getenv('DEEPRANKING_ENV', "testing")

    print(colored.green('============================'))
    print(colored.green('Your Environment is : ' + env))
    print(colored.green('============================'))

    if env == "testing":
        config_file = 'config_testing.json'
    elif env == "development":
        config_file = 'config.json'
    else:
        config_file = 'config_testing.json'

    cwd = os.getcwd()

    print(cwd)
    with open(cwd + '/trainer/' + config_file) as json_data:
        config = json.load(json_data)
        print(config)

    model_options = ['mildnet_vgg16_skip_1', 'mildnet_vgg16_skip_2', 'mildnet_vgg16_skip_3', 'mildnet_vgg16_skip_4',
                     'mildnet_512_512', 'mildnet_512_no_dropout',
                     'mildnet_1024_512', 'mildnet_vgg16_angular_loss_1', 'mildnet_vgg16_angular_loss_2',
                     'mildnet_vgg16_contrastive_loss', 'mildnet_vgg16_hinge_new_loss',
                     'mildnet_vgg16_lossless_loss', 'mildnet_all_trainable', 'mildnet_vgg16_cropped',
                     'mildnet_mobilenet',
                     'mildnet_vgg16_big', 'mildnet_vgg19',
                     'mildnet_without_skip_big', 'mildnet_without_skip', 'alexnet', 'mildnet', 'ranknet_vgg19',
                     'ranknet_resnet', 'ranknet_inception', 'vanila_vgg16',
                     'visnet_lrn2d_model', 'visnet_model', 'resnet_50_hinge_loss', 'resnet_50_contrastive_loss',
                     'inception_v3_hinge_loss', 'inception_v3_contrastive_loss']

    model_options_dict = {}
    for i in range(len(model_options)):
        model_options_dict[str(i + 1)] = model_options[i]

    print(colored.green('######### Please choose the model number #########'))
    for i in model_options_dict.keys():
        print(str(i) + ' :: ' + model_options_dict[i])
    print(colored.green('============================'))
    print(colored.green('choose a model number :'))
    print(colored.green('============================'))
    temp = input()

    file_id = model_options_dict[temp]
    model_id, loss, optimizer, train_epocs, batch_size, lr = choose_model_param(model_options_dict[temp])

    # comet-ml
    experiment = Experiment("iEJDqOgS8QPlGv7hK3MYESLE2", "deepranking-experiment3" + model_id,
                            "iiitian-chandan")

    train_csv = config['train_csv']
    val_csv = config['val_csv']
    hyperdash_key = config['hyperdash_key']
    job_dir = config['job_dir']
    data_path = config['data_path']
    job_dir, model_weight = add_paths(job_dir, file_id, env)

    batch_size *= 3
    is_full_data = False
    hyperdash_capture_io = True

    # Setting up Hyperdash
    def get_api_key():
        return hyperdash_key

    if hyperdash_key:
        exp = Experiment("final_project"+model_id, get_api_key, capture_io=hyperdash_capture_io)
        exp.param("model_name", job_dir.split("/")[-1])
        exp.param("data_path", data_path)
        exp.param("batch_size", batch_size)
        exp.param("train_epocs", train_epocs)
        exp.param("optimizer", optimizer)
        exp.param("lr", lr)
        if weights_path:
            exp.param("weights_path", weights_path)
        exp.param("loss", loss)
        exp.param("train_csv", train_csv)
        exp.param("val_csv", val_csv)

    # logging.info("Downloading Training Image from path {}".format(data_path))
    # downloads_training_images(data_path, is_cropped=("_cropped" in job_dir))

    logging.info("Building Model: {}".format(model_id))
    if model_id in globals():
        model_getter = globals()[model_id]
        model = model_getter()
    else:
        raise RuntimeError(colored.red("Failed. Model function {} not found".format(model_id)))

    if loss + "_fn" in globals():
        _loss_tensor = globals()[loss + "_fn"](batch_size)
    else:
        raise RuntimeError(colored.red("Failed. Loss function {} not found".format(loss + "_fn")))

    accuracy = accuracy_fn(batch_size)
    img_width, img_height = [int(v) for v in model.input[0].shape[1:3]]

    trainable_count, non_trainable_count = print_trainable_counts(model)

    if hyperdash_key:
        exp.param("trainable_count", trainable_count)
        exp.param("non_trainable_count", non_trainable_count)

    dg = DataGenerator({
        "rescale": 1. / 255,
        "horizontal_flip": True,
        "vertical_flip": True,
        "zoom_range": 0.2,
        "shear_range": 0.2,
        "rotation_range": 30,
        "fill_mode": 'nearest'
    }, data_path, train_csv, val_csv, target_size=(img_width, img_height))

    train_generator = dg.get_train_generator(batch_size, is_full_data)
    test_generator = dg.get_test_generator(batch_size)

    if model_weight != "None":
        # with file_io.FileIO(weights_path, mode='r') as input_f:
        #   with file_io.FileIO(job_dir+"/weights/"+model_weight, mode='w+') as output_f:
        #     output_f.write(input_f.read())
        print(colored.green("loading existing weights=========="))
        model.load_weights(job_dir + "/weights/" + model_weight)

    # model = multi_gpu_model(model, gpus=4)
    if optimizer == "mo":
        model.compile(loss=_loss_tensor,
                      optimizer=tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9, use_nesterov=True),
                      metrics=[accuracy])
    elif optimizer == "rms":
        model.compile(loss=_loss_tensor, optimizer=tf.train.RMSPropOptimizer(lr), metrics=[accuracy])
    else:
        logging.error(colored.red("Optimizer not supported"))
        return

    csv_logger = CSVLogger(job_dir, job_dir + "/output/training.log")
    model_checkpoint_path = job_dir + "/weights/" + datetime.strftime(datetime.now(),
                                                                      '%Y-%m-%d') + "_weights-improvement-{epoch:02d}-{val_loss:.2f}.h5"
    model_checkpointer = ModelCheckpoint(job_dir, model_checkpoint_path, save_best_only=False, save_weights_only=True,
                                         monitor="val_loss", verbose=1)
    tensorboard = TensorBoard(log_dir=job_dir + '/logs/', histogram_freq=0, write_graph=True, write_images=True)
    # test_accuracy = TestAccuracy(data_path)  # Not using test data as of now

    callbacks = [csv_logger, model_checkpointer, tensorboard]
    if hyperdash_key:
        callbacks.append(HyperdashCallback(exp))

    model_json = model.to_json()
    write_file_and_backup(model_json, job_dir, job_dir + "/output/model.def")

    with open(job_dir + "/output/model_code.pkl", 'wb') as f:
        dill.dump(model_getter, f)
    backup_file(job_dir, job_dir + "/output/model_code.pkl")

    model_code = inspect.getsource(model_getter)
    write_file_and_backup(model_code, job_dir, job_dir + "/output/model_code.txt")

    if is_tpu:
        model = tf.contrib.tpu.keras_to_tpu_model(
            model,
            strategy=tf.contrib.tpu.TPUDistributionStrategy(
                tf.contrib.cluster_resolver.TPUClusterResolver(os.environ['KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS'])
            )
        )

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=(train_generator.n // (train_generator.batch_size)),
                                  validation_data=test_generator,
                                  epochs=train_epocs,
                                  validation_steps=(test_generator.n // (test_generator.batch_size)),
                                  callbacks=callbacks)

    pd.DataFrame(history.history).to_csv(job_dir + "/output/history.csv")
    backup_file(job_dir, job_dir + "/output/history.csv")

    model.save_weights(job_dir + '/output/model.h5')
    backup_file(job_dir, job_dir + '/output/model.h5')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--weights-path',
        help='GCS location of pretrained weights path',
        default=None
    )
    parser.add_argument(
        '--is-tpu',
        help='is tpu used',
        default=False,
        type=bool
    )

    args = parser.parse_args()
    arguments = args.__dict__

    main(**arguments)
