# util
from tqdm import tqdm, tnrange

# data processing
import pandas as pd
import numpy as np

# keras, tensorflow
from keras.models import Model
from keras.layers import Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import load_img

from backbone.losses import bce_dice_loss
from backbone.models import build_model
from backbone.augmentation import ImageAugmenter

# Set some parameters
random_seed = 666666  # old iron double click 666
img_size_ori = 101
img_size_target = 101
im_width = 101
im_height = 101
im_chan = 1
basic_path = './input/'
path_train = basic_path + 'train/'
path_test = basic_path + 'test/'

path_train_images = path_train + 'images/'
path_train_masks = path_train + 'masks/'
path_test_images = path_test + 'images/'

train_df = pd.read_csv('./input/train.csv', index_col='id', usecols=[0])
depths_df = pd.read_csv('./input/depths.csv', index_col='id')
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

train_df['images'] = [np.array(load_img('{}/{}.png'.format(path_train_images, idx), grayscale=True)) / 255 for idx in
                      tqdm(train_df.index)]
train_df['masks'] = [np.array(load_img('{}/{}.png'.format(path_train_masks, idx), grayscale=True)) / 255 for idx in
                     tqdm(train_df.index)]

train_df['coverage'] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)


def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i:
            return i


train_df['coverage_class'] = train_df.coverage.map(cov_to_class)

n_fold = 5
train_df.sort_values('coverage_class', inplace=True)
train_df['fold'] = (list(range(n_fold)) * train_df.shape[0])[:train_df.shape[0]]
subsets = [train_df[train_df['fold'] == i] for i in range(n_fold)]

convert = lambda x: np.array(x.tolist()).reshape(-1, img_size_target, img_size_target, 1)


def get_train_val(n_fold_data, fold_idx):
    train = pd.concat([_ for idx, _ in enumerate(n_fold_data) if idx != fold_idx])
    valid = n_fold_data[fold_idx]
    x_train = convert(train['images'])
    x_valid = convert(valid['images'])
    y_train = convert(train['masks'])
    y_valid = convert(valid['masks'])
    return x_train, x_valid, y_train, y_valid


for fold_idx in range(n_fold):
    x_train, x_valid, y_train, y_valid = get_train_val(subsets, 0)

    aug = ImageAugmenter(
        x_train,
        mode='SIMPLE',
        labels=y_train,
        end_to_end=True)

    x_train, y_train = aug.generate()

    best_val_acc = 0
    tol = 0
    while best_val_acc < 0.962:
        print('Training fold {}...'.format(fold_idx))
        tol += 1
        # model
        input_layer = Input((img_size_target, img_size_target, 1))
        output_layer = build_model(input_layer, 16, 0.3)
        model = Model(input_layer, output_layer)
        model.compile(loss=bce_dice_loss, optimizer='adam', metrics=[bce_dice_loss, 'acc'])

        early_stopping = EarlyStopping(monitor='val_acc', mode='max', patience=10, verbose=1)
        model_checkpoint = ModelCheckpoint("./model_{}.model".format(fold_idx), monitor='val_acc',
                                           mode='max', save_best_only=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', mode='max', factor=0.2, patience=5, min_lr=0.00001, verbose=1)
        # reduce_lr = ReduceLROnPlateau(factor=0.2, patience=5, min_lr=0.00001, verbose=1)

        epochs = 200
        batch_size = 32
        history = model.fit(x_train, y_train,
                            validation_data=[x_valid, y_valid],
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[early_stopping, model_checkpoint, reduce_lr],
                            verbose=1)

        best_val_acc = max(history.history['val_acc']) if tol <= 5 else 1

    with open('log.txt', 'a+') as f:
        f.write('Model {}, {}\n'.format(fold_idx, best_val_acc))
