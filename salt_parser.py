import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm


# Quick RLEncoding needed for submission generation.
# Source: another kernel, thanks!
def rle_encode(im):
    pixels = im.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


class SaltParser(object):
    """
    Parser for Salt Competition.
    """

    def __init__(self,
                 data_src='../input/',
                 image_size=(128, 128),
                 pad_images=False,
                 grayscale=True,
                 load_test_data=True):

        self.data_src = data_src
        self.image_size = image_size
        self.pad_images = pad_images
        self.grayscale = grayscale
        self.load_test_data = load_test_data

        self.train_df = None
        self.test_df = None
        self.padding_pixels = None

        self.X_train = []
        self.y_train = []
        self.X_test = []

        self.orig_image_size = (101, 101)

        """
        # Arguments:

            data_src: directory containing data
            image_size: tuple specifying final image size
            pad_images: whether images should be padded or resized
            grayscale: whether to load images as grayscale
            load_test_data: whether to load test data

        """

    def initialize_data(self):

        """
        Initialize processing by loading .csv files.
        """

        train_df = pd.read_csv('{}train.csv'.format(self.data_src),
                               usecols=[0], index_col='id')
        depths_df = pd.read_csv('{}depths.csv'.format(self.data_src),
                                index_col='id')

        self.train_df = train_df.join(depths_df)
        self.test_df = depths_df[~depths_df.index.isin(train_df.index)]

    def load_data(self):

        """
        Load images and masks from training set.

        # Returns:
            self.X_train: np.array of training images
            self.y_train: np.array of training masks
            self.X_test: np.array of test images
        """

        print('Loading training set.')
        # Loop over ids in train_df
        for i in tqdm(self.train_df.index):
            # Load image and mask according to ID
            img_src = '{}train/images/{}.png'.format(self.data_src, i)
            mask_src = '{}train/masks/{}.png'.format(self.data_src, i)
            # Specify if image should be loaded in grayscale.
            if self.grayscale:
                img_temp = cv2.imread(img_src, 0)
            else:
                img_temp = cv2.imread(img_src)
            # Load mask
            mask_temp = cv2.imread(mask_src, 0)
            # Resize or pad image and mask
            if self.orig_image_size != self.image_size:
                if self.pad_images:
                    img_temp = self.__pad_image(img_temp)
                    mask_temp = self.__pad_image(mask_temp)
                else:
                    img_temp = cv2.resize(img_temp, self.image_size)
                    mask_temp = cv2.resize(mask_temp, self.image_size)
            # Append processed image and mask
            self.X_train.append(img_temp)
            self.y_train.append(mask_temp)

        # Transform into arrays
        self.X_train = np.asarray(self.X_train)
        self.y_train = np.asarray(self.y_train)
        # If images were loaded as grayscale, they are loaded as (HxW) arrays
        # Dimensions must be expanded for the model to be trained.
        if self.grayscale:
            self.X_train = np.expand_dims(self.X_train, -1)
        # Mask must be expanded obligatorily, as they are 1-channel by default.
        self.y_train = np.expand_dims(self.y_train, -1)

        # Output information about training set.
        print('Training set ready.')
        print('X_train shape: {}'.format(self.X_train.shape))
        print('y_train shape: {}'.format(self.y_train.shape))
        print('X_train - min: {}, max: {}'.format(
            np.min(self.X_train), np.max(self.X_train)))
        print('y_train - min: {}, max: {}'.format(
            np.min(self.y_train), np.max(self.y_train)))

        # Load test data.
        # Perform similar steps to the training processing part,
        # but there are no masks to be loaded.
        if self.load_test_data:
            print('Loading test set.')
            for i in tqdm(self.test_df.index):
                img_src = '{}test/images/{}.png'.format(self.data_src, i)
                if self.grayscale:
                    img_temp = cv2.imread(img_src, 0)
                else:
                    img_temp = cv2.imread(img_src)
                if self.orig_image_size != self.image_size:
                    if self.pad_images:
                        img_temp = self.__pad_image(img_temp)
                    else:
                        img_temp = cv2.resize(img_temp, self.image_size)
                self.X_test.append(img_temp)

            self.X_test = np.asarray(self.X_test)
            if self.grayscale:
                self.X_test = np.expand_dims(self.X_test, -1)

            print('Test set ready.')
            print('X_test shape: {}'.format(self.X_test.shape))
            print('X_test - min: {}, max: {}'.format(
                np.min(self.X_test), np.max(self.X_test)))

            return self.X_train, self.y_train, self.X_test

        return self.X_train, self.y_train

    def compute_coverage(self):

        """
        Compute salt coverage of each mask. This will serve as a basis for
        stratified split between training and validation sets.

        # Returns:
            self.train_df: training DF containing coverage information.
        """

        print('Compute mask coverage for each observation.')

        def cov_to_class(val):
            for i in range(0, 11):
                if val * 10 <= i:
                    return i

        # Output percentage of area covered by class
        self.train_df['coverage'] = np.mean(self.y_train / 255., axis=(1, 2))
        # Coverage must be split into bins, otherwise stratified split will not be possible,
        # because each coverage will occur only once.
        self.train_df['coverage_class'] = self.train_df.coverage.map(
            cov_to_class)

        return self.train_df

    def predictions_rle_encode(self,
                               y_pred_test,
                               confidence_threshold_best):

        """
        Run Length Encoding of predictions.
        This is needed for submission output.

        # Arguments:
            y_pred_test: model predictions
            confidence_threshold_best: confidence threshold, according to which
                masks are set to 1/0.
        # Returns:
            y_test_pred_rle: RLEncoded predictions.
        """

        # If images were padded, this padding must now be removed.
        # Otherwise encoding method will fail to properly encode predictions and
        # score will be bad.
        if self.pad_images:
            print('Remove padding from images.')
            y_min_pad, y_max_pad, x_min_pad, x_max_pad = self.padding_pixels[
                                                             0], self.padding_pixels[1], self.padding_pixels[2], \
                                                         self.padding_pixels[3]
            y_pred_test = y_pred_test[:, y_min_pad:-
            y_max_pad, x_min_pad:-x_max_pad, 0]

        # Situation is similar for previously resized images.
        # They must be resized again to their original size before encoding.
        else:
            y_pred_test = np.asarray([cv2.resize(x, self.orig_image_size)
                                      for x in y_pred_test])

        assert y_pred_test.shape == (18000, 101, 101), '\
        Test predictions shape must be equal to (18000, 101, 101).'

        print('Test predictions shape: {}'.format(y_pred_test.shape))

        # Perform mask predictions binarization and RLEncoding.
        y_test_pred_rle = {idx:
                               rle_encode(y_pred_test[i] > confidence_threshold_best)
                           for i, idx in enumerate(
            tqdm(self.test_df.index.values))}

        return y_test_pred_rle

    def generate_submission(self, y_test_pred_rle):

        """
        Submission generation based on encoded model predictions.

        # Arguments:
            y_test_pred_rle: RLEncoded predictions.
        # Returns:
            submission: generated submission.
        """

        submission = pd.DataFrame.from_dict(y_test_pred_rle, orient='index')
        submission.index.names = ['id']
        submission.columns = ['rle_mask']

        return submission

    def return_padding_borders(self):
        """
        Return padding borders in case intermediate operations on original images
        are needed.

        # Returns:
            self.padding_pixels: tuple of padding borders.
        """
        return self.padding_pixels

    def __pad_image(self, img):

        """
        Helper function for images padding.

        # Arguments:
            img: image as np.array

        # Returns:
            img: padded image as np.array
        """

        pad_floor = np.floor(
            (np.asarray(self.image_size) - np.asarray(self.orig_image_size)) / 2)
        pad_ceil = np.ceil((np.asarray(self.image_size) -
                            np.asarray(self.orig_image_size)) / 2)

        self.padding_pixels = np.asarray(
            (pad_floor[0], pad_ceil[0], pad_floor[1], pad_ceil[1])).astype(np.int32)

        y_min_pad, y_max_pad, x_min_pad, x_max_pad = self.padding_pixels[
                                                         0], self.padding_pixels[1], self.padding_pixels[2], \
                                                     self.padding_pixels[3]

        img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad,
                                 x_min_pad, x_max_pad,
                                 cv2.BORDER_REFLECT_101)

        assert img.shape[:2] == self.image_size, '\
        Image after padding must have the same shape as input image.'

        return img
