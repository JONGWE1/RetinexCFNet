import onnxruntime
import time

import cv2
import glob
import os
import argparse
import numpy as np


def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


def get_image_list(image_path):
    """Get image list"""
    valid_suffix = [
        '.JPEG', '.jpeg', '.JPG', '.jpg', '.BMP', '.bmp', '.PNG', '.png'
    ]
    image_list = []
    image_dir = None
    if os.path.isfile(image_path):
        if os.path.splitext(image_path)[-1] in valid_suffix:
            image_list.append(image_path)
        else:
            image_dir = os.path.dirname(image_path)
            with open(image_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if len(line.split()) > 1:
                        line = line.split()[0]
                    image_list.append(os.path.join(image_dir, line))
    elif os.path.isdir(image_path):
        image_dir = image_path
        for root, dirs, files in os.walk(image_path):
            for f in files:
                if '.ipynb_checkpoints' in root:
                    continue
                if os.path.splitext(f)[-1] in valid_suffix:
                    image_list.append(os.path.join(root, f))
    else:
        raise FileNotFoundError(
            '`--image_path` is not found. it should be an image file or a directory including images'
        )

    if len(image_list) == 0:
        raise RuntimeError('There are not image file in `--image_path`')

    return image_list, image_dir


def pad_(img_to_pad, d=16):
    width_ex = img_to_pad.shape[3] % d
    height_ex = img_to_pad.shape[2] % d
    left, right, top, bottom = 0, 0, 0, 0

    if width_ex != 0:
        width_topad = d - width_ex
        left = int(width_topad / 2)
        right = int(width_topad - left)
    if height_ex != 0:
        height_topad = d - height_ex
        top = int(height_topad / 2)
        bottom = int(height_topad - top)

    img_to_pad = np.pad(
        img_to_pad,
        [(0, 0), (0, 0), (top, bottom), (left, right)], mode='reflect')

    return img_to_pad, left, right, top, bottom


def pad_back_(img_to_unpad, left, right, top, bottom):
    _, _, height, width = img_to_unpad.shape
    return img_to_unpad[:, :, top: height - bottom, left: width - right]


def rgb2grey(rgb_img, need_channels=1):
    rgb_img = np.transpose(rgb_img, (0, 2, 3, 1))
    kernel = np.array([[0.299, 0.299, 0.299],
                       [0.587, 0.587, 0.587],
                       [0.114, 0.114, 0.114]], dtype='float32')
    gray = np.matmul(rgb_img, kernel)
    return np.transpose(gray, (0, 3, 1, 2)) if need_channels == 3 \
        else np.transpose(gray, (0, 3, 1, 2))[:, 0, :, :][:, np.newaxis, :, :]


def gradient(input_tensor, direction):
    n, channels, h, w = input_tensor.shape
    if direction == "x":
        input_tensor = np.pad(
            input_tensor, pad_width=[(0, 0), (0, 0), (0, 0), (0, 1)], mode='constant', constant_values=0)
        dx = input_tensor[:, :, :, 1:] - input_tensor[:, :, :, :w]
        return np.abs(dx)
    elif direction == "y":
        input_tensor = np.pad(
            input_tensor, pad_width=[(0, 0), (0, 0), (0, 1), (0, 0)], mode='constant', constant_values=0)
        dy = input_tensor[:, :, :h, :] - input_tensor[:, :, 1:, :]
        return np.abs(dy)


def main():
    args = parse_args()
    # model = onnxruntime.InferenceSession(args.model_path, providers=['CPUExecutionProvider'])
    model = onnxruntime.InferenceSession(args.model_path, providers=['CUDAExecutionProvider'])

    image_list, image_dir = get_image_list(args.image_path)
    print('Number of predict images = {}'.format(len(image_list)))

    inference_time = 0
    print("Start to predict...")
    for image in image_list:
        print(image)
        lowlight_image = cv2.imread(image)
        lowlight_image = cv2.cvtColor(lowlight_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        v_max, v_min = lowlight_image.max(), lowlight_image.min()
        lowlight_image = (lowlight_image - v_min) / (v_max - v_min + 1e-10)
        lowlight_image = np.transpose(lowlight_image, axes=[2, 0, 1])  # HWC to CHW

        lowlight = lowlight_image[np.newaxis, :]
        lowlight_grey = rgb2grey(lowlight, need_channels=1)
        lowlight_x = gradient(lowlight_grey, "x")
        lowlight_y = gradient(lowlight_grey, "y")
        guidance1 = lowlight_grey + lowlight_x
        guidance2 = lowlight_grey + lowlight_y
        guidance3 = lowlight_grey + lowlight_x + lowlight_y
        Illumination_guide = np.concatenate([lowlight_grey, guidance1, guidance2, guidance3], axis=1)

        lowlight, left, right, top, bottom = pad_(lowlight)
        Illumination_guide, left, right, top, bottom = pad_(Illumination_guide)

        st = time.time()
        enhanced_image = model.run([], {'Illumination_guide': Illumination_guide,
                                        'lowlight': lowlight})[0]
        inference_time += time.time() - st

        enhanced_image = pad_back_(enhanced_image, left, right, top, bottom)
        enhanced_image = np.transpose(enhanced_image, (0, 2, 3, 1)) * 255
        enhanced_image = np.squeeze(enhanced_image, axis=0).astype('uint8')

        # get the saved name
        if image_dir is not None:
            im_file = image.replace(image_dir, '')
        else:
            im_file = os.path.basename(image)
        if im_file[0] == '/' or im_file[0] == '\\':
            im_file = im_file[1:]

        # save prediction image
        save_image_path = os.path.join(args.save_dir, im_file)
        mkdir(save_image_path)
        cv2.imwrite(save_image_path, cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR))

    print("Total inference time: %f s, average inference time: %f s"
          % (inference_time, inference_time / len(image_list)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for prediction',
        type=str,
        default=None)
    parser.add_argument(
        '--image_path',
        dest='image_path',
        help=
        'The path of image, it can be a file or a directory including images',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the predicted results',
        type=str,
        default='./output/result')
    return parser.parse_args()


if __name__ == "__main__":
    main()
