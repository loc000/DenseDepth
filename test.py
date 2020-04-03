import argparse
import os

import wget

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
import cv2
import numpy as np
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import display_images
from utils import predict as depth_estimation
from matplotlib import pyplot as plt

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='examples/*.png', type=str, help='Input filename or folder.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

current_directory = os.getcwd()
os.chdir(os.path.dirname(__file__))
if not os.path.isfile(args.model):
    print("{} not found, downloading...".format(args.model))
    wget.download("https://s3-eu-west-1.amazonaws.com/densedepth/nyu.h5", args.model)
print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)
os.chdir(current_directory)

print('\nModel loaded ({0}).'.format(args.model))


# Input images
# inputs = load_images(glob.glob(args.input))

def predict(cv2_image):
    # inputs = load_image("examples/470_image.png")
    x = np.clip(
        np.asarray(cv2.cvtColor(cv2.resize(cv2_image, (640, 480)), cv2.COLOR_BGR2RGB), dtype=float) / 255,
        0, 1)
    inputs = np.stack([x], axis=0)
    del x
    print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

    # Compute results
    outputs = depth_estimation(model, inputs)
    outputs = (np.squeeze(outputs) * 255).astype(np.uint8)
    cv2.imshow("full depth", outputs)
    q1 = np.percentile(outputs, 25)
    outputs[outputs > q1] = 0
    cv2.imshow("color", cv2_image)
    cv2.imshow("depth > {}".format(q1), outputs)
    cv2.waitKey(0)
    print(outputs.shape)
    # exit()
    #
    # # matplotlib problem on ubuntu terminal fix
    # # matplotlib.use('TkAgg')
    #
    # # Display results
    # viz = display_images(outputs.copy(), inputs.copy())
    # plt.figure(figsize=(10, 5))
    # plt.imshow(viz)
    # plt.savefig('test.png')
    # plt.show()


if __name__ == '__main__':
    inputs = cv2.imread("examples/470_image.png")
    predict(inputs)
