import sys, os
sys.path.append("../..")
# import facerec modules
from facerec.feature import Fisherfaces
from facerec.distance import EuclideanDistance
from facerec.classifier import NearestNeighbor
from facerec.model import PredictableModel
from facerec.validation import KFoldCrossValidation
from facerec.visual import subplot
from facerec.util import minmax_normalize
# import numpy, matplotlib and logging
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import logging

def read_images(path, sz=None):
    """Reads the images in a given folder, resizes images on the fly if size is given.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes

    Returns:
        A list [X,y]

            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
    """
    c = 0
    X,y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = Image.open(os.path.join(subject_path, filename))
                    im = im.convert("L")
                    # resize to given size (if given)
                    if sz is not None:
                        im = im.resize(sz, Image.ANTIALIAS)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError as ioe:
                    print("I/O error {0}".format(ioe))
                except Exception as exep:
                    print("Unexpected error:", exep)
                    raise
            c = c+1
    return [X,y]


def read_img(paths, sz=None):
    c = 0
    X, y = [], []
    # paths = [
    #     r'C:\Users\Worker\Pycharm\Projects\facerec\files'
    #     r'\Painting_Art_Sunrises_and_sunsets_Mountains_Deer_529666_3840x2400.jpg',
    #     r'C:\Users\Worker\Pycharm\Projects\facerec\files\zagruzhennoe--1.jpg'
    # ]
    for filename in paths:
        try:
            im = Image.open(filename)
            im = im.convert("L")
            # resize to given size (if given)
            if sz is not None:
                im = im.resize(sz, Image.ANTIALIAS)
            X.append(np.asarray(im, dtype=np.uint8))
            y.append(c)
        except IOError as ioe:
            print("I/O error {0}".format(ioe))
        except Exception as exep:
            print("Unexpected error:", exep)
            raise
    c = c + 1
    return [X, y]


if __name__ == "__main__":
    # This is where we write the images, if an output_dir is given
    # in command line:
    out_dir = None
    # You'll need at least a path to your image data, please see
    # the tutorial coming with this source code on how to prepare
    # your image data:
    # if len(sys.argv) < 2:
    #     print("USAGE: facerec_demo.py </path/to/images>")
    #     sys.exit()
    # Now read in the image data. This must be a valid path!
    [X, y] = read_images(r'C:\Users\Worker\Pycharm\Projects\facerec\files')
    [X, y] = read_img(paths=[
        r'C:\Users\Worker\Pycharm\Projects\facerec\files'
        r'\Painting_Art_Sunrises_and_sunsets_Mountains_Deer_529666_3840x2400.jpg',
        r'C:\Users\Worker\Pycharm\Projects\facerec\files\zagruzhennoe--1.jpg'
    ])
    # Then set up a handler for logging:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # Add handler to facerec modules, so we see what's going on inside:
    logger = logging.getLogger("facerec")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    # Define the Fisherfaces as Feature Extraction method:
    feature = Fisherfaces()
    # Define a 1-NN classifier with Euclidean Distance:
    classifier = NearestNeighbor(dist_metric=EuclideanDistance(), k=1)
    # Define the model as the combination
    model = PredictableModel(feature=feature, classifier=classifier)
    # Compute the Fisherfaces on the given data (in X) and labels (in y):
    model.compute(X, y)
    # Then turn the first (at most) 16 eigenvectors into grayscale
    # images (note: eigenvectors are stored by column!)
    E = []
    for i in range(min(model.feature.eigenvectors.shape[1], 16)):
        e = model.feature.eigenvectors[:,i].reshape(X[0].shape)
        E.append(minmax_normalize(e,0,255, dtype=np.uint8))
    # Plot them and store the plot to "python_fisherfaces_fisherfaces.pdf"
    # subplot(title="Fisherfaces", images=E, rows=4, cols=4, sptitle="Fisherface", colormap=cm.jet, filename="fisherfaces.png")
    subplot(title="Fisherfaces", images=E, rows=4, cols=4, sptitle="Fisherface", colormap=cm.jet)
    # Perform a 10-fold cross validation
    cv = KFoldCrossValidation(model, k=10)
    cv.validate(X, y)
    # And print the result:
    print(cv)