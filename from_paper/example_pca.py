import sys
# import numpy and matplotlib colormaps
import numpy as np
import matplotlib.cm as cm


# import tinyfacerec modules
# from tinyfacerec . visual import subplot
from from_paper.subspace import pca
from from_paper.util import normalize, as_row_matrix, read_images

from from_paper.visual import subplot

from from_paper.subspace import project, reconstruct

# append tinyfacerec to module search path
sys.path.append("..")
# read images


[x, y] = read_images(r"C:\Users\Worker\Pycharm\Projects\facerec\files")
# perform a full pca
[D, W, mu] = pca(as_row_matrix(x), y)


# turn the first (at most ) 16 eigenvectors into grayscale
# images ( note : eigenvectors are stored by column !)
E = []
for i in range(min(len(x), 16)):
    e = W[:, i].reshape(x[0].shape)
    E.append(normalize(e, 0, 255))
# plot them and store the plot to " python_eigenfaces .pdf"
subplot(
    title=" Eigenfaces AT&T Facedatabase ", images=E, rows=4, cols=4, sptitle="Eigenface", colormap=cm.jet
)


# reconstruction steps
steps = [i for i in range(10, min(len(x), 320), 20)]
E = []
for i in range(min(len(steps), 16)):
    numEvs = steps[i]
    P = project(W[:, 0: numEvs], x[0].reshape(1, -1), mu)
    R = reconstruct(W[:, 0: numEvs], P, mu)
    # reshape and append to plots
    R = R.reshape(x[0].shape)
    E.append(normalize(R, 0, 255))
    # plot them and store the plot to " python_reconstruction . pdf "
subplot(
    title=" Reconstruction AT&T Facedatabase ", images=E, rows=4, cols=4,
    sptitle="Eigenvectors", sptitles=steps, colormap=cm.gray
)

