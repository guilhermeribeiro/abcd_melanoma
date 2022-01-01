import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import math

# Reading image as RGB pattern
image = cv2.imread(
    'D:\\Mega\\Doutorado-UFG\\Disciplinas\\Processamento Digital de Imagem\\Artigo\\complete_mednode_dataset\\melanoma\\132357.jpg')


# C feature - Color
r = image[:, :, 0].mean()
g = image[:, :, 1].mean()
b = image[:, :, 2].mean()

# Convert RGB image to grayscale
image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

(thresh, image_gray) = cv2.threshold(image_gray, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

def np_hist_to_cv(np_histogram_output):
    counts, bin_edges = np_histogram_output
    return counts.ravel().astype('float32')

G_x = cv2.reduce(image_gray/255, 0, cv2.REDUCE_SUM)
G_y = cv2.reduce(image_gray/255, 1, cv2.REDUCE_SUM)

h1 = np.histogram(G_x)
h2 = np.histogram(G_y.T)

# A feature - Assymetry
comparation = cv2.compareHist(np_hist_to_cv(h1), np_hist_to_cv(h2), cv2.HISTCMP_CORREL)


contours, hierarchy = cv2.findContours(image_gray, 2, 1)
# print(len(contours))
cnt = contours

for i in range(len(cnt)):
    (x, y), radius = cv2.minEnclosingCircle(cnt[i])
    center = (int(x), int(y))
    radius = int(radius)
    print(radius)
    plt.text(x-21, y+15, '+', fontsize=25, color='red')
    plt.text(10, -10, 'Centro: ' +str(center), fontsize=11, color='red')
    plt.imshow(image_gray, cmap='gray')
    plt.show(block=True)

pi = 3.14

d = math.sqrt((4 * (pi * (radius^2)))/pi)


d = radius