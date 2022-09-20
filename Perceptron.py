


#    @copy by sobhan siamak

import numpy as np
import math
# import pandas as pd
# import cv2
import random
from PIL import Image
import glob
from sklearn.linear_model import Perceptron
# from sklearn.neural_network import multilayer_perceptron
from sklearn.metrics import accuracy_score
# from skimage.util import random_noise



# p is the percentage of noise ==== p=0.1 or p=0.2
p = 0.2
#Train Images
dict ={'a':0, '1':1, '2':2, '3':3, '4':4, '5':5}
def LoadData():
    imageNames = glob.glob("Train Data/*.png")
    lbls = []
    img = []
    imgnoise = []
    bwimg = []
    bwnoise = []
    # c = 0
    for i in imageNames:
        s = i.split('//')[-1]
        s = s.split('_')[1]
        lbls += [dict[s]]
        im = Image.open(i)
        # im2 = cv2.imread(i)
        # im.show()
        # print(im.size)
        im = im.resize((256,256))
        #convert image to grayscale
        m,n = im.size
        im = im.convert('L')
        #grayscale noisy image
        im2 = im.load()
        imnoisy = sp_noise(im2, m, n, p)
        #convert image to binary
        bw = im.point(lambda x: 0 if x<64 else 255, '1')
        bw2 = bw.load()
        bwimgnoise = sp_noisebw(bw2, m, n, p)
        # c+=1
        # if c == 30:
        #     im.show()
        #     print(im.size[0])
        #     # im2 = sp_noise(im, 0.2)
        #     # im2.show()
        #     # noise_img = random_noise(im, mode='s&p', amount=0.2)
        #     im2 = im.load()
        #     imgnoise = sp_noise(im2, m, n, 0.008)
        #     # imgnoise.show()
        #     cv2.imshow('noisy image', imgnoise)
        #     cv2.waitKey(0)
        #
        #     bw.show()
        #     print(bw.size[1])
        a = np.array(im)
        a = np.reshape(a,(256*256, 1))
        a = np.squeeze(a)
        img += [a]

        anoisy = np.array(imnoisy)
        anoisy = np.reshape(anoisy,(256*256, 1))
        anoisy = np.squeeze(anoisy)
        imgnoise += [anoisy]

        bwa = np.array(bw)
        bwa = np.reshape(bwa,(256*256, 1))
        bwa = np.squeeze(bwa)
        # convert binary to bipolar
        bwa = np.where(bwa==0,-1,1)
        bwimg += [bwa]

        bwnoisy = np.array(bwimgnoise)
        bwnoisy = np.reshape(bwnoisy,(256*256, 1))
        bwnoisy = np.squeeze(bwnoisy)
        bwnoisy = np.where(bwnoisy==0,-1,1)
        bwnoise += [bwnoisy]


    return np.array(img), np.array(lbls), np.array(bwimg), np.array(imgnoise), np.array(bwnoise)

#Test Images
tdict ={'e':0, 'd':1, 'v':2, '6':3, '4':4, '5':5}
def LoadTestData():
    timageNames = glob.glob("Test Data/*.png")
    tlbls = []
    timg = []
    tbwimg = []
    for i in timageNames:
        ts = i.split('//')[-1]
        ts = ts.split('_')[1]
        tlbls += [tdict[ts]]
        tim = Image.open(i)
        # im.show()
        # print(im.size)
        tim = tim.resize((256,256))
        tim = tim.convert('L')
        bwt = tim.point(lambda x: 0 if x<64 else 255, '1')

        ta = np.array(tim)
        ta = np.reshape(ta,(256*256, 1))
        ta = np.squeeze(ta)
        timg += [ta]

        tbwa = np.array(bwt)
        tbwa = np.reshape(tbwa, (256*256, 1))
        tbwa = np.squeeze(tbwa)
        tbwa = np.where(tbwa==0,-1,1)
        tbwimg += [tbwa]
    return np.array(timg), np.array(tlbls), np.array(tbwimg)


def sp_noise(image, m, n, precent):
    imgNoise = np.zeros((m,n), np.uint8)
    threshold = 1 - precent
    for i in range(m):
        for j in range(n):
            rnd = random.random()
            if rnd < precent:
                imgNoise[i][j] = 0
            elif rnd > threshold:
                imgNoise[i][j] = 255
            else:
                imgNoise[i][j] = image[j,i]


    return imgNoise

def sp_noisebw(image, m, n, precent):
    imgNoise = np.zeros((m,n))
    threshold = 1 - precent
    for i in range(m):
        for j in range(n):
            rnd = random.random()
            if rnd < precent:
                imgNoise[i][j] = 0
            elif rnd > threshold:
                imgNoise[i][j] = 1
            else:
                imgNoise[i][j] = image[j,i]


    return imgNoise





a1, b1, bw1, a1noisy, bw1noisy = LoadData()

at, bt, bwt = LoadTestData()
#

# grayscale image perceptron for Train and Test Data
ppn = Perceptron(max_iter=1000, alpha=0.001, eta0=0.001)
ppn.fit(a1, b1)
y_pred = ppn.predict(a1)
ty_pred = ppn.predict(at)
anoisy_pred = ppn.predict(a1noisy)
# bwnoisy_pred = ppn.predict(bw1noisy)
print('Accuracy of Train Data for gray scale is: %.2f' % accuracy_score(b1, y_pred))
print('Accuracy of Test Data for gray scale is: %.2f' % accuracy_score(bt, ty_pred))
print('Accuracy of Noisy Train Data as a Test for gray scale is: %.2f' % accuracy_score(b1, anoisy_pred))

# print('Accuracy of Noisy Test Data for gray scale is: %.2f' % accuracy_score(b1, anoisy_pred))
# print('Accuracy of Noisy Test Data for binary(bipolar) images is: %.2f' % accuracy_score(b1, bwnoisy_pred))



# print('Real Labels of Train Data in Gray Scale :',b1)
# print('Predict Labels of Train Data in Gray Scale :',y_pred)
# print('Real Labels of Test Data in Gray Scale :',bt)
# print('Predict Labels of Test Data in Gray Scale :',ty_pred)

#binary(bipolar) image perceptron for Train and Test Data
bwp = Perceptron(max_iter=1000, alpha=0.001, eta0=0.001)
bwp.fit(bw1, b1)
bwy_pred = bwp.predict(bw1)
tbwy_pred = bwp.predict(bwt)
bwnoisy_pred = bwp.predict(bw1noisy)


print('Accuracy of Train Data for binary(bipolar) images is: %.2f' % accuracy_score(b1, bwy_pred))
print('Accuracy of Test Data for binary(bipolar) images is: %.2f' % accuracy_score(bt, tbwy_pred))
print('Accuracy of Noisy Train Data as a Test for binary(bipolar) images is: %.2f' % accuracy_score(b1, bwnoisy_pred))

# print('Real Labels of Train Data in binary(bipolar) images is:',b1)
# print('Predict Labels of Train Data in binary(bipolar) images is:',bwy_pred)
# print('Real Labels of Test Data in binary(bipolar) images is:',bt)
# print('Predict Labels of Test Data in binary(bipolar) images is:',tbwy_pred)



# print('Accuracy of Noisy Test Data for gray scale is: %.2f' % accuracy_score(b1, anoisy_pred))
# print('Accuracy of Noisy Test Data for binary(bipolar) images is: %.2f' % accuracy_score(b1, bwnoisy_pred))


#Multiclass single layer Perceptron from scratch without toolbox

def sigmoid(z):
    return 1 / (1 + math.exp(-z))


# Discerete Perceptron
class DPerceptron:
    def __init__(self, learning_rate=0.01, n_iteration=1000):
        self.lr = learning_rate
        self.n_iter = n_iteration
        self.act_func = self._step_func
        self.weights = None
        self.bias = None

    def makeweight(self, n):
        self.weights = np.zeros(n)
        return  self.weights


    def fit(self, X, y):
        num = 65536
        wtV0 = makeWeight(num)
        wtV1 = makeWeight(num)
        wtV2 = makeWeight(num)
        wtV3 = makeWeight(num)
        wtV4 = makeWeight(num)
        wtV5 = makeWeight(num)
        weightVs = [wtV0, wtV1, wtV2, wtV3, wtV4, wtV5]

        for i in range(self.n_iter):
            pass

    def predict(self, X):
        l_output = np.dot(X, self.weights) + self.bias
        y_pred = self.act_func(l_output)
        return y_pred

    def _step_func(self, inputs, weights):
        threshold = 0.0
        summation = 0.0
        for input, weight in zip(inputs, weights):
            summation += input * weight
        return 1.0 if summation > threshold else 0.0


# Continous Perceptron

class CPerceptron:
    def __init__(self, learning_rate=0.01, n_iteration=1000):
        self.lr = learning_rate
        self.n_iter = n_iteration
        self.act_func = self._sig_func
        self.weights = None
        self.bias = None


    def makeweight(self, n):
        self.weights = np.zeros(n)
        for i in range(n):
            randFloat = random.uniform(-1, 1)
            weights[i] = "%.2f" % randFloat
        return  self.weights


    def fit(self, X, y):
        num = 65536
        wtV0 = makeWeight(num)
        wtV1 = makeWeight(num)
        wtV2 = makeWeight(num)
        wtV3 = makeWeight(num)
        wtV4 = makeWeight(num)
        wtV5 = makeWeight(num)
        weightVs = [wtV0, wtV1, wtV2, wtV3, wtV4, wtV5]
        for i in range(self.n_iter):
            pass

    def predict(self, X):
        pass



    def _sig_func(self, inputs, weights):
        threshold = 0.0
        summation = 0.0
        for input, weight in zip(inputs, weights):
            summation += input * weight
        return sigmoid(summation)





#DPerceptron
#CPerceptron