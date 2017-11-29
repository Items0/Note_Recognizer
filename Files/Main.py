import numpy as np
from PIL import Image as im
import PIL.ImageStat as imageStat  # fajna clasa analizująca  imageStat.Stat(Image)._get[co chcę (mean, stddev ...)]
import matplotlib.pyplot as plt
from skimage.filters.edges import convolve
from skimage.feature import match_template, peak_local_max
from skimage import io, data, draw, measure
from skimage.draw import circle_perimeter, set_color
import math

MULTIPLE_STD_PARAM = 2.0
FILE_SUFIX = ""
MASK_EDGE_HORIZONTAL = np.array([[1, 2, 1],
                                 [0, 0, 0],
                                 [-1, -2, -1]])
MASK_EDGE_VERTICAL = np.array([[1, 0, -1],
                               [2, 0, -2],
                               [1, 0, -1]]) / 8
MASK_EDGE_LAPLACE = np.array([[-1, -1, -1],
                              [-1, 8, -1],
                              [-1, -1, -1]]) / 20
MASK_MEAN = np.array([[1, 1, 1],
                      [1, 2, 1],
                      [1, 1, 1]]) / 20
MASK_DILATATION = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]])

IDENT_PARAM = 0.52 # <-1,1>

def readBitmapFromFile(fileName):
    path = "Photos/" + fileName + ".jpg"
    image = im.open(path)
    image = image.resize((int(image.size[0] / np.sqrt(6)), int(image.size[1] / np.sqrt(6))))
    return np.asarray(image)


def writeBitmapToFile(bitmap, fileName):
    image = im.fromarray(np.uint8(bitmap))
    path = "Done/" + fileName + FILE_SUFIX + ".jpg"
    image.save(path)


def calculateAveragesRGBColor(bitmap):
    rSum, gSum, bSum = 0, 0, 0
    for row in bitmap:
        for column in row:
            rSum += column[0]
            gSum += column[1]
            bSum += column[2]
    pixelsCount = len(bitmap) * len(bitmap[0])
    return [rSum / pixelsCount, gSum / pixelsCount, bSum / pixelsCount]


def thresholding(bitmap, avegaresRGB, std=0):
    threshBitmap = []
    std *= MULTIPLE_STD_PARAM
    rThresh = avegaresRGB[0] - std if avegaresRGB[0] - std > 0 else 0
    gThresh = avegaresRGB[1] - std if avegaresRGB[1] - std > 0 else 0
    bThresh = avegaresRGB[2] - std if avegaresRGB[2] - std > 0 else 0
    print(avegaresRGB)
    print(rThresh, " ", gThresh, " ", bThresh)
    print(std)
    for row in bitmap:
        threshRow = []
        for cell in row:
            r = 255 if rThresh < cell[0] else 0
            g = 255 if gThresh < cell[1] else 0
            b = 255 if bThresh < cell[2] else 0
            threshRow.append([r, g, b])
        threshBitmap.append(threshRow)
    return threshBitmap


def createBitmapWithMask(bitmap, mask):
    return np.abs(convolve(bitmap, mask[:, :, None]))


def edgeDetect(bitmap):
    bitmap = createBitmapWithMask(bitmap, MASK_MEAN)
    bitmap = createBitmapWithMask(bitmap, MASK_EDGE_LAPLACE)
    return bitmap


def doNegative(bitmap):
    negativeBitmap = []
    for row in bitmap:
        reverseRow = []
        for column in row:
            r = 255 - column[0]
            g = 255 - column[1]
            b = 255 - column[2]
            reverseRow.append([r, g, b])
        negativeBitmap.append(reverseRow)
    return negativeBitmap


def makeGrayScale(bitmap):
    grayBitmap = []
    for row in bitmap:
        grayRow = []
        for cell in row:
            color = int(0.299 * cell[0] + 0.587 * cell[1] + 0.114 * cell[2])
            grayRow.append([color, color, color])
        grayBitmap.append(grayRow)
    return grayBitmap


def makeBlobs(bitmap):
    bitmap = doNegative(bitmap)
    bitmap = createBitmapWithMask(bitmap, MASK_DILATATION)
    bitmap = createBitmapWithMask(bitmap, MASK_DILATATION)
    bitmap = createBitmapWithMask(bitmap, MASK_DILATATION)
    bitmap = createBitmapWithMask(bitmap, MASK_DILATATION)
    bitmap = createBitmapWithMask(bitmap, MASK_DILATATION)
    bitmap = createBitmapWithMask(bitmap, MASK_DILATATION)
    bitmap = createBitmapWithMask(bitmap, MASK_DILATATION)
    bitmap = createBitmapWithMask(bitmap, MASK_DILATATION)
    im.fromarray(np.uint8(bitmap)).show()


def detectVerticalEdge(bitmap):
    return createBitmapWithMask(bitmap, MASK_EDGE_VERTICAL)


def detectLineVertical(verticals, bitmap):
    detectedBitmap = []
    print(len(verticals))
    print(len(verticals[0]))
    for y in range(len(verticals)):
        detectedRow = []
        for x in range(len(verticals[0])):
            avgColor = (bitmap[y, x, 0] + bitmap[y, x, 1] + bitmap[y, x, 2]) / 3
            if avgColor > 100:
                detectedRow.append([255, 255, 255])
            else:
                detectedRow.append([0, 0, 0])
        detectedBitmap.append(detectedRow)
    writeBitmapToFile(detectedBitmap, "test")


def makeImage(fileName):
    bitmap = readBitmapFromFile(fileName)
    bitmap = edgeDetect(bitmap)
    bitmap = thresholding(bitmap, calculateAveragesRGBColor(bitmap))
    bitmap = makeGrayScale(bitmap)
    # bitmap = thresholding(bitmap, [128,128,128])
    # makeBlobs(bitmap)
    #detectLineVertical(detectVerticalEdge(bitmap), makeGrayScale(bitmap))
    writeBitmapToFile(bitmap, fileName)

def findElement(myImage, myElement, myCopy, ax, myColor):
    result = match_template(myImage, myElement)
    tab = peak_local_max(result, 20, threshold_rel=IDENT_PARAM)
    for el in tab:
        y = el[0]
        x = el[1]
        height, width = myElement.shape
        myImage[y:y+height, x:x+width] = 0
        #print (y)
        #print (x)
        rect = plt.Rectangle((x,y), width, height, edgecolor=myColor, fill=False)
        ax.add_patch(rect)
    return myImage, myCopy

def findSth(elements):
    for i in elements:
        mean = round(np.mean(i),3)
        std = round(np.std(i),3)
        var = round(np.var(i),3)
        print(mean, std,var)

def loadElements(myNames):
    elements = []
    for i in myNames:
        elements.append(io.imread("Patterns/" + i +".jpg", as_grey=True))
    return elements

def main():
    myNames = ['chord3', 'chord2','trebleClef', 'bassClef', 'eighthNote', 'quarterNote', 'wholeNote']
    frameColor = ['yellow','coral','b', 'r', 'm', 'c', 'g']

    fileName = "GGC0"
    makeImage(fileName)
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    elements = loadElements(myNames)
    #line = io.imread("Patterns/line.jpg", as_grey=True)
    #findSth(elements)
    myImage = io.imread("Done/JGC0.jpg", as_grey=True)
    myCopy = io.imread("Done/JGC0.jpg", as_grey=True)
    for i in range(len(elements)):
        myImage, myCopy = findElement(myImage, elements[i], myCopy, ax, frameColor[i])
    io.imshow(myCopy)
    plt.show()


if __name__ == '__main__':
    main()
