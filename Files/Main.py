import numpy as np
from PIL import Image as im
import PIL.ImageStat as imageStat  # fajna clasa analizująca  imageStat.Stat(Image)._get[co chcę (mean, stddev ...)]
import matplotlib.pyplot as plt
from skimage.filters.edges import convolve

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
    bitmap = createBitmapWithMask(bitmap, MASK_EDGE_VERTICAL)
    im.fromarray(np.uint8(bitmap)).show()


def makeImage(fileName):
    bitmap = readBitmapFromFile(fileName)
    bitmap = edgeDetect(bitmap)
    bitmap = thresholding(bitmap, calculateAveragesRGBColor(bitmap))
    bitmap = makeGrayScale(bitmap)
   # bitmap = thresholding(bitmap, [128,128,128])
    #makeBlobs(bitmap)
    detectVerticalEdge(bitmap)
    writeBitmapToFile(bitmap, fileName)


def main():
    fileName = "JGC0"
    makeImage(fileName)


if __name__ == '__main__':
    main()
