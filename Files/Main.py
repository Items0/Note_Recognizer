import numpy as np
from PIL import Image as im

MULTIPLE_STD_PARAM = 2.0
FILE_SUFIX = "std2.0"

def readBitmapFromFile(fileName):
    path = "Photos/" + fileName + ".jpg"
    image = im.open(path)
    image = image.resize((int(image.size[0]/np.sqrt(6)), int(image.size[1]/np.sqrt(6))))
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
    print(rThresh," ", gThresh," ", bThresh)
    print(std)
    for row in bitmap:
        threshRow = []
        for cell in row:
            r = 255 if rThresh > cell[0] else 0
            g = 255 if gThresh > cell[1] else 0
            b = 255 if bThresh > cell[2] else 0
            threshRow.append([r, g, b])
        threshBitmap.append(threshRow)
    return threshBitmap


def makeImage(fileName):
    bitmap = readBitmapFromFile(fileName)
    bitmap = thresholding(bitmap, calculateAveragesRGBColor(bitmap), np.std(bitmap))
    writeBitmapToFile(bitmap, fileName)


def main():
    fileName = "JGC0"
    makeImage(fileName)


if __name__ == '__main__':
    main()
