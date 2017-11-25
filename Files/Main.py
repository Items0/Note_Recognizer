import numpy as np
import matplotlib.pyplot as plt
import skimage
import math
from PIL import Image as im
from skimage.feature import match_template
from skimage import io, data, draw
from skimage.draw import circle_perimeter

MULTIPLE_STD_PARAM = 2.0
FILE_SUFIX = "std2.0"

def readBitmapFromFile(fileName):
    path = "Photos/" + fileName + ".jpg"
    return np.asarray(im.open(path))


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

def findElement(myImage, myElement, myCopy):
    result = match_template(myImage, myElement)
    y, x = np.unravel_index(np.argmax(result), result.shape)
    height, width = myElement.shape
    myImage[y:y+height, x:x+width] = 0
    print (y)
    print (x)
    rr, cc = circle_perimeter(math.ceil(y+height/2),math.ceil(x+width/2), min(height, width))
    myCopy[rr, cc] = 1
    return myImage, myCopy



def makeImage(fileName):
    bitmap = readBitmapFromFile(fileName)
    bitmap = thresholding(bitmap, calculateAveragesRGBColor(bitmap), np.std(bitmap))
    writeBitmapToFile(bitmap, fileName)


def main():
    '''fileName = ["GGC0", "GGC3", "GGO3", "GPN3", "GPN6", "GPO0", "JBO0", "JBO3", "JGC0", "JGC3", "JPC6",
    "JPN3", "JPO3", "NBO0", "NBO6", "NGC3", "NPN0", "PBO0", "PGC0", "PGO3", "PPC3", "PPO3"]

    for i in fileName:
        makeImage(i)'''

    #fileName = "JGC0"
    #makeImage(fileName)
    fig = plt.figure(figsize=(15, 10))
    myImage = io.imread("Photos/JPN3.jpg", as_grey=True)
    myCopy = io.imread("Photos/JPN3.jpg", as_grey=True)
    myElement = io.imread("Patterns/myNute.jpg", as_grey=True)
    for i in range(0,4):
        myImage, myCopy = findElement(myImage, myElement, myCopy)
    io.imshow(myCopy)
    plt.show()

if __name__ == '__main__':
    main()
