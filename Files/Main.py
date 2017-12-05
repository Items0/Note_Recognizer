import numpy as np
from PIL import Image as im
import PIL.ImageStat as imageStat  # fajna clasa analizująca  imageStat.Stat(Image)._get[co chcę (mean, stddev ...)]
import matplotlib.pyplot as plt
import skimage
from skimage.filters.edges import convolve
import skimage.morphology as morph
from skimage.morphology import square
from skimage.feature import match_template, peak_local_max
from skimage import io, data, draw, measure
from skimage.draw import circle_perimeter, set_color
import math
from skimage.measure import (moments, moments_central, moments_normalized,
                             moments_hu)
import pandas as pd

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
                              [-1, -1, -1]])
MASK_MEAN = np.array([[1, 1, 1],
                      [1, 2, 1],
                      [1, 1, 1]]) / 20
MASK_DILATATION = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]])

IDENT_PARAM = 0.52  # <-1,1>


def readBitmapFromFile(fileName):
    path = "Photos/" + fileName + ".jpg"
    image = im.open(path)
    image = image.resize((int(image.size[0] / np.sqrt(6)), int(image.size[1] / np.sqrt(6))))
    return np.asarray(image)


def writeBitmapToFile(bitmap, fileName):
    image = im.fromarray(np.uint8(bitmap))
    path = "Done/" + fileName + FILE_SUFIX + ".jpg"
    image.save(path)


def filterImage(copy, original):
    print("Obrabiam obrazek")
    # im.fromarray(np.uint8(original * 255)).show()
    copy = np.abs(convolve(copy, MASK_MEAN))
    copy = skimage.filters.sobel(copy)
    copy = copy > skimage.filters.threshold_li(copy)
    copy = morph.erosion(copy)
    copy = morph.dilation(copy)

    copy, blob, original = toHorizontalLevel(copy, original)
    image2 = np.asarray(im.fromarray(np.uint8(copy)))

    copy = fillEmptySpaceInImage(image2)

    starts, stops = detectStartsAndEndsBlobs(blob)
    copyParts, originalParts = divideImageOnParts(copy, original, starts, stops)

    return copyParts, originalParts


def cutNotesFromImage(image):
    print("Wycinam nutki z obrazka")
    verticalEdge = skimage.filters.sobel_v(image)
    zerosMatrix = np.zeros((len(verticalEdge), len(verticalEdge[0])))
    right = 50
    left = 50
    down = 170
    positions = []
    detectNotes = []
    for y in range(len(verticalEdge)):
        for x in range(len(verticalEdge[0])):
            if zerosMatrix[y, x] == 0 and verticalEdge[y, x] != 0 and y + down < len(verticalEdge) \
                    and x - left >= 0 and x + right < len(verticalEdge[0]) \
                    and isNotMarkArea(zerosMatrix, x, y, left, right, down):
                positions.append((x, y))
                note = np.zeros((down, left + right))
                for j in range(down):
                    for i in range(left):
                        zerosMatrix[y + j, x - i] = 1
                        note[j, left - i] = image[y + j][x - i]
                    for k in range(right):
                        zerosMatrix[y + j, x + k] = 1
                        note[j, left + k] = image[y + j][x + k]
                detectNotes.append(note)
    return detectNotes, positions


def isNotMarkArea(image, x, y, left, right, down):
    for i in range(down):
        for j in range(left):
            if image[y + i, x - j] == 1:
                return False
        for k in range(right):
            if image[y + i, x + j] == 1:
                return False
    return True


def fillEmptySpaceInImage(image):
    print("Wypełniam puste przestrzenie konturów")
    limit = 20
    image.setflags(write=1)
    for y in range(len(image)):
        start = 0
        changes = 0
        lenght = 0
        for x in range(len(image[0])):
            if x + 1 < len(image[0]) and image[y, x] == 1 and image[y, x + 1] == 0:
                changes += 1
                if changes == 1:
                    start = x
                    lenght = 0
            if changes == 1 and lenght != 0 and x + 1 < len(image[0]) and image[y, x] == 0 and image[y, x + 1] == 1:
                changes += 1
                if changes == 2:
                    for i in range(x - start):
                        image[y, x - i] = 1
                    start = 0
                    lenght = 0
                    changes = 0
            if changes != 0:
                lenght += 1
                if lenght > limit:
                    lenght = 0
                    changes = 0
                    start = 0
            lenght += 1
    return image


def toHorizontalLevel(image, original):
    print("Poziomuję obrazek")
    while True:
        blob = makeBlobs(image)

        image2 = np.asarray(im.fromarray(np.uint8(blob)).resize((int(len(blob) / 30), (int(len(blob[0]) / 30)))))
        print((len(image2) * len(image2[0])), " ", (len(image2) * len(image2[0]) / 100))
        zerosMatrix = np.zeros((len(image2), len(image2[0])))
        limit = len(image2) * len(image2[0]) / 100
        detectOneStaff(image2, zerosMatrix, limit)

        xPositions, yPositions = prepareDataToRegression(zerosMatrix)
        aParameter, bParameter = np.polyfit(xPositions, yPositions, 1)
        print(aParameter)
        print(bParameter)
        angle = np.arctan(aParameter) * 188 / np.pi
        print("angle ", angle)

        image = np.asarray(im.fromarray(np.uint8(image)).rotate(angle, expand=True))
        blob = np.asarray(im.fromarray(np.uint8(blob)).rotate(angle, expand=True))
        original = np.asarray(im.fromarray(np.float_(original)).rotate(angle, expand=True))
        # im.fromarray(np.uint8(original * 255)).show()
        if np.abs(angle) < 10:
            break
    return image, blob, original


def makeBlobs(image):
    print("Robię blob'y")
    blob = morph.dilation(image, square(30))
    return blob


def rowContainWhite(row):
    for i in range(len(row)):
        if row[i] == 1:
            return True
    return False


def detectStartsAndEndsBlobs(image):
    print("Szukam początków i końców blob'ów")
    starts = []
    ends = []
    isBlob = False
    counter = 0
    for row in image:
        whiteInRow = rowContainWhite(row)
        if not isBlob and whiteInRow:
            starts.append(counter)
            isBlob = True
        if isBlob and not whiteInRow:
            ends.append(counter)
            isBlob = False
        counter += 1
    return starts, ends


def divideImageOnParts(copy, original, starts, stops):
    print("Dzielę obrazek na części")
    copyParts = []
    copyPart = []
    originalParts = []
    originalPart = []
    rewrite = False
    emptyLine = []
    for i in range(len(copy[0])):
        emptyLine.append(0)
    for i in range(len(copy)):
        if not rewrite and i in starts:
            rewrite = True
            copyPart = []
            originalPart = []
        if rewrite and i in stops:
            rewrite = False
            for i in range(200):
                copyPart.append(emptyLine)
                originalPart.append(emptyLine)
            copyParts.append(copyPart)
            originalParts.append(originalPart)
        if rewrite:
            copyPart.append(copy[i] * 1)
            originalPart.append(original[i])
    return copyParts, originalParts


def detectOneStaff(blob, zerosMatrix, limit):
    print("Szukam bloba z pięciolinią")
    for y in range(len(blob)):
        for x in range(len(blob[0])):
            if (blob[y, x] == 1 and zerosMatrix[y, x] == 0):
                counter = markOneShape(blob, zerosMatrix, x, y, 0)
                if (counter > limit):
                    return


def markOneShape(blob, zerosMatrix, x, y, counter):
    zerosMatrix[y, x] = 1
    if y - 1 >= 0 and blob[y - 1, x] == 1 and zerosMatrix[y - 1, x] == 0:
        counter = markOneShape(blob, zerosMatrix, x, y - 1, counter + 1)
    if y + 1 < len(blob) and blob[y + 1, x] == 1 and zerosMatrix[y + 1, x] == 0:
        counter = markOneShape(blob, zerosMatrix, x, y + 1, counter + 1)
    if x - 1 >= 0 and blob[y, x - 1] == 1 and zerosMatrix[y, x - 1] == 0:
        counter = markOneShape(blob, zerosMatrix, x - 1, y, counter + 1)
    if x + 1 < len(blob[0]) and blob[y, x + 1] == 1 and zerosMatrix[y, x + 1] == 0:
        counter = markOneShape(blob, zerosMatrix, x + 1, y, counter + 1)
    return counter


def prepareDataToRegression(image):
    print("Przygotowuję dane do regresji")
    xPositions = []
    yPositions = []
    for y in range(len(image)):
        for x in range(len(image[0])):
            if image[y, x] == 1:
                xPositions.append(x)
                yPositions.append(y)
    return xPositions, yPositions


def calculateRegressionFromMachineLearning(xPositions, yPositions):
    print("Liczę regresję z machine learning")
    # todo mocno zmniejszyć dane wejściowe
    aParameter = 0.0  # inicjalizuj wagi
    bParameter = 0.0  #
    learning_rate = 0.0001  # stala uczenia
    maxIteration = 10000  # liczba iteracji

    dividor = 1000

    for i in range(maxIteration):
        if i % dividor == 0:
            print("Skończyłem ", str(i / dividor), "%")
        toB = 0.0
        toA = 0.0
        for j in range(len(xPositions)):
            toB += aParameter * xPositions[j] + bParameter - yPositions[j]
            toA += (aParameter * xPositions[j] + bParameter - yPositions[j]) * xPositions[j]
        aParameter = aParameter - learning_rate * (1 / len(xPositions)) * toA
        bParameter = bParameter - learning_rate * (1 / len(xPositions)) * toB
    return aParameter, bParameter


def findElement(myImage, myElement, myCopy, ax, myColor):
    result = match_template(myImage, myElement)
    tab = peak_local_max(result, 20, threshold_rel=IDENT_PARAM)
    for el in tab:
        y = el[0]
        x = el[1]
        height, width = myElement.shape
        myImage[y:y + height, x:x + width] = 0
        # print (y)
        # print (x)
        rect = plt.Rectangle((x, y), width, height, edgecolor=myColor, fill=False)
        ax.add_patch(rect)
    return myImage, myCopy

    # io.imshow(myImage)
    # plt.show()

    # myElement = io.imread("Patterns/full_note.jpg", as_grey=True)
    # for i in range(0, 4):
    #     myImage, myCopy = findElement(myImage, myElement, myCopy)
    # io.imshow(myCopy)
    # plt.show()


def findSth(elements):
    for i in elements:
        mean = round(np.mean(i), 3)
        std = round(np.std(i), 3)
        var = round(np.var(i), 3)
        print(mean, std, var)


def loadElements(myNames):
    elements = []
    for i in myNames:
        elements.append(io.imread("Patterns/" + i + ".jpg", as_grey=True))
    return elements


def readPatternsFromFile(filenames):
    patterns = []
    for name in filenames:
        path = "Notes/" + name
        patterns.append(io.imread(path, as_grey=True))
    return patterns


def getMomentsHu(image):
    m = moments(image)
    cr = m[0, 1] / m[0, 0]
    cc = m[1, 0] / m[0, 0]
    mu = moments_central(image, cr, cc)
    mn = moments_normalized(mu)
    hu = moments_hu(mn)
    l = [norm(f) for f in hu]
    return l


table = []
norm = lambda x: -np.sign(x) * np.log10(np.abs(x))


def preparePatternsMomentHu():
    paternImageNames = ['15chord1.jpg', '25chord1.jpg', 'a2.jpg', 'a4.jpg', 'b2.jpg', 'b4.jpg', 'Bass.jpg', 'C2.jpg',
                        'C4.jpg', 'D1.jpg', 'D2.jpg', 'D4.jpg', 'D8.jpg', 'E1.jpg', 'e1.jpg', 'E8.jpg', 'F1.jpg',
                        'f8.jpg', 'g2.jpg', 'G4.jpg', 'G8.jpg', 'Violin.jpg']
    paternImages = readPatternsFromFile(paternImageNames)
    patternsMomentHu = []  # dwuwymiarowa tablica n*7 elementów
    for image in paternImages:
        patternsMomentHu.append(getMomentsHu(image))
    return patternsMomentHu





def main():
    myNames = ['chord3', 'chord2', 'trebleClef', 'bassClef', 'eighthNote', 'quarterNote', 'wholeNote']
    frameColor = ['yellow', 'coral', 'b', 'r', 'm', 'c', 'g']

    patternHu = preparePatternsMomentHu()
    copyImage = io.imread("Photos/JGC0.jpg", as_grey=True)
    originalImage = io.imread("Photos/JGC0.jpg", as_grey=True)

    copyParts, originalParts = filterImage(copyImage, originalImage)


    for i in range(len(copyParts)):
        detectNotes, positions = cutNotesFromImage(copyParts[i])
        isNotes = np.zeros(len(detectNotes))
        print(isNotes)



    for i in range(len(copyParts)):
        img = im.fromarray(np.uint8(copyParts[i]) * 255)
        img.save(str(i) + ".jpg")
        ori = np.float_(originalParts[i]) * 255
        ori = np.uint8(ori)
        img = im.fromarray(ori)
        img.save(str(i) + "o.jpg")



    # fig = plt.figure(figsize=(15, 10))4
    # ax = fig.add_subplot(111)
    # elements = loadElements(myNames)
    # # line = io.imread("Patterns/line.jpg", as_grey=True)
    # # findSth(elements)
    # myImage = filterImage(myImage
    #
    # for i in range(len(elements)):
    #     myImage, myCopy = findElement(myImage, elements[i], myCopy, ax, frameColor[i])
    # io.imshow(myCopy)
    # plt.show()


if __name__ == '__main__':
    main()
