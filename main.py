# Code to Perform Block Matching

import numpy as np
import cv2
import random
import time
import os

debug = True

def YCrCb2BGR(image):
    """
    Converts numpy image into from YCrCb to BGR color space
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

def BGR2YCrCb(image):
    """
    Converts numpy image into from BGR to YCrCb color space
    """
    return cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)

def segmentImage(anchor, blockSize=16):
    """
    Determines how many macroblocks an image is composed of
    :param anchor: I-Frame
    :param blockSize: Size of macroblocks in pixels
    :return: number of rows and columns of macroblocks within
    """
    h, w = anchor.shape
    hSegments = int(h / blockSize)
    wSegments = int(w / blockSize)
    totBlocks = int(hSegments * wSegments)

    #if debug:
    #    print(f"Height: {h}, Width: {w}")
    #    print(f"Segments: Height: {hSegments}, Width: {wSegments}")
    #    print(f"Total Blocks: {totBlocks}")

    return hSegments, wSegments

def getCenter(x, y, blockSize):
    """
    Determines center of a block with x, y as top left corner coordinates and blockSize as blockSize
    :return: x, y coordinates of center of a block
    """
    return (int(x + blockSize/2), int(y + blockSize/2))

def getAnchorSearchArea(x, y, anchor, blockSize, searchArea):
    """
    Returns image of anchor search area
    :param x, y: top left coordinate of macroblock in Current Frame
    :param anchor: I-Frame
    :param blockSize: size of block in pixels
    :param searchArea: size of search area in pixels
    :return: Image of anchor search area
    """
    h, w = anchor.shape
    cx, cy = getCenter(x, y, blockSize)

    sx = max(0, cx-int(blockSize/2)-searchArea) # ensure search area is in bounds
    sy = max(0, cy-int(blockSize/2)-searchArea) # and get top left corner of search area

    # slice anchor frame within bounds to produce anchor search area
    anchorSearch = anchor[sy:min(sy+searchArea*2+blockSize, h), sx:min(sx+searchArea*2+blockSize, w)]

    return anchorSearch

def getBlockZone(p, aSearch, tBlock, blockSize):
    """
    Retrieves the block searched in the anchor search area to be compared with the macroblock tBlock in the current frame
    :param p: x,y coordinates of macroblock center from current frame
    :param aSearch: anchor search area image
    :param tBlock: macroblock from current frame
    :param blockSize: size of macroblock in pixels
    :return: macroblock from anchor
    """
    px, py = p # coordinates of macroblock center
    px, py = px-int(blockSize/2), py-int(blockSize/2) # get top left corner of macroblock
    px, py = max(0,px), max(0,py) # ensure macroblock is within bounds

    aBlock = aSearch[py:py+blockSize, px:px+blockSize] # retrive macroblock from anchor search area


    try:
        assert aBlock.shape == tBlock.shape # must be same shape

    except Exception as e:
        print(e)
        print(f"ERROR - ABLOCK SHAPE: {aBlock.shape} != TBLOCK SHAPE: {tBlock.shape}")

    return aBlock

def getMAD(tBlock, aBlock):
    """
    Returns Mean Absolute Difference between current frame macroblock (tBlock) and anchor frame macroblock (aBlock)
    """
    return np.sum(np.abs(np.subtract(tBlock, aBlock)))/(tBlock.shape[0]*tBlock.shape[1])

def getBestMatch(tBlock, aSearch, blockSize): #3 Step Search
    """
    Implemented 3 Step Search. Read about it here: https://en.wikipedia.org/wiki/Block-matching_algorithm#Three_Step_Search
    :param tBlock: macroblock from current frame
    :param aSearch: anchor search area
    :param blockSize: size of macroblock in pixels
    :return: macroblock from anchor search area with least MAD
    """
    step = 4
    ah, aw = aSearch.shape
    acy, acx = int(ah/2), int(aw/2) # get center of anchor search area

    minMAD = float("+inf")
    minP = None

    while step >= 1:
        p1 = (acx, acy)
        p2 = (acx+step, acy)
        p3 = (acx, acy+step)
        p4 = (acx+step, acy+step)
        p5 = (acx-step, acy)
        p6 = (acx, acy-step)
        p7 = (acx-step, acy-step)
        p8 = (acx+step, acy-step)
        p9 = (acx-step, acy+step)
        pointList = [p1,p2,p3,p4,p5,p6,p7,p8,p9] # retrieve 9 search points

        for p in range(len(pointList)):
            aBlock = getBlockZone(pointList[p], aSearch, tBlock, blockSize) # get anchor macroblock
            MAD = getMAD(tBlock, aBlock) # determine MAD
            if MAD < minMAD: # store point with minimum mAD
                minMAD = MAD
                minP = pointList[p]

        step = int(step/2)

    px, py = minP # center of anchor block with minimum MAD
    px, py = px - int(blockSize / 2), py - int(blockSize / 2) # get top left corner of minP
    px, py = max(0, px), max(0, py) # ensure minP is within bounds
    matchBlock = aSearch[py:py + blockSize, px:px + blockSize] # retrieve best macroblock from anchor search area

    return matchBlock



def blockSearchBody(anchor, target, blockSize, searchArea=7):
    """
    Facilitates the creation of a predicted frame based on the anchor and target frame
    :param anchor: I-Frame
    :param target: Current Frame to create a P-Frame from
    :param blockSize: size of macroBlock in pixels
    :param searchArea: size of searchArea extended from blockSize
    :return: predicted frame
    """
    h, w = anchor.shape
    hSegments, wSegments = segmentImage(anchor, blockSize)


    predicted = np.ones((h, w))*255
    bcount = 0
    for y in range(0, int(hSegments*blockSize), blockSize):
        for x in range(0, int(wSegments*blockSize), blockSize):
            bcount+=1
            targetBlock = target[y:y+blockSize, x:x+blockSize] #get current macroblock

            anchorSearchArea = getAnchorSearchArea(x, y, anchor, blockSize, searchArea) #get anchor search area

            #print("AnchorSearchArea: ", anchorSearchArea.shape)

            anchorBlock = getBestMatch(targetBlock, anchorSearchArea, blockSize) #get best anchor macroblock
            predicted[y:y+blockSize, x:x+blockSize] = anchorBlock #add anchor block to predicted frame

            #cv2.imwrite("OUTPUT/predictedtestFrame.png", predicted)
            #print(f"ITERATION {bcount}")

    #cv2.imwrite("OUTPUT/predictedtestFrame.png", predicted)

    #time.sleep(10)

    assert bcount == int(hSegments*wSegments) #check all macroblocks are accounted for

    return predicted

def getResidual(target, predicted):
    """Create residual frame from target frame - predicted frame"""
    return np.subtract(target, predicted)

def getReconstructTarget(residual, predicted):
    """Reconstruct target frame from residual frame plus predicted frame"""
    return np.add(residual, predicted)

def showImages(*kwargs): #shows images
    for k in range(len(kwargs)):
        cv2.imshow(f"Image: {k}", k)
        cv2.waitKey(-1)

def getResidualMetric(residualFrame):
    """Calculate residual metric from average of sum of absolute residual values in residual frame"""
    return np.sum(np.abs(residualFrame))/(residualFrame.shape[0]*residualFrame.shape[1])

def preprocess(anchor, target, blockSize):

    if isinstance(anchor, str) and isinstance(target, str):
        anchorFrame = BGR2YCrCb(cv2.imread(anchor))[:, :, 0] # get luma component
        targetFrame = BGR2YCrCb(cv2.imread(target))[:, :, 0] # get luma component

    elif isinstance(anchor, np.ndarray) and isinstance(target, np.ndarray):
        anchorFrame = BGR2YCrCb(anchor)[:, :, 0] # get luma component
        targetFrame = BGR2YCrCb(target)[:, :, 0] # get luma component

    else:
        raise ValueError

    #resize frame to fit segmentation
    hSegments, wSegments = segmentImage(anchorFrame, blockSize)
    anchorFrame = cv2.resize(anchorFrame, (int(wSegments*blockSize), int(hSegments*blockSize)))
    targetFrame = cv2.resize(targetFrame, (int(wSegments*blockSize), int(hSegments*blockSize)))

    #if debug:
        #print(f"A SIZE: {anchorFrame.shape}")
        #print(f"T SIZE: {targetFrame.shape}")


    return (anchorFrame, targetFrame)

def main(anchorFrame, targetFrame, outfile="OUTPUT", saveOutput=False, blockSize = 16):
    """
    Calculate residual frame and metric along with other artifacts
    :param anchor: file path of I-Frame or I-Frame
    :param target: file path of Current Frame or Current Frame
    :return: residual metric
    """
    anchorFrame, targetFrame = preprocess(anchorFrame, targetFrame, blockSize) #processes frame or filepath to frame

    predictedFrame = blockSearchBody(anchorFrame, targetFrame, blockSize)
    residualFrame = getResidual(targetFrame, predictedFrame)
    naiveResidualFrame = getResidual(anchorFrame, targetFrame)
    reconstructTargetFrame = getReconstructTarget(residualFrame, predictedFrame)
    #showImages(targetFrame, predictedFrame, residualFrame)

    residualMetric = getResidualMetric(residualFrame)
    naiveResidualMetric = getResidualMetric(naiveResidualFrame)

    rmText = f"Residual Metric: {residualMetric:.2f}"
    nrmText = f"Naive Residual Metric: {naiveResidualMetric:.2f}"

    isdir = os.path.isdir(outfile)
    if not isdir:
        os.mkdir(outfile)

    if saveOutput:
        cv2.imwrite(f"{outfile}/targetFrame.png", targetFrame)
        cv2.imwrite(f"{outfile}/predictedFrame.png", predictedFrame)
        cv2.imwrite(f"{outfile}/residualFrame.png", residualFrame)
        cv2.imwrite(f"{outfile}/reconstructTargetFrame.png", reconstructTargetFrame)
        cv2.imwrite(f"{outfile}/naiveResidualFrame.png", naiveResidualFrame)
        resultsFile = open(f"{outfile}/results.txt", "w"); resultsFile.write(f"{rmText}\n{nrmText}\n"); resultsFile.close()

    print(rmText)
    print(nrmText)

    return residualMetric, residualFrame

if __name__ == "__main__":
    pass
    """
    anchorPath = "testImages/personFrame1.png"
    targetPath = "testImages/personFrame2.png"
    main(anchorPath, targetPath)
    """
