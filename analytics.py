from main import *
from frameCollect import *
import matplotlib
import matplotlib.pyplot as plt
import pickle
import os
import shutil
from zipfile import ZipFile

def zipdir(path, ziph): #creates zip file
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

def plot(x, y, name, ylabel="residualMetric"): # plot metric on graph
    assert len(x) == len(y)

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x, y, label=f'$x = Frame, $y = {ylabel}')
    plt.title(name)
    ax.legend()
    fig.savefig(f'PLOT2/{name}_plot.png')

def graphSession(filepath, pframeSet=300, countMax = 1000, skip=0, process=["residualMetric"], meanMetricSize=10): # create graph of mean compounded residualMetric
    """
    Graph the metric chosen in process for the video in filepath
    filepath: path to video file
    pframeSet: number of predicted frames every i-frame
    countMax: number of data points on the graphs
    skip: how many frames to skip per frame
    process: 3 Parts, process 1 = residualMetric, process 2 = sizeMetric, process 3 = meanMetric
        Process 1: create graph of residualMetric per I-Frame interval

        Process 2: Graph the data size of zip file containing the I-Frame and Residual Frame to
        analyze data savings across a video.

        Process 3: create graph of mean compounded residualMetric for x Frames specified by meanMetricSize
    """
    assert "residualMetric" in process or "sizeMetric" in process or "meanMetric" in process

    video_name, video, frame_width, frame_height, videofps, videoframecount = read_video(filepath)
    print(f"FPS: {videofps}")
    running = True
    count = 0

    residualTally = []
    sizeTally = []
    meanTally = []
    meanBuffer = []

    running, frame = read_frame(video, skipFrame=skip)
    while running and count < countMax:
        print(f"Count {count} of {countMax}")

        if count % pframeSet == 0:
            print("REESTABLISHING I-FRAME")
            iframe = frame

        residualMetric, residualFrame = main(iframe, frame)

        if "residualMetric" in process:
            residualTally.append(residualMetric)

        if "sizeMetric" in process:

            if os.path.exists('sizeBuffer'):
                shutil.rmtree("sizeBuffer")
            os.mkdir('sizeBuffer')

            cv2.imwrite(os.path.join("sizeBuffer", 'iframe.png'), iframe)
            cv2.imwrite(os.path.join("sizeBuffer", 'residual.png'), residualFrame)

            if os.path.exists('sizeBuffer.zip'):
                os.remove("sizeBuffer.zip")

            zipObj = ZipFile('sizeBuffer.zip', 'w') # create a ZipFile object
            zipObj.write(os.path.join("sizeBuffer", 'iframe.png')) # Add multiple files to the zip
            zipObj.write(os.path.join("sizeBuffer", 'residual.png'))
            zipObj.close() # close the Zip File

            sizeTally.append(os.path.getsize('sizeBuffer.zip'))

        if "meanMetric" in process:
            meanBuffer.append(residualMetric)
            if len(meanBuffer) == meanMetricSize:
                meanTally.append(sum(meanBuffer)/meanMetricSize)
                meanBuffer.pop(0)


        running, frame = read_frame(video)
        count+=1

    print(f"Residual Tally: {residualTally}")

    metricRange = np.arange(count)

    if "residualMetric" in process:
        plot(metricRange, residualTally, f"residual_{video_name}_{pframeSet}predF{skip}skipF", ylabel="residualMetric")

    if "sizeMetric" in process:
        plot(metricRange, sizeTally, f"size_{video_name}_{pframeSet}predF{skip}skipF", ylabel=" zipped [residualFrame, iFrame] Size")

    if "meanMetric" in process:
        meanMetricRange = np.arange(len(meanTally))
        plot(meanMetricRange, meanTally, f"mean_{video_name}_{pframeSet}predF{skip}skipF{meanMetricSize}meanSize", ylabel=f"{meanMetricSize} Frame Mean Compounded residualMetric")

    return residualTally, sizeTally, meanTally

### TESTS ####

def test1(): #analyze 4 examples
    imagePath = "testImages"
    outPath = "OUTPUT"

    setsToTest = [("personFrame1.png", "personFrame2.png", "personFrameOutput"),
                  ("personFrameLong1.png", "personFrameLong2.png", "personFrameLongOutput"),
                  ("carFrame1.png", "carFrame2.png", "carFrameOutput"),
                  ("carFrameLong1.png", "carFrameLong2.png", "carFrameLongOutput")
                  ]

    for pairs in setsToTest:
        anchorPath = f"{imagePath}/{pairs[0]}"
        targetPath = f"{imagePath}/{pairs[1]}"

        print("Pair", pairs)
        timer = time.time()

        main(anchorPath, targetPath, f"{outPath}/{pairs[2]}", saveOutput=True)
        totTime = time.time() - timer
        print(f"Done, executed in {totTime:.2f} s\n")

def test2(filepath, pframeSet=300, countMax = 1000, skip=0, process=["residualMetric"]): # create graph of residualMetric per I-Frame interval
    return graphSession(filepath, pframeSet, countMax, skip, process)

def test3(filepath, pframeSet=300, countMax = 1000, skip=0, process=["pickleMetric"]): # create graph of residualMetric + I-frame size
    return graphSession(filepath, pframeSet, countMax, skip, process)

def test4(filepath, pframeSet=300, countMax = 1000, skip=0, process=["meanMetric"]): # create graph of residualMetric + I-frame size
    return graphSession(filepath, pframeSet, countMax, skip, process)

### TESTS ####

if __name__ == "__main__":
    timer = time.time()

    """
    videopathA = "/Users/gbanuru/PycharmProjects/BlockMatching/VIDEOS/UAV123_person1_resized360.mp4"
    videopathB = "/Users/gbanuru/PycharmProjects/BlockMatching/VIDEOS/UAV123_car2_resized360.mp4"
    videopathC = "/Users/gbanuru/PycharmProjects/BlockMatching/VIDEOS/CityWalk_resized360.mp4"
    videopathD = "/Users/gbanuru/PycharmProjects/BlockMatching/VIDEOS/HouseTour_resized360.mp4"

    PROCESS = ["meanMetric"]

    graphSession(videopathA, pframeSet=5, process=PROCESS, meanMetricSize=5)
    graphSession(videopathB, pframeSet=5, process=PROCESS, meanMetricSize=5)
    graphSession(videopathC, pframeSet=5, process=PROCESS, meanMetricSize=5)
    graphSession(videopathD, pframeSet=5, process=PROCESS, meanMetricSize=5)

    graphSession(videopathA, pframeSet=10, process=PROCESS, meanMetricSize=10)
    graphSession(videopathB, pframeSet=10, process=PROCESS, meanMetricSize=10)
    graphSession(videopathC, pframeSet=10, process=PROCESS, meanMetricSize=10)
    graphSession(videopathD, pframeSet=10, process=PROCESS, meanMetricSize=10)
    """

    totTime = time.time() - timer
    print(f"Done, executed in {totTime:.2f} s")