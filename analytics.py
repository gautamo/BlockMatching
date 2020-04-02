from main import *
from frameCollect import *
import matplotlib
import matplotlib.pyplot as plt
import pickle
import os
import shutil

def plot(x, y, name, ylabel="residualMetric"): # plot metric on graph
    assert len(x) == len(y)

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x, y, label=f'$x = Frame, $y = {ylabel}')
    plt.title(name)
    ax.legend()
    fig.savefig(f'PLOT/{name}_plot.png')

def graphSession(filepath, pframeSet=300, countMax = 1000, skip=0, process=["residualMetric"], meanMetricSize=10): # create graph of mean compounded residualMetric
    """
    Graph the metric chosen in process for the video in filepath
    filepath: path to video file
    pframeSet: number of predicted frames every i-frame
    countMax: number of data points on the graphs
    skip: how many frames to skip per frame
    process: 3 Parts, process 1 = residualMetric, process 2 = meanMetric
        Process 1: create graph of residualMetric per I-Frame interval
        Process 2: create graph of mean compounded residualMetric for x Frames specified by meanMetricSize
    """
    assert "residualMetric" in process or "meanMetric" in process

    video_name, video, frame_width, frame_height, videofps, videoframecount = read_video(filepath)
    running = True
    count = 0

    residualTally = []
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

    if "meanMetric" in process:
        meanMetricRange = np.arange(len(meanTally))
        plot(meanMetricRange, meanTally, f"mean_{video_name}_{pframeSet}predF{skip}skipF{meanMetricSize}meanSize", ylabel=f"{meanMetricSize} Frame Mean Compounded residualMetric")

    return residualTally, meanTally

if __name__ == "__main__":
    timer = time.time()

    """
    videopathA = "VIDEOS/UAV123_person1_resized360.mp4"
    PROCESS = ["meanMetric"]
    graphSession(videopathA, pframeSet=5, process=PROCESS, meanMetricSize=5)
    graphSession(videopathA, pframeSet=10, process=PROCESS, meanMetricSize=10)

    """

    totTime = time.time() - timer
    print(f"Done, executed in {totTime:.2f} s")
