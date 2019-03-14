import cv2
import numpy
import sys


class Merge:
    def __init__(self):
        if len(sys.argv) == 1:
            self.inPut = cv2.VideoCapture(input("Input file name:"))
            self.outPut = cv2.VideoCapture(input("Output file name:"))
        else:
            self.inPut = cv2.VideoCapture(sys.argv[1])
            self.outPut = cv2.VideoCapture(sys.argv[2])
        self.fps = int(self.inPut.get(cv2.CAP_PROP_FPS))
        self.length = int(self.inPut.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fourcc = int(self.inPut.get(cv2.CAP_PROP_FOURCC))
        self.frameSize = (int(self.inPut.get(cv2.CAP_PROP_FRAME_WIDTH)) * 2, int(self.inPut.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def main(self):
        out_cap = cv2.VideoWriter('merge.mp4', self.length, self.fourcc, self.fps, self.frameSize)
        out_cap.open('merge.mp4', self.fourcc, self.fps, self.frameSize)
        for _ in range(self.length):
            ret, frameIn = self.inPut.read()
            ret, frameOut = self.outPut.read()
            frame = numpy.hstack((frameIn, frameOut))
            out_cap.write(frame)


if __name__ == '__main__':
    merge = Merge()
    merge.main()
