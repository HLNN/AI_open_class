import cv2


class Video:
    def __init__(self):
        self.in_cap = None
        self.fps = 0
        self.length = 0
        self.fourcc = 0
        self.frameSize = None

    def read(self):
        self.in_cap = cv2.VideoCapture("test.mp4")
        self.fps = int(self.in_cap.get(cv2.CAP_PROP_FPS))
        self.length = int(self.in_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fourcc = int(self.in_cap.get(cv2.CAP_PROP_FOURCC))
        self.frameSize = (int(self.in_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.in_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def write(self):
        out_cap = cv2.VideoWriter('out.mp4', self.length, self.fourcc, self.fps, self.frameSize)
        out_cap.open('out.mp4', self.fourcc, self.fps, self.frameSize)
        for _ in range(self.length):
            ret, frame = self.in_cap.read()
            out_cap.write(cv2.Canny(frame, 100, 300))

    def main(self):
        self.read()
        self.write()


if __name__ == '__main__':
    video = Video()
    video.main()
