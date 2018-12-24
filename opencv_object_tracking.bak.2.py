# USAGE
# python opencv_object_tracking.py
# python opencv_object_tracking.py --video dashcam_boston.mp4 --tracker csrt --point x y w h

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2

# construct the argument parser and parse the arguments
def argParse():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str, help="path to input video file")
    ap.add_argument("-t", "--tracker", type=str, default="kcf", help="OpenCV object tracker type")
    ap.add_argument("-p", "--point", nargs='+', type=int, help="point")
    args = vars(ap.parse_args())

    return args

class Tracker:

    def __init__(self, args):
        self.args = args
        self.chkInit = None
        self.initBB = None


    def chkVideo(self):
        if not self.args.get("video", False):
            print("[INFO] starting video stream...")
            vs = VideoStream(src=0).start()
            time.sleep(1.0)
        else:
            vs = cv2.VideoCapture(self.args["video"])
        return vs

    def chkPoint(self):
        if "point" in self.args:
            self.initBB = self.args["point"]

    def selectTracker(self):
        OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.TrackerTLD_create,
            "medianflow": cv2.TrackerMedianFlow_create,
            "mosse": cv2.TrackerMOSSE_create
        }

        tracker = OPENCV_OBJECT_TRACKERS[self.args["tracker"]]()

        return tracker

    def run(self):

        fps = None
        min = 0.0
        max = 0.0
        avg = 0.0
        idx = 0
        sum = 0

        self.chkPoint()
        vs = self.chkVideo()
        tracker = self.selectTracker()

        while True:
            frame = vs.read()
            frame = frame[1] if self.args.get("video", False) else frame
            idx += 1

            if frame is None:
                break

            frame = imutils.resize(frame, width=500)
            (H, W) = frame.shape[:2]

            if self.initBB is not None:

                if self.chkInit is None:
                    self.initBB=tuple(self.initBB)
                    tracker.init(frame, self.initBB)
                    self.chkInit = True
                    fps = FPS().start()
                else:
                    (success, box) = tracker.update(frame)

                    if success:
                        (x, y, w, h) = [int(v) for v in box]
                        cv2.rectangle(frame, (x, y), (x + w, y + h),
                            (0, 255, 0), 2)

                    fps.update()
                    fps.stop()

                    temp = fps.fps()
                    if temp < min: min = temp
                    if temp > max: max = temp
                    sum = sum + temp
                    avg = sum/idx


                    info = [
                        ("Tracker", self.args["tracker"]),
                        ("FPS", "{:.2f}".format(fps.fps())),
                        ("MIN", "{:.2f}".format(min)),
                        ("MAX", "{:.2f}".format(max)),
                        ("AVG", "{:.2f}".format(avg)),
                    ]

                    # loop over the info tuples and draw them on our frame
                    for (i, (k, v)) in enumerate(info):
                        text = "{}: {}".format(k, v)
                        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the 's' key is selected, we are going to "select" a bounding
            # box to track
            if key == ord("s"):
                # select the bounding box of the object we want to track (make
                # sure you press ENTER or SPACE after selecting the ROI)
                self.initBB = cv2.selectROI("Frame", frame, fromCenter=False,
                    showCrosshair=True)

                # start OpenCV object tracker using the supplied bounding box
                # coordinates, then start the FPS throughput estimator as well
                tracker.init(frame, self.initBB)
                fps = FPS().start()
                self.chkInit = True

            # if the `q` key was pressed, break from the loop
            elif key == ord("q"):
                break

        # if we are using a webcam, release the pointer
        if not self.args.get("video", False):
            vs.stop()

        # otherwise, release the file pointer
        else:
            vs.release()

        # close all windows
        cv2.destroyAllWindows()


if __name__ == "__main__":
    args = argParse()
    tracker = Tracker(args)
    tracker.run()