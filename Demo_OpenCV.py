import cv2
import numpy as np
import imutils


OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}

tracker = OPENCV_OBJECT_TRACKERS["medianflow"]() 
initBB = None

# vs = cv2.VideoCapture(r"C:\Users\Minh Tien\Desktop\demo_yolov4.mp4")
vs = cv2.VideoCapture(0)

tracking_obj = []
id=0

while True:
    _, frame = vs.read()
    if frame is None:
        break
    id+=1
    #frame = imutils.resize(frame, width=500)
    frame = cv2.flip(frame, 1)

    for i in range(1, len(tracking_obj)):
        cv2.line(frame, (tracking_obj[i-1][0], tracking_obj[i-1][1]), (tracking_obj[i][0], tracking_obj[i][1]), (0, 255, 0), 3)

    H, W, _ = frame.shape
    if initBB is not None:
        (success, box) = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.circle(frame, (x + w//2,y + h//2), 1, (255, 0, 0), -1)
            tracking_obj.append([x + w//2, y + h//2])

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        initBB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        tracker.init(frame, initBB)
    elif key == ord("q"):
        break
    
    #vs.release()
    
cv2.destroyAllWindows()

