import numpy as np
import cv2
import cv2.aruco as aruco
import glob,time

'''
    drawMarker(...)
        drawMarker(dictionary, id, sidePixels[, img[, borderBits]]) -> img
'''

def downsample(fname,rate):
    # Opens the Video file
    cap = cv2.VideoCapture(fname)
    i = 0
    ret, frame = cap.read()
    vid_writer = cv2.VideoWriter('downsample/downsampled.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15,(frame.shape[1], frame.shape[0]))

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break

        # only write out when there are multipliers of 10
        if(i%rate==0):
            vid_writer.write(frame)
            print("writing frame " + str(i))

        i += 1

    vid_writer.release()
    cap.release()
    cv2.destroyAllWindows()


def calib_camera():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # find all files with extension .jpg
    images = glob.glob('camera_calib/*.jpg')
    total_count = len(images)
    i=0

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        i+=1

        print("Calibrating based on image : " + str(i) + "/" + str(total_count))
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    print(imgpoints)

    # get the calibrated parameters
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # save the calibration parameters
    np.savez("calib_parameters",mtx = mtx,dist=dist,rvecs = rvecs,tvecs = tvecs)

    cv2.destroyAllWindows()


def generate_tag():
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    print(aruco_dict)
    # second parameter is id number
    # last parameter is total image size
    img = aruco.drawMarker(aruco_dict, 4, 700)
    cv2.imwrite("test_marker4.jpg", img)

    cv2.imshow('frame', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def hand_tracking(frame,protoFile,weightsFile,centroid,width):
    # protoFile = "HandPose/hand/pose_deploy.prototxt"
    # weightsFile = "HandPose/hand/pose_iter_102000.caffemodel"
    nPoints = 22
    POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12],
                  [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    threshold = 0.2

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    aspect_ratio = frameWidth / frameHeight

    inHeight = 368
    inWidth = int(((aspect_ratio * inHeight) * 8) // 8)

    # vid_writer = cv2.VideoWriter('HandPose/output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15,
    #                              (frame.shape[1], frame.shape[0]))

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    time_elapse = 0

    t = time.time()
    frameCopy = np.copy(frame)

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                    (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)

    output = net.forward()

    print("forward = {}".format(time.time() - t))

    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (frameWidth, frameHeight))

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        if prob > threshold:
            cv2.circle(frameCopy, (int(point[0]), int(point[1])), 6, (0, 255, 255), thickness=-1,
                       lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, .8,
                        (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(point[0]), int(point[1])))
        else:
            points.append(None)

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.circle(frame, points[partA], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    # print("Time Taken for frame = {}".format(time.time() - t))

    time_elapse += time.time() - t

    # cv2.imshow('Output-Skeleton', frame)

    # cv2.imwrite("video_output/{:03d}.jpg".format(k), frame)
    # print("total = {}".format(time.time() - t))

    print("total time elapsed = {}".format(time_elapse))

    interact_status = "no interaction detected"

    pt_count = 0


    for pt in points:
        if pt is not None:
            x = pt[0]
            y = pt[1]
            if centroid[0] - width <= x <= centroid[0] + width and centroid[1] - width <= y <= centroid[
                1] + width:
                pt_count += 1

    if pt_count >= 10:
        interact_status = "turning on the stove"
        print("Detected interaction")

    frame = cv2.putText(frame, interact_status, (centroid[0], centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                       (0, 0, 255))

    return frame,points


def detect_tag_and_hand():

    # loaded saved calibration parameters
    with np.load('calib_parameters.npz') as X:
        mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

    # load parameters for the dnn
    protoFile = "HandPose/hand/pose_deploy.prototxt"
    weightsFile = "HandPose/hand/pose_iter_102000.caffemodel"

    cap = cv2.VideoCapture(0)

    # cap = cv2.VideoCapture("turning_on_the_stove.MOV")


    ret, frame = cap.read()

    # initialize the video writer
    vid_writer = cv2.VideoWriter('interaction.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15,(frame.shape[1], frame.shape[0]))



    while (True):
        # Capture frame-by-frame
        hasFrame, frame = cap.read()
        # print(frame.shape) #480x640
        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not hasFrame:
            cv2.waitKey()
            break

        gray = frame

        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()

        # print(parameters)

        '''    detectMarkers(...)
            detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedI
            mgPoints]]]]) -> corners, ids, rejectedImgPoints
            '''
        # lists of ids and the corners beloning to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters, cameraMatrix=mtx, distCoeff=dist)


        # draw borders on the original image
        gray = aruco.drawDetectedMarkers(gray, corners)
        # draw a bounding box around the marker


        # if a tag has been detected
        if(len(corners)!=0):
            p1 = corners[0][0][0]
            p2 = corners[0][0][1]
            p3 = corners[0][0][2]
            p4 = corners[0][0][3]
            centroid = ((p1 + p2 + p3 + p4) / 4).astype(int)
            # pixel value has to be int
            width = int(3*np.linalg.norm(p1 - p2))

            # print(width)

            # draw circles around
            gray = cv2.rectangle(gray, (centroid[0] - width, centroid[1] - width),(centroid[0] + width, centroid[1] + width), (0, 255, 0), 2)

        #   check hand points tracking

        #   if a certain number of handpoints are within the boundary, we consider this an interaction
        #     check hands in here
            points = []
            # gray,interaction_status = hand_tracking(gray,protoFile,weightsFile,centroid,width)


        vid_writer.write(gray)

        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    vid_writer.release()
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    # downsample the video to speedup computation
    # downsample("HandPose/reference_video.mp4","HandPose/downsample",10)
    # calib_camera()
    generate_tag()
    # downsample("watering the plant.mp4",10)
    # detect_tag_and_hand()
    # perform hand tracking on a slowed down video
    # hand_tracking()