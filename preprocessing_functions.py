import numpy as np
import cv2
import cv2.aruco as aruco

import glob,time

def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
    # applying gray scale so we don't worry about colour

    try:
        image = cv2.resize(image, size)
        cv2.imshow('frame c', image)
    except Exception as e:
        print(str(e))
    # print(type(image))

    # this step is very important otherwise you won't be able to append arrays
    return image.flatten().reshape(1,image.flatten().shape[0])


def image_to_array(f_name,label):
    # f_name correspond to training video
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    # write a
    cap = cv2.VideoCapture(f_name)

    # read the first frame
    hasFrame, frame = cap.read()

    print(frame.shape)
    # resize and rotate the image
    frame = resize_img(frame, 50)

    # draw_img(frame)
    #  obtain the save data
    scale_x,scale_y,scale_w,scale_h,mtx,dist = load_saved_data()

    X  = np.array([[]])

    while (True):
        # Capture frame-by-frame
        hasFrame, frame = cap.read()

        if not hasFrame:
            # cv2.waitKey()
            break

        # resize and rotate the image
        gray = resize_img(frame,50)


        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()

        # print(parameters)
        # lists of ids and the corners beloning to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters, cameraMatrix=mtx, distCoeff=dist)

        gray = aruco.drawDetectedMarkers(gray, corners)

        if(len(corners)!=0):
            # get current centroid

            corner_cp = np.array(((corners[0][0][0] + corners[0][0][1] + corners[0][0][2] + corners[0][0][3]) / 4)).astype(int)

            # print('before',corner_cp)
            increment = [scale_x*(corners[0][0][0][0]-corner_cp[0]),scale_y*(corners[0][0][1][1] - corner_cp[1])]
            # print('increment',increment)
            bound_cp = corner_cp + increment
            # print('after',bound_cp)

            l_v = tuple(np.array([bound_cp[0]-scale_w*abs((corners[0][0][0][0] - corner_cp[0])),bound_cp[1]-scale_h*abs((corners[0][0][1][1] - corner_cp[1]))]).astype(int))
            r_v = tuple(np.array([bound_cp[0]+scale_w*abs((corners[0][0][0][0] - corner_cp[0])),bound_cp[1]+scale_h*abs((corners[0][0][1][1] - corner_cp[1]))]).astype(int))

            gray = cv2.rectangle(gray,l_v,r_v,color = (255,0,0))
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            # crop the image
            gray = gray[l_v[1]:r_v[1],l_v[0]:r_v[0]]

            # saved the cropped image to file
            frame_flattened = image_to_feature_vector(gray)

            if frame_flattened.shape[1]!=0:
                if X.shape[1] == 0:
                    X = frame_flattened
                else:
                    X = np.append(X, frame_flattened, axis=0)

            # only print the cropped image
            # cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if label == 'on':
        t = np.ones([X.shape[0],1])
    else:
        # t is zero if the label is off
        t = np.zeros([X.shape[0],1])

    print(label,X.shape,t.shape)

    cap.release()
    cv2.destroyAllWindows()
    return X,t

def extract_training_data():

    # cap = cv2.VideoCapture('stove pics/stove_off_vid.mp4')
    cap = cv2.VideoCapture('stove pics/training_off.mp4')


    hasFrame, frame = cap.read()

    draw_img(resize_img(frame,50))

    # extracting features from the training videos
    X0, t0 = image_to_array('stove pics/training_off.mp4', 'off')
    # X0, t0 = image_to_array('stove pics/tobii stove off training.mp4', 'off')

    X1, t1 = image_to_array('stove pics/training_on.mp4', 'on')
    # X1, t1 = image_to_array('stove pics/tobii stove on training.mp4', 'on')


    X = np.append(X1, X0, axis=0)
    t = np.append(t1, t0, axis=0)

    # applying PCA for some reason the accuracy is really high witout PCA
    mean, eigenvectors = cv2.PCACompute(X.astype(np.float32), mean=None, maxComponents=X.shape[0])

    X = (X-mean)@eigenvectors.T

    print(X.shape)

    np.savez('PCA parameters',mean = mean,eigenv = eigenvectors)
    np.savez("training data and labels",X = X,t = t)


def load_knn():
    # load saved data
    with np.load('training data and labels.npz') as X:
        X, t = [X[i] for i in ('X', 't')]

    knn = cv2.ml.KNearest_create()
    knn.train(X.astype(np.float32), cv2.ml.ROW_SAMPLE, t.astype(np.float32))

    return knn


def load_SVM():
    with np.load('training data and labels.npz') as X:
        X, t = [X[i] for i in ('X', 't')]

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

    svm.train(X.astype(np.float32), cv2.ml.ROW_SAMPLE, t.astype(int))

    # evaluating training accuracy


    return svm


def draw_img(img):

    # aurco marker detection
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    # draw borders on the original image
    img = aruco.drawDetectedMarkers(img, corners)

    # r returns the upper left and lower right coords
    r = cv2.selectROI("Image", img, fromCenter=False, showCrosshair=True)
    img = cv2.rectangle(img,(r[0],r[1]),(r[0] + r[2],r[1] + r[3]),(0, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(0)

    # save the coords of the bounding box and the corners of aruco marker
    np.savez("bounding_box", r=((r[0],r[1]),(r[0] + r[2],r[1] + r[3])),corners = corners)
    print("saving region of interest to file")
    return r


def resize_img(img,pct):
    # scale the original image by a factor 5 otherwise too big
    scale_percent = pct
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    img_resized = np.array([[]])

    try:
        img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    except Exception as e:
        print(str(e))

    # img_resized = cv2.resize(img,None,fx=0.2,fy=0.2)

    return img_resized



def load_saved_data():
    # load calib parameters from storage
    with np.load('calib_parameters.npz') as X:
        mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

    # load bounding box parameters from storage
    with np.load('bounding_box.npz') as X:
        r,corners= [X[i] for i in ('r', 'corners')]

    # get centroid for the bounding box
    bound_cp = np.array([(r[0][0]+r[1][0])/2,(r[0][1]+r[1][1])/2]).astype(int)
    corner_cp = np.array(((corners[0][0][0] + corners[0][0][1] + corners[0][0][2] + corners[0][0][3]) / 4)).astype(int)

    scale_x = (bound_cp[0]- corner_cp[0])/(corners[0][0][0][0]- corner_cp[0])
    scale_y = (bound_cp[1] -corner_cp[1])/(corners[0][0][1][1] - corner_cp[1])

    scale_w = abs((bound_cp[0]-r[0][0])/(corners[0][0][0][0] - corner_cp[0]))
    scale_h = abs((bound_cp[1]-r[0][1])/(corners[0][0][1][1] - corner_cp[1]))

    return scale_x,scale_y,scale_w,scale_h,mtx,dist