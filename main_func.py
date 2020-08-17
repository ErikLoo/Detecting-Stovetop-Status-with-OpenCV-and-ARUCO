import numpy as np
import cv2
import cv2.aruco as aruco

import glob,time

from preprocessing_functions import extract_training_data,load_saved_data,load_SVM,resize_img,image_to_feature_vector

def track_object_status(f_name):


    # load PCA parameters
    with np.load("PCA parameters.npz") as Y:
        mean, eigenv = [Y[i] for i in ('mean', 'eigenv')]

    scale_x,scale_y,scale_w,scale_h,mtx,dist = load_saved_data()

    svm = load_SVM()

    frame_c = 0
    error_c = 0

    cap = cv2.VideoCapture(f_name)
    #

    ret, frame = cap.read()
    gray = resize_img(frame, 50)
    vid_writer = cv2.VideoWriter('stove pics/output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15,
                                 (gray.shape[1], gray.shape[0]))

    while (cap.isOpened()):
        # Capture frame-by-frame
        hasFrame, frame = cap.read()



        if not hasFrame:
            # cv2.waitKey()
            break
        # resize and rotate the image
        gray = resize_img(frame,50)


        color_frame = gray

        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()

        # print(parameters)
        # lists of ids and the corners beloning to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters, cameraMatrix=mtx, distCoeff=dist)

        if(len(corners)!=0):
            # get current centroid
            frame_c+=1

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

            # segment out the frame
            frame_flattened = image_to_feature_vector(gray[l_v[1]:r_v[1], l_v[0]:r_v[0]])


            if frame_flattened.shape[1]!=0:
                # ret, results, neighbours, _ = knn.findNearest(frame_flattened.astype(np.float32), 5)
                # appying PCA transformation
                frame_flattened = (frame_flattened - mean) @ eigenv.T
                results = svm.predict(frame_flattened)
                label_pred =results[1][0]

                if label_pred == 0:
                    interact_status = 'off'
                elif label_pred == 1:
                    interact_status = 'on'
                else:
                    interact_status = 'Unknown'

                # if results[1][0]!=label:
                #     error_c+=1
                #     interact_status = 'not detecting anything'
                # else:
                #     interact_status = label_text
                gray = cv2.putText(color_frame, interact_status, (int(bound_cp[0]), int(bound_cp[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255),thickness=3)

        # draw borders on the original image
        gray = aruco.drawDetectedMarkers(color_frame, corners)

        # rotate each frame by 90 degrees clockwise and resize it
        vid_writer.write(gray)
        cv2.imshow('frame', gray)



        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # if label == 0:
    #     status = 'off'
    # else:
    #     status = 'on'

    # print('framec',frame_c)
    # print(object_name,status,'svm accuracy:',(frame_c-error_c)/frame_c,frame_c-error_c,'/',frame_c)
    vid_writer.release()
    cap.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':

    extract_training_data()

    track_object_status('stove pics/test vid 1.mp4')
