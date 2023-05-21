from ultralytics import YOLO
import torch
import cv2
import numpy
from PIL import Image
import os
print(torch.cuda.is_available())
model_pose = YOLO("C:/Trash_Detector/yolov8x-pose-p6.pt") # hands model
model_objects = YOLO("C:/Trash_Detector/yolov8x.pt") #trash model
vid = cv2.VideoCapture(0)
track = 0
i = 0
violation = False

def dist(p1,p2):
    return ( abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) )

frame_width = int(vid.get(3))
frame_height = int(vid.get(4))

size = (frame_width, frame_height)
buffer = cv2.VideoWriter('C:/Trash_Detector/buffer/temp.avi',
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             4, size)
while True:
    ret, frame = vid.read()



    # Detect the human and pose:
    human_pose = model_pose.track(frame)

    try:
        # hands coordinates:
        hands = getattr(human_pose[0][0], "keypoints")
        hands = hands.cpu()
        hands = hands.numpy()
        p1_lhand = [int(hands[9][0]), int(hands[9][1])]
        p2_rhand = [int(hands[10][0]), int(hands[10][1])]

        s= 30
        face_s = [int(hands[0][0]) - s, int(hands[0][1]) - s]
        face_e = [int(hands[0][0]) + s, int(hands[0][1]) + s]
        frame_1 = cv2.rectangle(frame, face_s, face_e, color=(0, 255, 0), thickness=1)


        # Human box coordinates:
        Human_boxes = getattr(human_pose[0][0], "boxes")
        Human_boxes = getattr(Human_boxes[0][0], "xyxy")
        Human_boxes = Human_boxes.cpu()
        Human_boxes = Human_boxes.numpy()[0]
        st_p = [int(Human_boxes[0]), int(Human_boxes[1])] #starting point of the person box, x1H y1H
        en_p = [int(Human_boxes[2]), int(Human_boxes[3])] #Ending Point of the person box, x2H y2H
        #frame_1 = cv2.rectangle(frame_1, st_p, en_p, color=(0, 255, 0), thickness=3)




        # Fed the new image to second model to detect the trash:
        trash = model_objects.track(frame_1, classes=39, conf=0.1)
        try:
            # Take the coordinates of the trash & convert its coordinates to the original image:
            trash_box = getattr(trash[0][0], "boxes")
            trash_box = getattr(trash_box[0][0], "xyxy")
            trash_box = trash_box.cpu()
            trash_box = trash_box.numpy()[0]
            st_b = [trash_box[0], trash_box[1]]  # x1b y1b
            en_b = [trash_box[2], trash_box[3]]  # x2b y2b
            cen_object = [int((st_b[0]+en_b[0])/2),int((st_b[1]+en_b[1])/2)]

            # (annotated_frame, (int(keypoints[10][0]), int(keypoints[10][1])), radius=2,color=(255, 0, 0), thickness=10)
            frame_1 = cv2.line(frame_1, p1_lhand, cen_object, color=(0,255,0), thickness=1)
            dist_l = dist(p1_lhand, cen_object)

            frame_1 = cv2.line(frame_1, p2_rhand, cen_object, color=(0, 255, 0), thickness=1)
            dist_r = dist(p2_rhand, cen_object)
            sum_dist = dist_r + dist_l

            frame_1 = cv2.putText(img=frame_1, text='dist:' +str(sum_dist),org=(0, 70),fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale=1.0,color=(0, 0, 255),thickness=1)
            frame_1 = trash[0].plot()
            buffer.write(frame_1)

            if int(sum_dist) > 450:
                directory = r'C:/Trash_Detector/faces/'
                os.chdir(directory)
                # slice the image:
                # [y1:y2,x1:x2]
                img_crop = frame[face_s[1]:face_e[1], face_s[0]:face_e[0]]
                print(face_s, face_e)
                #image = Image.fromarray(img_crop)  # the cropped image of the person
                filename = 'face'+str(i)+'.jpg'
                cv2.imwrite(filename, img_crop)
                i+=1


            cv2.imshow('frame', frame_1)
        except:
            cv2.imshow('frame', frame_1)
            buffer.write(frame_1)


    except:
        cv2.imshow('frame', frame)
        buffer.write(frame)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        vid.release()
        buffer.release()
        break