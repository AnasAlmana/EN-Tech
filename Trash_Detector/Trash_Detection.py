import torch
from ultralytics import YOLO
import cv2

print(torch.cuda.is_available())

model_objects = YOLO("D:/YOLOv8/ultralytics/yolov8n.pt") #trash model
vid = cv2.VideoCapture(2)
classes = list(range(0,80))
human_bottle = [0, 39]
while (True):
    ret, annotated_frame = vid.read() # Open the camera and read the frame

    #-----------------------1st detect the human ----------------------
    human_model = model_objects.track(annotated_frame, conf=0.75)
    annotated_frame = human_model[0].plot()


    #----------------------2nd cut the detected person-----------------
    data = getattr(human_model[0], "boxes")
    all_boxes = getattr(data,"boxes") # all the boxes in the image will be stored in this variable<<
    """points = points.cpu()
    points = points[0].numpy()"""

    print(all_boxes)



    #----------------------3rd detect the trash inside the box of the person---------
    trash_model = model_objects.track(annotated_frame,classes = [39], conf=0.75)
    annotated_frame = trash_model[0].plot()

    cv2.imshow('frame', annotated_frame)
    #boxes = getattr(result[0][0], "boxes")
    #id = result[0].Boxes()


    if cv2.waitKey(1) & 0xFF == ord('q'):
        vid.release()
        break