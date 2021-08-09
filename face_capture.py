import cv2
import uuid
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required=True,help="input first name")
args = vars(ap.parse_args())

# nameClass = input('Enter your first name : ')
os.mkdir(os.path.join('./dataset/',args["name"]))

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    cv2.imwrite('./dataset/'+args["name"]+'/'+args["name"]+str(uuid.uuid4())+'.jpg',frame)
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
