import cv2
import uuid

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
i = 0
while True:
    ret, frame = cap.read()
    cv2.imwrite('./Images/cam'+str(uuid.uuid4())+'.jpg',frame)
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()