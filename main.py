from argparse import ArgumentParser
import cv2

parser = ArgumentParser()
parser.add_argument("filename")
args = parser.parse_args()

cap = cv2.VideoCapture(args.filename)
bgSubtractor = cv2.createBackgroundSubtractorMOG2()

while cap.isOpened():
    ret, frame = cap.read()
    blur = cv2.GaussianBlur(frame, (5, 5), 2)
    fg = bgSubtractor.apply(blur)

    _, threshold = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)

    opening = cv2.erode(
        threshold, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=20
    )
    opening = cv2.dilate(
        opening, cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20)), iterations=4
    )

    contours = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if ret == True:
        cv2.imshow("Frame", frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
