import cv2
import pytesseract

img = cv2.imread("samples/sample_3.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

data = pytesseract.image_to_data(th, output_type=pytesseract.Output.DICT)
boxes = []
n = len(data['level'])

for i in range(n):
    text = data['text'][i].strip()

    conf = int(data['conf'][i])

    if text:

        (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])

        boxes.append((x, y, x + w, y + h))

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("tesseract boxes", img);
cv2.waitKey(0);
cv2.destroyAllWindows()
