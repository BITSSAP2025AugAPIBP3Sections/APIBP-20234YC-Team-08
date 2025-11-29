from PIL import Image

def crop_image(img):
    rect_length = 166
    rect_height = 40

    rect_offset_x = 307
    rect_offset_y = 0

    crop_area = (rect_offset_x, rect_offset_y, rect_offset_x + rect_length, rect_offset_y + rect_height)

    cropped_img = img.crop(crop_area)

    cropped_img.show()

def contour_images(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 50 and h > 20:  # filter out tiny noise
            roi = img[y:y+h, x:x+w]
            cv2.imwrite(f"samples/region_{x}_{y}.jpg", roi)
