import cv2
import matplotlib.pyplot as plt
import time


def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



if __name__ == '__main__':
    test1 = cv2.imread('image1.jpg')
    gray_img = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray_img, cmap='gray')
    plt.show()

    haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

    print('Faces found:', len(faces))

    for (x, y, w, h) in faces:
        cv2.rectangle(test1, (x, y), (x+w, y+h), (0, 255, 0), 2)

    plt.imshow(convertToRGB(test1))
    plt.show()
