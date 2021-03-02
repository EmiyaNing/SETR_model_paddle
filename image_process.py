import cv2

def image_process(path):
    img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] > 58:
                img[i][j] = 0

    cv2.imwrite(path, img)



if __name__ == '__main__':
    path = "./dummy_data\GroundTruth_trainval_png/2008_000009.png"

    img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] > 58:
                img[i][j] = 0

    cv2.imwrite(path, img)