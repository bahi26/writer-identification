import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import label

#img = cv2.imread(r"G:\college\4th_year\Pattern_Recognition\Project\data\01\2\2.PNG")
def crop(gray):
    height, width = gray.shape[:2]

    # grayscale
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # binary
    ret, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)

    # kernel_size = 5
    # blur_gray = cv2.GaussianBlur(thresh,(kernel_size, kernel_size),0)

    kernel = np.ones((5, 5), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=2)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(img_dilation, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    check = 0.4 * width
    for line in lines:
        for x1, y1, x2, y2 in line:
            if (abs(x1 - x2) >= check):
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 5)

    labeled_img, num_connected_comp = label(line_image)
    objs = ndimage.find_objects(labeled_img)

    # list of tuples(x1,y1,x2,y2) each tuple for one connected component
    coorinates_bounding_rec = []

    for i in range(0, len(objs)):
        width_bounding_rec = (int(objs[i][1].stop) - 1) - (int(objs[i][1].start))
        height_bounding_rec = (int(objs[i][0].stop) - 1) - int(objs[i][0].start)
        if height_bounding_rec > 5:  # to exclude noise and points over letter
            coorinates_bounding_rec.append(
                (int(objs[i][1].start), int(objs[i][0].stop) - 1, int(objs[i][1].stop) - 1, int(objs[i][0].start)))
            cv2.rectangle(line_image, (coorinates_bounding_rec[len(coorinates_bounding_rec) - 1][0],
                                       coorinates_bounding_rec[len(coorinates_bounding_rec) - 1][1]), (
                          coorinates_bounding_rec[len(coorinates_bounding_rec) - 1][2],
                          coorinates_bounding_rec[len(coorinates_bounding_rec) - 1][3]), (0, 0, 255), 1)

    coorinates_bounding_rec.sort(key=lambda tup: tup[3])

    size = len(coorinates_bounding_rec)
    cropped = gray[coorinates_bounding_rec[size - 2][1]:coorinates_bounding_rec[size - 1][3], 0:width]
    return cropped