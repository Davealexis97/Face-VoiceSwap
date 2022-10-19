import cv2
import numpy as np
import dlib
import time


def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


photo = cv2.imread("img1.jpg")
photo_gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
msk = np.zeros_like(photo_gray)
photo2 = cv2.imread("img2.jpg")
photo2_gray = cv2.cvtColor(photo2, cv2.COLOR_BGR2GRAY)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
height, width, channels = photo2.shape
photo2_new_face = np.zeros((height, width, channels), np.uint8)




# Face 1
faces = detector(photo_gray)
for face in faces:
    landmarks = predictor(photo_gray, face)
    landmarks_points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x, y))



    points = np.array(landmarks_points, np.int32)
    convexhull = cv2.convexHull(points)
    # cv2.polylines(photo, [convexhull], True, (255, 0, 0), 3)
    cv2.fillConvexPoly(msk, convexhull, 255)

    face_image_1 = cv2.bitwise_and(photo, photo, msk=msk)

    # Delaunay triangulation
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])


        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)

        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)

        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)



# Face 2
faces2 = detector(photo2_gray)
for face in faces2:
    landmarks = predictor(photo2_gray, face)
    landmarks_points2 = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points2.append((x, y))


    points2 = np.array(landmarks_points2, np.int32)
    convexhull2 = cv2.convexHull(points2)

lines_space_msk = np.zeros_like(photo_gray)
lines_space_new_face = np.zeros_like(photo2)
# Triangulation of both faces
for triangle_index in indexes_triangles:
    # Triangulation of the first face
    tr1_pt1 = landmarks_points[triangle_index[0]]
    tr1_pt2 = landmarks_points[triangle_index[1]]
    tr1_pt3 = landmarks_points[triangle_index[2]]
    triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)


    rect1 = cv2.boundingRect(triangle1)
    (x, y, w, h) = rect1
    cropped_triangle = photo[y: y + h, x: x + w]
    cropped_tr1_msk = np.zeros((h, w), np.uint8)


    points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                       [tr1_pt2[0] - x, tr1_pt2[1] - y],
                       [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

    cv2.fillConvexPoly(cropped_tr1_msk, points, 255)

    # Lines space
    cv2.line(lines_space_msk, tr1_pt1, tr1_pt2, 255)
    cv2.line(lines_space_msk, tr1_pt2, tr1_pt3, 255)
    cv2.line(lines_space_msk, tr1_pt1, tr1_pt3, 255)
    lines_space = cv2.bitwise_and(photo, photo, msk=lines_space_msk)

    # Triangulation of second face
    tr2_pt1 = landmarks_points2[triangle_index[0]]
    tr2_pt2 = landmarks_points2[triangle_index[1]]
    tr2_pt3 = landmarks_points2[triangle_index[2]]
    triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)


    rect2 = cv2.boundingRect(triangle2)
    (x, y, w, h) = rect2

    cropped_tr2_msk = np.zeros((h, w), np.uint8)

    points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                        [tr2_pt2[0] - x, tr2_pt2[1] - y],
                        [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

    cv2.fillConvexPoly(cropped_tr2_msk, points2, 255)

    # Warp triangles
    points = np.float32(points)
    points2 = np.float32(points2)
    M = cv2.getAffineTransform(points, points2)
    warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, msk=cropped_tr2_msk)

    # Reconstructing destination face
    photo2_new_face_rect_area = photo2_new_face[y: y + h, x: x + w]
    photo2_new_face_rect_area_gray = cv2.cvtColor(photo2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
    _, msk_triangles_designed = cv2.threshold(photo2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, msk=msk_triangles_designed)

    photo2_new_face_rect_area = cv2.add(photo2_new_face_rect_area, warped_triangle)
    photo2_new_face[y: y + h, x: x + w] = photo2_new_face_rect_area



# Face swapped (putting 1st face into 2nd face)
photo2_face_msk = np.zeros_like(photo2_gray)
photo2_head_msk = cv2.fillConvexPoly(photo2_face_msk, convexhull2, 255)
photo2_face_msk = cv2.bitwise_not(photo2_head_msk)


photo2_head_noface = cv2.bitwise_and(photo2, photo2, msk=photo2_face_msk)
result = cv2.add(photo2_head_noface, photo2_new_face)

(x, y, w, h) = cv2.boundingRect(convexhull2)
center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

seamlessclone = cv2.seamlessClone(result, photo2, photo2_head_msk, center_face2, cv2.NORMAL_CLONE)

cv2.imshow("seamlessclone", seamlessclone)
cv2.waitKey(0)



cv2.destroyAllWindows()
