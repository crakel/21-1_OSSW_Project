import dlib
import cv2 as cv
import numpy as np
from imutils import face_utils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv.VideoCapture(0)

# range는 끝값이 포함안됨
ALL = list(range(0, 68))
RIGHT_EYEBROW = list(range(17, 22))
LEFT_EYEBROW = list(range(22, 27))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
NOSE = list(range(27, 36))
NOSE_TIP = list(range(30, 31))
MOUTH_OUTLINE = list(range(48, 61))
MOUTH_INNER = list(range(61, 68))
JAWLINE = list(range(0, 17))


# 유클리드 거리 계산 함수
def euclidean_dist(x, y):
    return np.linalg.norm(x - y)


# EAR 계산 함수
def eye_aspect_ratio(eye):
    A = euclidean_dist(eye[1], eye[5])
    B = euclidean_dist(eye[2], eye[4])
    C = euclidean_dist(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear


EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

COUNTER = 0
TOTAL = 0
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

index = ALL
location_list = []
history_list = []
Draw = True


def draw(img_frame, locations):
    for i in range(len(locations) - 1):
        if locations[0] is None or locations[1] is None:
            continue

        cv.line(img_frame, tuple(locations[i]), tuple(locations[i + 1]), (0, 255, 255), 3)

    return img_frame


while True:

    ret, img_frame = cap.read()
    h, w, c = img_frame.shape

    img_draw = np.ones((h, w), dtype=np.uint8) * 255
    img_frame = cv.flip(img_frame, 1)
    img_gray = cv.cvtColor(img_frame, cv.COLOR_BGR2GRAY)

    rects = detector(img_gray, 0)

    for rect in rects:
        shape = predictor(img_frame, rect)  # 얼굴에서 68개 점 찾기
        eye_shape = face_utils.shape_to_np(shape)

        leftEye = eye_shape[lStart:lEnd]
        rightEye = eye_shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv.convexHull(leftEye)
        rightEyeHull = cv.convexHull(rightEye)
        cv.drawContours(img_frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv.drawContours(img_frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1

        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            COUNTER = 0

        cv.putText(
            img_frame,
            "Blinks: {}".format(TOTAL),
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        cv.putText(
            img_frame,
            "EAR: {:.2f}".format(ear),
            (300, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        # 그리는 부분 ############################################
        list_points = []
        for p in shape.parts():
            list_points.append([p.x, p.y])

        list_points = np.array(list_points)

        # for i, pt in enumerate(list_points[index]):
        #     pt_pos = (pt[0], pt[1])
        #     cv.circle(img_frame, pt_pos, 2, (0, 255, 0), -1)
        #
        # cv.rectangle(img_frame, (face.left(), face.top()), (face.right(), face.bottom()),
        #              (0, 0, 255), 3)

        nose_x = list_points[NOSE_TIP][0][0]
        nose_y = list_points[NOSE_TIP][0][1]

        if Draw:
            location_list.append((nose_x, nose_y))
        else:
            history_list.append(location_list.copy())
            location_list.clear()

    img_frame = draw(img_frame, location_list)

    for locations in history_list:
        img_frame = draw(img_frame, locations)

    cv.imshow('result', img_frame)

    key = cv.waitKey(1)

    if key == 27:  # ESC
        break

    elif key == 32:  # space bar
        location_list.clear()
        history_list.clear()

    elif key == ord('v'):  # 이니셜 포인트 지정. 눈 깜빡임 컨트롤로 바꾸기
        Draw = not Draw

    elif key == ord('e'):
        Draw = False
        img_draw = draw(img_draw, location_list)
        for locations in history_list:
            img_draw = draw(img_draw, locations)
        # img_draw=cv.GaussianBlur(img_draw,(5,5),0)
        cv.imshow('draw_result', img_draw)

    elif key == ord('0'):
        index = NOSE_TIP
    elif key == ord('1'):
        index = ALL
    elif key == ord('2'):
        index = LEFT_EYEBROW + RIGHT_EYEBROW
    elif key == ord('3'):
        index = LEFT_EYE + RIGHT_EYE
    elif key == ord('4'):
        index = NOSE
    elif key == ord('5'):
        index = MOUTH_OUTLINE + MOUTH_INNER
    elif key == ord('6'):
        index = JAWLINE

cap.release()

#실행 취소 버튼 만들기(눈 연속 깜빡임?)
#입력 완료 버튼 만들기(눈 오래 감고 있기?)->입력 완료 시 그려진 그림을 내보내기
#생각보다 움직임의 범위가 커야 함
#고개를 숙이거나 좌우로 고개를 바꾸면 인식률이 떨어짐
#line smoothing
