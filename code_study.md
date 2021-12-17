import cv2, dlib, sys  .cv2는 openCV library / dlib라이브러리에서는 HOG특성을 활용하거나 또는 대신 학습된 CNN모델을 사용할 수 있다. HOG는 픽셀값의 변화로 파악할 수 있는 영상 밝기 변화의 방향을 그래디언트로 표현하고, 이로부터 객체 형태를 찾아낼 수 있다. 얼굴 탐색 이외에도 보행자 검출 등에 활용할 수 있다. 
import numpy as np
 .행렬연산을 위한 numpy
 
scaler = 0.5  0~1 사이의 범위로 데이터를 표준화해주는 ‘0~1변환’
인공신경망, 딥러닝 할 때 변수들을 ‘0~1’범위로 변환해서 사용해야 한다.
. -> 이미지가 너무 크므로 이미지 크기 조정한 것이다. 
# initialize face detector and shape predictor
detector = dlib.get_frontal_face_detector() .dlib에 있는 정면 얼굴 검출기
기본 얼굴탐색 객체는 사실 만들어진 이미지 피라미드의 이미지들을 슬라이딩 윈도방식으로 탐색할 때, HOG특성을 사용한 뒤 그 결과를 분류하는 선형 분류기를 이용하는 방식으로 구현되어 있다. 
predictor =
 dlib.shape_predictor('openface/models/dlib/shape_predictor_68_face_landmarks.dat')
.dlib에 정의된 68개의 랜드마크 포인트를 활용하여 얼굴 구조를 파악하는 방법 (머신러닝에서 학습된 코드) 
# load video 
cap = cv2.VideoCapture(0) .카메라에서 이미지 가져오기 
# load overlay image
overlay = cv2.imread('samples/ryan_transparent.pngpek160114_273', cv2.IMREAD_UNCHANGED)
 .overlay할 이미지 경로 읽어오기, / UNCHANGED->파일 이미지를 BGRA(알파채널까지)타입으로 읽기 
# overlay function . 이미지를 동영상에 띄우는 함수 
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
  bg_img = background_img.copy()
  # convert 3 channels to 4 channels
  if bg_img.shape[2] == 3: .bg_img.shape[2]가 3과 같으면 
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA) 
 .cv2.COLOR_BGR2BGRA : RGB또는 BGR이미지에 알파 채널 추가 (알파채널은 각 화소에 대해 색상 표현의 데이터로부터 분리한 보조 데이터를 일컫는다. (색상채널 RGB,CMYK)자체에 작업을 할 수 없기 때문에 이미지에 영향을 끼치지 않는다.  
알파채널은 선택영역을 만들어 이미지의 선택영역을 정밀하게 수정할 수 있는 기능을 가지고 있다. 알파채널은 흰색과 검은색으로 이루어지는데, 흰색으로 된 부분을 수정할 수 있고, 검은색으로 이루어진 부분은 수정할 수 없다. (그레이 컬러 모두 256가지) 최종적으로 RGB색상채널에 효과를 적용. / 알파채널에서는 선택영역을 저장하고 불러올 수 있는 기능을 가지고 있는데, 저장된 선택 영역을 불러들여서 이미지에 적용이 가능. 
  if overlay_size is not None:  . overlay_size가 none이 아니면?
    img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)
 
  b, g, r, a = cv2.split(img_to_overlay_t) .split함수를 사용하여 색 채널별로 분리
(blue, green, red,  a는 알파채널) -하나의 색채널을 가지고 있기 때문에 grayscale로 보여지게 된다. 
 
  mask = cv2.medianBlur(a, 5)
. a:블러링 필터를 적용할 원본 이미지 , val=5: 커널 사이즈, val x val 크기의 박스 내에 있는 모든 픽셀들의 median 값을 취해서 중앙에 있는 픽셀에 적용함 /median filter는 화면에 소금-후추노이즈(소금과 후추를 뿌린 듯한 노이즈)를 제거하는데 매우 효과적이다. 
  h, w, _ = img_to_overlay_t.shape
  roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
 
  img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
  img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)
 
  bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)
 
  # convert 4 channels to 4 channels
  bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)
 
  return bg_img
 
face_roi = []
face_sizes = []
 
# loop
while True:   . 비디오가 끝날 때 까지 계속 읽어줘야 하므로. 
  # read frame buffer from video
  ret, img = cap.read()
  if not ret: . 만약 프레임이 없으면 break로 바로 종료시켜준다. 
    break
 
  # resize frame
  img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))
.img를 2번째 인수 dsize= (int(img.shape[1]`~)크기로 축소해준다.  resize메소드는 크기를 정수형으로 받기 때문에 int를 꼭 써줘야 한다. 
  ori = img.copy()
. 원본이미지를 ori에 저장
  # find faces
  if len(face_roi) == 0: 
    faces = detector(img, 1)  
  else:
    roi_img = img[face_roi[0]:face_roi[1], face_roi[2]:face_roi[3]]
    # cv2.imshow('roi', roi_img)
    faces = detector(roi_img)
 
  # no faces
  if len(faces) == 0:
    print('no faces!')
 
  # find facial landmarks 
  for face in faces:
    if len(face_roi) == 0:
      dlib_shape = predictor(img, face)
      shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])
    else:
      dlib_shape = predictor(roi_img, face) 
. predictor(img,face) : img의 face영역안의 얼굴 특징점 찾기, 이미지와 구한 얼굴 영역들어감
      shape_2d = np.array([[p.x + face_roi[2], p.y + face_roi[0]] for p in dlib_shape.parts()])
. dlib객체를 numpy객체로 변환(연산을 쉽게하기위해) 
    for s in shape_2d:
      cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
. 얼굴 특징점은 68개인데 for loop문을 돌면서 68개 점을 opencv의 circle을 이용하여 점을 그린다. ->특징점 구한이유는 얼굴의 좌상단,우하단 구해서 얼굴 사이즈 구하고, 얼굴중심 구할 것.  
    # compute face center
    center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)
. 얼굴 중심 구하기 np.mean() 평균구하기 / np.astype(np.int) : numpy행렬을 np.int타입으로 변환 
    # compute face boundaries
    min_coords = np.min(shape_2d, axis=0) . 얼굴 좌상단
    max_coords = np.max(shape_2d, axis=0) . 얼굴 우하단 (np.max():최대값 찾기)
 
    # draw min, max coords . (circle그려줌 . 코드 필요없음)
    cv2.circle(img, center=tuple(min_coords), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(img, center=tuple(max_coords), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
. 두께는 2 선의 타입은 cv2.LINE_AA(?) 이미지에 동그랗게 그려준다?
    # compute face size
    face_size = max(max_coords - min_coords)
    face_sizes.append(face_size)
    if len(face_sizes) > 10:
      del face_sizes[0]
    mean_face_size = int(np.mean(face_sizes) * 1.3)
.(*1.8) 라이언 사이즈 조정. 
 
    # compute face roi
    face_roi = np.array([int(min_coords[1] - face_size / 2), int(max_coords[1] + face_size / 2), int(min_coords[0] - face_size / 2), int(max_coords[0] + face_size / 2)])
    face_roi = np.clip(face_roi, 0, 10000)
 
    # draw overlay on face
    result = overlay_transparent(ori, overlay, center_x + 8, center_y - 25, overlay_size=(mean_face_size, mean_face_size))
. 라이언 이미지를 센터 X+8, Y-25중심으로 놓고, OVERLAY사이즈만큼 resize해서 원본 이미지에 넣어준다. 그결과물을 result에 저장
  # visualize
  cv2.imshow('original', ori)
  cv2.imshow('facial landmarks', img) 
  cv2.imshow('result', result)   . 동영상을 읽어서 이미지 띄우기 
  if cv2.waitKey(1) == ord('q'): 
 . 1밀리세컨드만큼 대기/ 이걸 넣어야 동영상이 제대로 보인다. 
    sys.exit(1)
 
. 좌상단: face.left(), face.top()
 우하단: face.right(), face.bottom() 
 
*이미지는 배경이 투명해야 한다 (->png파일이어야 한다)
