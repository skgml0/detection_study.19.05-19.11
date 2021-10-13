import dlib, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

# 얼굴을 파악하기 위한 라이브러리들
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

#문 개폐를 위한 확인 배열
identi = []

def load_image():
    cap = cv2.VideoCapture('http://192.168.0.9:8080/?action=stream') #0 or -1

    while cap.isOpened():
        result, frame = cap.read()
        if result:
            #gray=cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == 27: #esc
                break
        else:
            print('no camera!')
            break
    cap.release()
    cv2.destroyAllWindows()

# a = load_image()

# 얼굴찾는 함수
def find_faces(img):
    dets = detector(img, 1)

    if len(dets) == 0:
        return np.empty(0), np.empty(0), np.empty(0)

    rects, shapes = [], []
    shapes_np = np.zeros((len(dets), 68, 2), dtype=np.int)
    for k, d in enumerate(dets):
        rect = ((d.left(), d.top()), (d.right(), d.bottom()))
        rects.append(rect)

        shape = sp(img, d)

        # dlib shape을 numpy array로 변환
        for i in range(0, 68):
            shapes_np[k][i] = (shape.part(i).x, shape.part(i).y)

        shapes.append(shape)

    return rects, shapes, shapes_np
# 얼굴에 대한 인코딩 함수
def encode_faces(img, shapes):
    face_descriptors = []
    for shape in shapes:
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        face_descriptors.append(np.array(face_descriptor))

    return np.array(face_descriptors)
# 인물에 대한 사진 수집
img_paths = {
    'leej': 'img/lee.jpg',
    'han': 'img/han.jpg',
    'parkh': 'img/parkh.jpg',
    'jae': 'img/jae.jpg'
}
# 인물들의 사진 배열초기화
descs = {
    'lee': None,
    'han': None,
    'parkh': None,
    'jae': None
}

for name, img_path in img_paths.items():
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    _, img_shapes, _ = find_faces(img_rgb)
    descs[name] = encode_faces(img_rgb, img_shapes)[0]

np.save('img/descs.npy', descs)
print(descs)

import urllib.request

urllib.request.urlretrieve('http://192.168.0.9:8090/?action=snapshot', 'img/test.png')

img_bgr = cv2.imread('img/test.png') #cv2가 경로에 해당하는 사진을 bgr로 읽어옴

while img_bgr.isOpened():
    result, frame = img_bgr.read()
    if result:
        #gray=cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == 27: #esc
            break
    else:
        print('no camera!')
        break
    img_bgr.release()
    cv2.destroyAllWindows()

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) #cv2가 bgr에서 rgb로 변환

rects, shapes, _ = find_faces(img_rgb)
descriptors = encode_faces(img_rgb, shapes)

fig, ax = plt.subplots(1, figsize=(20, 20))
ax.imshow(img_rgb)

for i, desc in enumerate(descriptors):

    found = False
    for name, saved_desc in descs.items():
        dist = np.linalg.norm([desc] - saved_desc, axis=1) #유클리디안 디스턴스 함수

        if dist < 0.35: #좌표상 얼굴들 좌표에 대한 거리가 0.4 미만일 때 수행.
            found = True

            text = ax.text(rects[i][0][0], rects[i][0][1], name,
                           color='b', fontsize=40, fontweight='bold')
            text.set_path_effects([path_effects.Stroke(linewidth=10, foreground='white'), path_effects.Normal()])
            rect = patches.Rectangle(rects[i][0],
                                     rects[i][1][1] - rects[i][0][1],
                                     rects[i][1][0] - rects[i][0][0],
                                     linewidth=2, edgecolor='w', facecolor='none')
            ax.add_patch(rect)
            identi.append(1)

            break

    if not found:
        ax.text(rects[i][0][0], rects[i][0][1], 'unknown',
                color='r', fontsize=20, fontweight='bold')
        rect = patches.Rectangle(rects[i][0],
                                 rects[i][1][1] - rects[i][0][1],
                                 rects[i][1][0] - rects[i][0][0],
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        identi.append(0) # 식별하지 못했을 경우 0 입력



if(0 in identi):
    print(0, '인식 불가') #사진에 unKnown이 있을 경우
else:
    print(1, '통과') #사진이 얼굴을 다 인식했을 경우

plt.axis('off')
plt.savefig('result/output.png') #결과물 저장
plt.show()