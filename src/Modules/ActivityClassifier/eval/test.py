import cv2
import torch
import time
import torchvision.transforms as T
import numpy as np
from PIL import Image
from torchvision import models


def img2tensor(img):
    # the format of img needs to be bgr format

    img = img[..., ::-1]  # bgr2rgb
    img = img.transpose(2, 0, 1)  # (H, W, CH) -> (CH, H, W)
    img = np.ascontiguousarray(img)

    tensor = torch.tensor(img, dtype=torch.float32)
    return tensor


# CONFIG
# PATH = './weight/model_weight_3_1_4.pth'
# PATH = './weight/model_weight_1.pth'
# PATH = './weight/vgg16/vgg16_2.pth'
# PATH = './weight/vgg11/vgg11_7.pth'
PATH = '../weight/vgg11/my/vgg11_7.pth'
# IMG_PATH = './data/test1.JPG'
# IMG_PATH = './data/img2/img_3766.jpg'
IMG_PATH = './data/img/1.jpg'
# IMG_SIZE = (640, 480)
IMG_SIZE = (64, 64)
LABEL = [
    "safe driving",
    "texting - right",
    "talking on the phone - right",
    "texting - left",
    "talking on the phone - left",
    "operating the radio",
    "drinking",
    "reaching behind",
    "hair and makeup",
    "talking to passenger",
]


# Device 설정
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")
print(device)

# Model 설정
# model = DistractedDriverDetectionModel(input_size=3, num_classes=10)
model = models.vgg11()
# model = models.vgg16()
model = model.to(device)
model.load_state_dict(torch.load(PATH, map_location=device))
model.eval()

# # 캠 설정
# # cap = cv2.VideoCapture('./data/test.mp4')
# cap = cv2.VideoCapture(1)
#
# prevTime = 0
# while cap.isOpened():
#     # 실시간 이미지 불러오기
#     success, image = cap.read()
#     if not success:
#         print("카메라를 찾을 수 없습니다.")
#         # 동영상을 불러올 경우는 'continue' 대신 'break'를 사용합니다.
#         break
#
#     # 예측 이미지 생성
#     # image = cv2.flip(image, 0)
#     image = cv2.flip(image, 1)
#     predict_img = cv2.resize(image, dsize=IMG_SIZE, interpolation=cv2.INTER_AREA)
#     predict_img = img2tensor(predict_img)
#     predict_img = predict_img.to(device)
#     predict_img = predict_img.unsqueeze(dim=0)
#
#     result = model(predict_img)
#     label = LABEL[result.argmax()]
#
#     image = cv2.resize(image, dsize=(960, 540), interpolation=cv2.INTER_AREA)
#     currTime = time.time()
#     fps = 1 / (currTime - prevTime)
#     prevTime = currTime
#     cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
#     image = cv2.putText(image, label, (200, 100), 2, 1, (0, 0, 0), 3)
#
#     cv2.imshow('Test', image)
#
#     if cv2.waitKey(5) & 0xFF == 27:
#         break
# cap.release()
#
# exit()


img = cv2.imread(IMG_PATH)
img = cv2.flip(img, 1)
print(img.shape)
# img = cv2.flip(img, 1)
r_img = cv2.resize(img, dsize=IMG_SIZE, interpolation=cv2.INTER_AREA)
print(r_img.shape)
t_img = img2tensor(r_img)
print(t_img)
print(t_img.size())
t_img = t_img.unsqueeze(dim=0)
print(t_img.size())
t_img = t_img.to(device)

cv2.imshow('test', img)
cv2.waitKey(0)
cv2.imshow('test', r_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# train_transforms = T.Compose([
#     T.Resize((64, 64)),
#     T.ToTensor(),
# ])

# p_img = Image.open(IMG_PATH)
# p_img = train_transforms(p_img)
# p_img = p_img.to(device)

# print(p_img)
# print(p_img.size())
# p_img = p_img.unsqueeze(dim=0)
# print(p_img.size())

result = model(t_img)
print(result)
print(result.argmax())
