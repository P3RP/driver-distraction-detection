import cv2
import torch
import time
import torchvision.transforms as T
import numpy as np
from PIL import Image
from torchvision import models

from src.utils.dirutil import get_root_path
from src.Modules.ActivityClassifier.model import DDDModel


def img2tensor(img):
    # the format of img needs to be bgr format

    img = img[..., ::-1]  # bgr2rgb
    img = img.transpose(2, 0, 1)  # (H, W, CH) -> (CH, H, W)
    img = np.ascontiguousarray(img)

    tensor = torch.tensor(img, dtype=torch.float32)
    return tensor


def make_tensor(img):
    pilimg = np.array(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    imgTensor = T.ToTensor()(pilimg)
    imgTensor = imgTensor.unsqueeze(0)

    return imgTensor


# CONFIG
PATH = '../weight/ddd/model_weight_3_1_4.pth'
# PATH = './weight/model_weight_1.pth'
# PATH = './weight/vgg16/vgg16_2.pth'
# PATH = './weight/vgg11/vgg11_7.pth'
# PATH = './weight/vgg11/my/vgg11_7.pth'
# IMG_PATH = './data/test1.JPG'
# IMG_PATH = './data/img2/img_3766.jpg'
IMG_PATH = './data/img/9.jpg'
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
model = DDDModel(input_size=3, num_classes=10)
# model = models.vgg11()
# model = models.vgg16()
model = model.to(device)
model.load_state_dict(torch.load(PATH, map_location=device))
model.eval()

# 캠 설정
cap = cv2.VideoCapture('../../../../data/side_mod.mp4')
# cap = cv2.VideoCapture('../../../../data/side-1.mov')
# cap = cv2.VideoCapture(0)

prevTime = 0
while cap.isOpened():
    # 실시간 이미지 불러오기
    success, image = cap.read()
    if not success:
        print("카메라를 찾을 수 없습니다.")
        # 동영상을 불러올 경우는 'continue' 대신 'break'를 사용합니다.
        break

    # 예측 이미지 생성
    # image = cv2.flip(image, 0)
    # image = cv2.flip(image, 1)
    predict_img = cv2.resize(image, dsize=IMG_SIZE, interpolation=cv2.INTER_AREA)
    # predict_img = img2tensor(predict_img)
    predict_img = make_tensor(predict_img)
    predict_img = predict_img.to(device)
    # predict_img = predict_img.unsqueeze(dim=0)

    result = model(predict_img)
    label = LABEL[result.argmax()]

    image = cv2.resize(image, dsize=(960, 540), interpolation=cv2.INTER_AREA)
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
    image = cv2.putText(image, label, (200, 100), 2, 1, (0, 0, 0), 3)

    cv2.imshow('Test', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()

exit()


img = cv2.imread(IMG_PATH)
r_img = cv2.resize(img, dsize=IMG_SIZE, interpolation=cv2.INTER_AREA)
pilimg = np.array(Image.fromarray(cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)))
imgTensor = T.ToTensor()(pilimg)
imgTensor = imgTensor.unsqueeze(0)

# PILimg = np.array(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
# imgTensor = transforms.ToTensor()(PILimg)
# # imgTensor, _ = pad_to_square(imgTensor, 0)
# imgTensor = resize(imgTensor, 416)
# imgTensor = imgTensor.unsqueeze(0)
# imgTensor = Variable(imgTensor.type(Tensor))


# print(r_img.shape)
# t_img = img2tensor(r_img)
# print(t_img)
# print(t_img.size())
# t_img = t_img.unsqueeze(dim=0)
# print(t_img.size())
# t_img = t_img.to(device)

# cv2.imshow('test', img)
# cv2.waitKey(0)
# cv2.imshow('test', r_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# train_transforms = T.Compose([
#     T.Resize((64, 64)),
#     T.ToTensor(),
# ])
#
# p_img = Image.open(IMG_PATH)
# p_img = train_transforms(p_img)
# p_img = p_img.to(device)
#
# print(p_img)
# print(p_img.size())
# p_img = p_img.unsqueeze(dim=0)
# print(p_img.size())

def colorBackgroundText(img, text, font, fontScale, textPos, textThickness=1, textColor=(0, 255, 0), bgColor=(0, 0, 0),
                        pad_x=3, pad_y=3):
    """
    Draws text with background, with  control transparency
    @param img:(mat) which you want to draw text
    @param text: (string) text you want draw
    @param font: fonts face, like FONT_HERSHEY_COMPLEX, FONT_HERSHEY_PLAIN etc.
    @param fontScale: (double) the size of text, how big it should be.
    @param textPos: tuple(x,y) position where you want to draw text
    @param textThickness:(int) fonts weight, how bold it should be
    @param textPos: tuple(x,y) position where you want to draw text
    @param textThickness:(int) fonts weight, how bold it should be.
    @param textColor: tuple(BGR), values -->0 to 255 each
    @param bgColor: tuple(BGR), values -->0 to 255 each
    @param pad_x: int(pixels)  padding of in x direction
    @param pad_y: int(pixels) 1 to 1.0 (), controls transparency of  text background
    @return: img(mat) with draw with background
    """
    (t_w, t_h), _ = cv2.getTextSize(text, font, fontScale, textThickness)  # getting the text size
    x, y = textPos
    cv2.rectangle(img, (x - pad_x, y + pad_y), (x + t_w + pad_x, y - t_h - pad_y), bgColor, -1)  # draw rectangle
    cv2.putText(img, text, textPos, font, fontScale, textColor, textThickness)  # draw in text

    return img

# result = model(t_img)
# result = model(p_img)
result = model(imgTensor)
# print(result)
print(int(result.argmax()))

image = cv2.resize(img, dsize=(960, 540), interpolation=cv2.INTER_AREA)

image = cv2.flip(image, 1)
image = colorBackgroundText(image, f'{LABEL[int(result.argmax())]}', 2, 2, (0, 500), 1, (255, 255, 255))
# image = cv2.putText(image, label, (200, 100), 2, 1, (0, 0, 0), 3)
cv2.imshow('test', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
