import cv2
import torch

from src.utils.imgutil import make_img_to_tensor

from .model import DDDModel


class ActivityClassifier:
    IMG_SIZE: tuple = (64, 64)

    def __init__(self, weight):
        # ------------------------------------------------------
        # 추론을 위한 모델 생성
        # 장비 설정
        self.device = torch.device("cpu")

        # Model 설정
        self.model = DDDModel(input_size=3, num_classes=10)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(weight, map_location=self.device))
        self.model.eval()

    def predict(self, img):
        # 학습에 사용된 사이즈로 이미지 수정
        img = cv2.resize(img, dsize=self.IMG_SIZE, interpolation=cv2.INTER_AREA)

        # 이미지를 Tensor로 수정
        img_tensor = make_img_to_tensor(img)
        img_tensor = img_tensor.to()

        # 이미지 추론
        result = self.model(img_tensor)
        return result.argmax()
