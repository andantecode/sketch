from typing import Callable, List
import streamlit as st
import pytorch_lightning as pl
from transformers import ConvNextV2ForImageClassification
from albumentations.pytorch import ToTensorV2
import pandas as pd
import albumentations as A
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import numpy as np
from PIL import Image

st.set_page_config(page_title='Sketch Image Classification', layout='wide')

st.title(':crystal_ball: Sketch Image Classifier in 500 classes:crystal_ball:')
st.markdown('---')

st.info('분류하고 싶은 이미지를 업로드 해주세요!')
uploaded_file = st.file_uploader('이미지 업로드', type=['jpg', 'png', 'jpeg'], label_visibility="hidden")

class AlbumentationsTransform:
    def __init__(self, is_train: bool = True):
        # 공통 변환 설정: 이미지 리사이즈, 정규화, 텐서 변환
        common_transforms = [
            A.Resize(224, 224),  # 이미지를 224x224 크기로 리사이즈
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 정규화
            ToTensorV2()  # albumentations에서 제공하는 PyTorch 텐서 변환
        ]

        # 검증/테스트용 변환: 공통 변환만 적용
        self.transform = A.Compose(common_transforms)

    def __call__(self, image) -> torch.Tensor:
        # 이미지가 NumPy 배열인지 확인
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a NumPy array (OpenCV format).")

        # 이미지에 변환 적용 및 결과 반환
        transformed = self.transform(image=image)  # 이미지에 설정된 변환을 적용

        return transformed['image']  # 변환된 이미지의 텐서를 반환

class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        
        log_probs = torch.nn.functional.log_softmax(pred, dim=-1)  # Apply softmax to get log-probabilities
        # Compute loss
        return torch.mean(torch.sum(-target * log_probs, dim=-1))
    
class Eva02ConvNextClassifier(pl.LightningModule):
    def __init__(self, num_classes=500, lr=3e-5, weight_decay=1e-2):
        super().__init__()
        
        self.convnext = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-large-22k-224")
        self.eva02 = timm.create_model('eva02_large_patch14_clip_224.merged2b', pretrained=True, num_classes=0)

        self.convnext.classifier = nn.Identity()

        self.convnext_output_dim = 1536
        self.eva02_output_dim = 1024
        
        combined_dim = self.eva02_output_dim + self.convnext_output_dim # self.swin_output_dim + 

        self.classifier = nn.Linear(combined_dim, num_classes)
        
        self.lr = lr

        self.weight_decay = weight_decay
        self.loss_fn_crossentropy = nn.CrossEntropyLoss()
        self.loss_fn = SoftTargetCrossEntropy()


    def forward(self, pixel_values):
        convnext_features = self.convnext(pixel_values).logits

        eva02_output = self.eva02.forward_features(pixel_values)
        eva02_features = self.eva02.forward_head(eva02_output, pre_logits=True)

        combined_features = torch.cat((convnext_features, eva02_features), dim=1)
        
        # 결합된 특징을 classifier에 통과시켜 최종 출력
        logits = self.classifier(combined_features)
        return logits

@st.cache_resource
def load_model():
    '''
    sktech image classifier load
    '''
    models = []
    root_path = '../checkpoints/eva02convnext2_large_mixupcutmix_kfold'
    for dirpath, dirnames, filenames in os.walk(root_path):
        for file in filenames:
            file_path = os.path.join(dirpath, file)
            models.append(Eva02ConvNextClassifier.load_from_checkpoint(file_path))
    
    return models

def inference(
    models: List,
    device: torch.device,
    transform: Callable,
    uploaded_file
):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image)
    image = image.unsqueeze(0)

    for model in models:
        model.to(device)
        model.eval()
    
    with torch.no_grad():
        image = image.to(device)

        all_logits = []
        for model in models:
            logits = model(image)
            all_logits.append(logits)
        
        avg_logits = torch.mean(torch.stack(all_logits), dim=0)
        avg_probs = F.softmax(avg_logits, dim=1)
        preds = avg_probs.argmax(dim=1)
        predicted_probabilities = avg_probs[torch.arange(avg_probs.size(0)), preds]

    return preds, predicted_probabilities

def show_result(uploaded_file, probability, info_df: pd.DataFrame, class_name: int):
    original_image = Image.open(uploaded_file)

    class_info = info_df[info_df['target'] == class_name].iloc[0]
    class_image_path = os.path.join('../../data/train', class_info['image_path'])
    class_image = Image.open(class_image_path)

    col1, col2 = st.columns(2)

    with col1:
        st.image(original_image.resize((512, 512)), caption='업로드 한 이미지', use_column_width=True)
    
    with col2:
        st.image(class_image.resize((512, 512)), caption=f'클래스 {class_name}의 예시 이미지', use_column_width=True)
    
    if probability.item() > 0.5:
        st.info(f'예측된 클래스는 {class_name}이고, 신뢰도는 {probability.item():.2%}입니다.')
    else:
        st.error(f'예측된 클래스는 {class_name}이고, 신뢰도는 {probability.item():.2%}입니다.')

    
transform = AlbumentationsTransform(is_train=False)
models = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_info = pd.read_csv('../../data/train.csv')

if st.button("예측하기"):
    # 모델 예측
    result, probability = inference(models, device, transform, uploaded_file)
    # 예측 결과를 보여주는 부분 (가상의 예측 결과)
    show_result(uploaded_file, probability, train_info, result.item())





