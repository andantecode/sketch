# 🤗 [CV 기초 프로젝트] 스케치 이미지 분류 경진대회 🤗

이 레포지토리는 스케치 이미지 분류 프로젝트를 진행하며 관련 내용을 기록하고자 만들었습니다.

## 프로젝트

- 사용 언어: python
- 사용 라이브러리: pytorch, pytorch-lightning

- 주요 폴더 설명


- **experiments**: 데이터 증강 기법 및 모델링을 진행하며 작성한 노트북      
    - **basic**: 단일 모델 모델링      
    - **ensemble**: 앙상블 모델링   
        - **bagging**: 2개 이상의 모델의 예측 결과를 이용해 hard voting 또는 soft voting 하여 추론 실험   
        - **snapshot**: 단일 모델의 다양한 시점에서의 가중치를 활용하여 voting 하는 방식으로 추론 실험   
        - **stacking**: 2개의 모델을 활용하여 마지막 layer의 feature map을 concat하여 classifier의 input으로 활용하여 모델 학습 및 추론 실험   
        - **kfold**: dataset의 fold를 k개로 분할하여 각 fold가 모두 validation_set으로 활용되게끔 k개의 모델을 학습시키고 해당 모델들을 활용한 앙상블 추론   
    - **etc**: 실험하며 사용했던 temp 노트북, accuracy를 가늠하기 위한 노트북   
    - **lightning_logs**: 학습 손실 및 검증 손실 로그 기록

- **sktech**: 최종 모델 eva02+convnext2_large_224_in22k (stacking)을 kfold 기법을 활용한 학습 및 추론 코드

    - **sketch_predictor**: streamlit을 활용한 프로토타입 코드

> 직접 실험을 위해선, clone 이후 dataset을 root 폴더에, eva02convnext2_large_mixupcutmix_kfold checkpoints 폴더를 sketch 폴더에 위치시켜야 합니다. (경로 문제)


