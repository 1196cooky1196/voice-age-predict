# 🎙️ voice-age-predict (Common Voice)  
Fundamental ML project: **voice feature engineering + NN classifier** to predict **(Gender / Age / Gender+Age)** labels from speech.

- Dataset: Mozilla **Common Voice** (Kaggle mirror)  
- Core idea: **hand-crafted acoustic features** (MFCC / spectral / energy / F0 stats) → **MLP classifier**  
- Outputs: trained model (`best_model.keras`) + top-k prediction on a single audio

---

## 📌 What this project does
1) Build a feature table from raw audio files (Common Voice clips)  
2) Train a neural network classifier from the feature table  
3) Predict top-k labels for a given audio file (supports `.m4a` via ffmpeg convert)

> Codebase is organized as `preprocess.py` (feature extraction), `train.py` (training pipeline), `model.py` (MLP), `test.py` (inference).  
> (Reference: `preprocess.py`, `train.py`, `model.py`, `test.py`) 

---

## 🧱 Execution Pipeline (End-to-End)

```mermaid
flowchart TD
    A[Common Voice Audio Clips<br/>(.wav / .mp3 / .m4a ...)] --> B[Preprocess: Feature Extraction<br/>librosa + stats]
    B --> C[Feature Table<br/>CSV / XLSX<br/>[filename, gender, features..., label]]
    C --> D[Train: Stratified Split<br/>train / val / test]
    D --> E[Normalize Features<br/>StandardScaler<br/>(keep gender raw)]
    E --> F[MLP Classifier (Keras)<br/>Dense x N + BN + Dropout]
    F --> G[Best Model Checkpoint<br/>best_model.keras]
    G --> H[Test/Inference: Single Audio]
    H --> I[Optional: ffmpeg convert to WAV<br/>(for .m4a etc)]
    I --> J[Extract Features + gender_hint]
    J --> K[Scale with train-fitted scaler<br/>(keep gender raw)]
    K --> L[Predict Top-k Labels<br/>softmax probabilities]

    C -.-> M[Optional: Permutation Importance<br/>feature impact report]
'''

✅ Pipeline Notes (그림 설명)

Preprocess 단계: librosa로 음성을 로드한 뒤, 스펙트럼/에너지/F0/MFCC 기반 통계 특징을 뽑아 한 줄(feature vector) 로 만든 다음 CSV/XLSX 테이블로 저장한다. 


Train 단계: feature table을 로드→컬럼 정규화→라벨 기준 stratified split→StandardScaler로 스케일링하되 gender(첫 열)는 원값 유지→MLP 학습→최고 성능 모델을 best_model.keras로 저장한다.

Test 단계: 단일 오디오 입력을 같은 방식으로 특징 추출하고, 학습에서 만든 스케일 규칙을 적용한 뒤 softmax 확률 top-k를 출력한다. .m4a 등은 ffmpeg로 임시 WAV 변환을 지원한다. 


Feature Importance(선택): 검증셋에서 컬럼을 하나씩 셔플해 정확도 하락(Δacc)을 측정하는 Permutation Importance로 중요한 특징을 뽑을 수 있다.

🧠 Model Architecture (MLP Classifier)

Input is a 111-D vector = [gender_code(1)] + [acoustic_features(110)]

acoustic_features = 3(spectral) + 25×2(MFCC mean/std) + 25×2(ΔMFCC mean/std) + 4(RMS stats) + 3(F0 stats) = 110

```mermaid
flowchart LR
    X[Input Vector<br/>111 dims<br/>(gender + features)] --> BN0[BatchNorm]

    BN0 --> D1[Dense 1024 + ReLU] --> BN1[BatchNorm] --> DP1[Dropout 0.2]
    DP1 --> D2[Dense 1024 + ReLU] --> BN2[BatchNorm] --> DP2[Dropout 0.1]
    DP2 --> D3[Dense 1024 + ReLU] --> BN3[BatchNorm] --> DP3[Dropout 0.2]
    DP3 --> D4[Dense 1024 + ReLU] --> BN4[BatchNorm] --> DP4[Dropout 0.1]
    DP4 --> D5[Dense 1024 + ReLU] --> BN5[BatchNorm] --> DP5[Dropout 0.2]
    DP5 --> D6[Dense 1024 + ReLU] --> BN6[BatchNorm]

    BN6 --> OUT[Dense = num_classes<br/>Softmax]
'''

✅ Model Notes (그림 설명)

이 모델은 CNN/RNN 없이 “특징공학 + MLP”로 끝내는 구조다.

입력은 [gender_code] + [음향 통계 특징]이고, 여러 층의 Dense(1024) + BN + Dropout을 반복해 비선형 결합을 학습한다.

출력은 softmax(num_classes)이며, 클래스 수는 학습 라벨(예: Female_twentieth, Male_thirties 등)의 유니크 개수로 자동 결정된다.
