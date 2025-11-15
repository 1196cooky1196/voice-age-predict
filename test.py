import time
start_time = time.time()

# test.py
# - 기존 모듈 재사용: preprocess.Config/FeatureExtractor, train.DataTableLoader/FeatureTableNormalizer/DatasetBuilder
# - main 안에서 직접 age_gender_show() 호출
# - CLI 지원: python test.py <audio> [gender_hint] [model_path] [feature_table]

# test.py
# - 기존 모듈 재사용: preprocess.Config/FeatureExtractor, train.DataTableLoader/FeatureTableNormalizer/DatasetBuilder
# - .m4a 등 FFmpeg 의존 포맷은 실행 중 WAV로 임시 변환 후 특징추출
# - main 안에서 직접 호출: python test.py <audio> [gender_hint] [model_path] [feature_table]

# test.py
# - 기존 모듈 재사용: preprocess.Config/FeatureExtractor, train.DataTableLoader/FeatureTableNormalizer/DatasetBuilder
# - .m4a 등은 ffmpeg(subprocess)로 임시 WAV 변환 후 특징추출
# - 실행: python test.py <audio> [gender_hint] [model_path] [feature_table]

import os
import sys
import tempfile
import subprocess
import numpy as np
import tensorflow as tf

from preprocess import Config, FeatureExtractor
from train import DataTableLoader, FeatureTableNormalizer, DatasetBuilder

import imageio_ffmpeg

_KO = {"Female": "여성", "Male": "남성", "Other": "기타"}

def _decode_label(label: str):
    if "_" in label:
        g, age = label.split("_", 1)
    else:
        g, age = label, ""
    return _KO.get(g, g), age

def _ffmpeg_path() -> str:
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"  # PATH에 있는 ffmpeg 사용 시

def _convert_to_wav_ffmpeg(src_path: str, target_sr: int = 16000) -> str:
    """ffmpeg로 16kHz/mono/PCM WAV 임시 파일 생성 (ffprobe 불필요)"""
    ff = _ffmpeg_path()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out_wav = tmp.name
    tmp.close()
    cmd = [
        ff, "-y", "-hide_banner", "-loglevel", "error",
        "-i", src_path,
        "-ar", str(target_sr), "-ac", "1",
        "-f", "wav", out_wav,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not os.path.isfile(out_wav):
        msg = proc.stderr.decode(errors="ignore")
        raise RuntimeError(f"ffmpeg 변환 실패: {msg[:500]}")
    return out_wav

def age_gender_show(
    audio_path: str,
    *,
    model_path: str,
    feature_table: str,
    gender_hint: float = 0.0,
    topk: int = 3,
):
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"오디오 파일 없음: {audio_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"모델 파일 없음: {model_path}")
    if not os.path.isfile(feature_table):
        raise FileNotFoundError(f"특징 테이블 없음: {feature_table}")

    # 1) 학습 테이블 기반 라벨/스케일러 세팅(재학습 아님)
    loader = DataTableLoader()
    normalizer = FeatureTableNormalizer()
    ds = DatasetBuilder()
    df = normalizer.normalize(loader.load(feature_table))
    _ = ds.fit_label_encoder(df)
    _ = ds.fit_scaler_on_train(df)

    # 2) 특징 추출 (librosa 실패 시 ffmpeg로 변환 후 재시도)
    cfg = Config()
    ext = FeatureExtractor(cfg)

    tmp_wav = None
    try:
        try:
            feats = ext.extract(audio_path)  # 110차원
        except Exception:
            # librosa가 직접 못 읽는 포맷(.m4a 등): ffmpeg 변환
            sr = cfg.sr_resample if cfg.sr_resample else 16000
            tmp_wav = _convert_to_wav_ffmpeg(audio_path, target_sr=sr)
            feats = ext.extract(tmp_wav)

        x_raw = np.array([[gender_hint] + feats], dtype="float32")  # (1, D)

        # 3) 스케일링(첫 열 gender는 원값 유지 규칙 적용)
        if getattr(ds.scaler, "n_features_in_", None) and ds.scaler.n_features_in_ != x_raw.shape[1]:
            raise RuntimeError(
                f"입력 차원({x_raw.shape[1]}) ≠ 스케일러 기대치({ds.scaler.n_features_in_}). "
                "학습/추론 특징 정의 확인."
            )
        x = ds.scaler.transform(x_raw)
        x[:, 0] = x_raw[:, 0]

        # 4) 예측
        model = tf.keras.models.load_model(model_path)
        prob = model.predict(x, verbose=0)[0]
        idx = np.argsort(prob)[::-1][:topk]
        classes = ds.encoder.classes_
        tops = [(str(classes[i]), float(prob[i])) for i in idx]

        # 5) 출력
        print("가수의 상위 3개의 예측 결과:")
        for rank, (lab, p) in enumerate(tops, start=1):
            gko, age = _decode_label(lab)
            print(f"{rank}. 성별: {gko}, 나이: {age}, 확률: {p*100:.2f}%")
        return tops

    finally:
        if tmp_wav and os.path.isfile(tmp_wav):
            try:
                os.remove(tmp_wav)
            except Exception:
                pass

if __name__ == "__main__":
    # ===== 여기만 수정해서 컨트롤 =====
    AUDIO_PATH    = "maybelove.m4a"   # 새 오디오 경로(.m4a 포함 가능)
    MODEL_PATH    = "checkpoints/best_model.keras"
    FEATURE_TABLE = "train_age.csv.xlsx"
    GENDER_HINT   = 0.0                             # 모르면 0.0, 남성:-1, 여성:+1
    TOPK          = 3

    # CLI: python test.py <audio> [gender_hint] [model_path] [feature_table]
    if len(sys.argv) >= 2:
        AUDIO_PATH = sys.argv[1]
    if len(sys.argv) >= 3:
        GENDER_HINT = float(sys.argv[2])
    if len(sys.argv) >= 4:
        MODEL_PATH = sys.argv[3]
    if len(sys.argv) >= 5:
        FEATURE_TABLE = sys.argv[4]

    age_gender_show(
        AUDIO_PATH,
        model_path=MODEL_PATH,
        feature_table=FEATURE_TABLE,
        gender_hint=GENDER_HINT,
        topk=TOPK,
    )



print(f"total time is {time.time() - start_time}")