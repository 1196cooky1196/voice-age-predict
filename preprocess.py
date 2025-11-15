# preprocess.py
import os, csv, time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import librosa
from scipy.stats import skew, kurtosis

import imageio_ffmpeg
from pydub import AudioSegment


# =========================
# Config & common
# =========================
@dataclass
class Config:
    # ----- Data constraints -----
    valid_ages: Tuple[str, ...] = ("teens", "twenties", "thirties", "fourties", "fifties", "sixties")

    # ----- Audio/feature params -----
    sr_resample: Optional[int] = None   # e.g., 16000 to resample; None keeps native SR
    fmin_note: str = "C2"
    fmax_note: str = "C7"
    n_mfcc: int = 25

    # ----- Training -----
    seed: int = 42
    batch_size: int = 1024
    epochs: int = 300
    lr: float = 1e-3
    clipnorm: float = 1.0
    patience_es: int = 5
    patience_rlr: int = 3
    rlr_factor: float = 0.5
    min_lr: float = 1e-6
    best_model_path: str = "best_model.keras"

    # ----- Labeling -----
    label_style: str = "Gender_Age"     # "Gender_Age" | "Gender" | "Age"

    # ----- Plot/verbosity -----
    plot_training: bool = True
    verbose: int = 1

    # ----- Logging for feature build -----
    log_every_n: int = 25     # N개마다 진행 로그 출력
    show_each: bool = False   # True면 매 파일마다 출력


GENDER_MAP: Dict[str, int] = {"male": -1, "female": +1, "other": 0}


def ensure_dir(path: str) -> None:
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


# =========================
# Metadata
# =========================
class Metadata:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def load_and_filter(self, meta_csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(meta_csv_path)
        df = df.dropna(subset=["gender"])
        df = df[df["age"].isin(self.cfg.valid_ages)]
        return df.reset_index(drop=True)

    @staticmethod
    def get_age(df: pd.DataFrame, filename: str) -> str:
        try:
            return df.loc[df["filename"] == filename, "age"].values[0]
        except Exception:
            return "unknown"

    @staticmethod
    def get_gender_code(df: pd.DataFrame, filename: str) -> int:
        try:
            gender = df.loc[df["filename"] == filename, "gender"].values[0]
            return GENDER_MAP.get(gender, GENDER_MAP["other"])
        except Exception:
            return GENDER_MAP["other"]

    @staticmethod
    def gender_prefix_from_code(code: int) -> str:
        return "Female" if code > 0 else "Male" if code < 0 else "Other"


# =========================
# Feature Extractor
# =========================
class FeatureExtractor:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def extract(self, audio_path: str) -> List[float]:
        y, sr = librosa.load(audio_path, sr=self.cfg.sr_resample)
        feats: List[float] = []

        # Spectral
        feats.append(float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))))
        feats.append(float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))))
        feats.append(float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))))

        # MFCC + delta
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.cfg.n_mfcc)
        delta = librosa.feature.delta(mfccs)
        for coeff in mfccs:
            feats += [float(np.mean(coeff)), float(np.std(coeff))]
        for d in delta:
            feats += [float(np.mean(d)), float(np.std(d))]

        # Energy stats
        rms = librosa.feature.rms(y=y)[0]
        feats += [float(np.mean(rms)), float(np.std(rms)), float(skew(rms)), float(kurtosis(rms))]

        # F0 stats
        try:
            f0, _, _ = librosa.pyin(
                y,
                fmin=librosa.note_to_hz(self.cfg.fmin_note),
                fmax=librosa.note_to_hz(self.cfg.fmax_note),
            )
            f0 = f0[~np.isnan(f0)]
            if len(f0) == 0:
                raise ValueError
            feats += [float(np.mean(f0)), float(np.std(f0)), float(np.std(np.diff(f0)))]
        except Exception:
            feats += [0.0, 0.0, 0.0]

        return feats

    @staticmethod
    def header(n_mfcc: int) -> List[str]:
        cols = ["filename", "gender"]
        cols += ["spectral_centroid", "spectral_bandwidth", "spectral_rolloff"]
        cols += [f"mfcc{i+1}_mean" for i in range(n_mfcc)]
        cols += [f"mfcc{i+1}_std" for i in range(n_mfcc)]
        cols += [f"delta_mfcc{i+1}_mean" for i in range(n_mfcc)]
        cols += [f"delta_mfcc{i+1}_std" for i in range(n_mfcc)]
        cols += ["rms_mean", "rms_std", "rms_skew", "rms_kurtosis"]
        cols += ["f0_mean", "f0_std", "f0_jitter"]
        cols += ["label"]
        return cols


# =========================
# Feature CSV Builder (with progress logs)
# =========================
# preprocess.py 중 변경된 클래스 전체
import os, csv, time
import pandas as pd
import numpy as np
import librosa
from scipy.stats import skew, kurtosis
import imageio_ffmpeg

class FeatureCSVBuilder:
    def __init__(self, cfg, meta, extractor) -> None:
        self.cfg = cfg
        self.meta = meta
        self.ext = extractor
        # 지연 import: 옵션 B에서 이 클래스를 실제로 쓸 때만 pydub 로드
        from pydub import AudioSegment
        AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

    def _make_label(self, gender_code: int, age: str) -> str:
        gp = self.meta.gender_prefix_from_code(gender_code)
        if self.cfg.label_style == "Gender_Age": return f"{gp}_{age}"
        if self.cfg.label_style == "Gender":     return gp
        if self.cfg.label_style == "Age":        return age
        return f"{gp}_{age}"

    def build_csv(self, audio_dir: str, df: pd.DataFrame, out_csv_path: str) -> str:
        out_dir = os.path.dirname(out_csv_path)
        if out_dir: os.makedirs(out_dir, exist_ok=True)

        # 헤더
        with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
            import csv as _csv
            _csv.writer(f).writerow(self.ext.header(self.cfg.n_mfcc))

        total = len(df); ok = skipped = err = 0
        t0 = time.time()
        if self.cfg.verbose:
            print(f"[BUILD] Start feature CSV → {out_csv_path}", flush=True)
            print(f"[BUILD] Total rows: {total}", flush=True)

        for i, (_, row) in enumerate(df.iterrows(), start=1):
            filename = row["filename"]
            audio_path = os.path.join(audio_dir, filename)

            if not os.path.isfile(audio_path):
                skipped += 1
                if self.cfg.verbose and (self.cfg.show_each or i % self.cfg.log_every_n == 0):
                    print(f"[SKIP {i}/{total}] not found: {audio_path}", flush=True)
                continue

            try:
                feats = self.ext.extract(audio_path)
                gcode = self.meta.get_gender_code(df, filename)
                age = self.meta.get_age(df, filename)
                label = self._make_label(gcode, age)

                with open(out_csv_path, "a", newline="", encoding="utf-8") as f:
                    import csv as _csv
                    _csv.writer(f).writerow([filename, gcode] + feats + [label])

                ok += 1
                if self.cfg.verbose and (self.cfg.show_each or i % self.cfg.log_every_n == 0):
                    print(f"[OK {i}/{total}] wrote: {filename}", flush=True)
            except Exception as e:
                err += 1
                if self.cfg.verbose:
                    print(f"[ERR {i}/{total}] {filename}: {e}", flush=True)

        if self.cfg.verbose:
            dt = time.time() - t0
            print(f"[DONE] wrote={ok}  skipped={skipped}  error={err}  time={dt:.2f}s → {out_csv_path}", flush=True)

        return out_csv_path

