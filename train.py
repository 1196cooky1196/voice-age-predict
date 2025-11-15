'''
# train.py
# Excel(xlsx/xls)도 지원. 필요하면: pip install openpyxl

import os
from typing import Optional
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from model import Classifier


class DataTableLoader:
    """CSV/TSV/Excel(xlsx/xls)을 자동 감지해서 DataFrame으로 로드"""
    def load(self, path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".xlsx", ".xls"):
            # 엑셀: 첫 번째 시트 로드 (필요 시 sheet_name 인자 처리로 확장)
            return pd.read_excel(path)
        else:
            # 텍스트: 구분자/인코딩 자동 추정
            last_err = None
            for enc in ("utf-8", "cp949", "euc-kr", "latin1"):
                try:
                    return pd.read_csv(path, sep=None, engine="python", encoding=enc)
                except UnicodeDecodeError as e:
                    last_err = e
                    continue
                except Exception:
                    # 파일 없음 등 기타 예외는 그대로 전파
                    raise
            raise RuntimeError(f"텍스트 파일 인코딩을 추정할 수 없습니다: {path}\n{last_err}")


class FeatureTableNormalizer:
    """
    우리 파이프라인이 기대하는 컬럼 구조로 정리:
    [filename, gender] + (수치 features...) + [label]
    """
    _GENDER_MAP = {"male": -1, "female": +1, "other": 0, "남": -1, "여": +1}

    @staticmethod
    def _findcol(df: pd.DataFrame, name_lower: str) -> Optional[str]:
        for c in df.columns:
            if str(c).strip().lower() == name_lower:
                return c
        return None

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # 완전히 빈 컬럼/Unnamed 제거, 공백 제거
        df = df.dropna(axis=1, how="all")
        df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
        df.columns = [str(c).strip() for c in df.columns]

        # 필수 컬럼 탐색(대소문자/공백 무시)
        filename_col = self._findcol(df, "filename")
        gender_col   = self._findcol(df, "gender")
        label_col    = self._findcol(df, "label")

        if filename_col is None:
            raise ValueError("'filename' 컬럼이 없습니다.")
        if gender_col is None:
            raise ValueError("'gender' 컬럼이 없습니다. {-1,0,+1} 또는 male/female/other")
        if label_col is None:
            # 혹시 대소문자/공백 변종이 있을 수 있으니 한 번 더 보정
            alts = [c for c in df.columns if str(c).strip().lower() == "label"]
            if alts:
                df = df.rename(columns={alts[0]: "label"})
                label_col = "label"
            else:
                raise ValueError("'label' 컬럼이 없습니다.")

        # gender 문자열 → 코드화
        if df[gender_col].dtype == object:
            df[gender_col] = df[gender_col].map(
                lambda x: self._GENDER_MAP.get(str(x).strip().lower(), np.nan)
            )
            if df[gender_col].isna().any():
                bad = df.loc[df[gender_col].isna(), gender_col].unique()[:5]
                raise ValueError(f"gender 매핑 실패 값 존재: {bad}  허용: {list(self._GENDER_MAP.keys())}")
        # float로 캐스팅(스케일러와 호환)
        df[gender_col] = df[gender_col].astype(float)

        # 특징 컬럼들: 숫자화
        middle_cols = [c for c in df.columns if c not in [filename_col, gender_col, label_col]]
        for c in middle_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df[middle_cols] = df[middle_cols].fillna(0.0)

        # 최종 순서로 재정렬
        ordered = [filename_col, gender_col] + middle_cols + [label_col]
        return df[ordered]


class DatasetBuilder:
    """LabelEncoder/StandardScaler 수명 주기 관리 (train에만 fit)"""
    def __init__(self) -> None:
        self.encoder: Optional[LabelEncoder] = None
        self.scaler: Optional[StandardScaler] = None

    @staticmethod
    def _df_to_X(df: pd.DataFrame) -> np.ndarray:
        return np.array(df.iloc[:, 1:-1], dtype=float)

    def fit_label_encoder(self, df_train: pd.DataFrame) -> np.ndarray:
        self.encoder = LabelEncoder()
        return self.encoder.fit_transform(df_train.iloc[:, -1])

    def transform_labels(self, df: pd.DataFrame) -> np.ndarray:
        assert self.encoder is not None, "Call fit_label_encoder() first."
        return self.encoder.transform(df.iloc[:, -1])

    def fit_scaler_on_train(self, df_train: pd.DataFrame) -> np.ndarray:
        X_raw = self._df_to_X(df_train)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_raw)
        # 첫 열(gender 코딩)은 원값 유지
        X_scaled[:, 0] = X_raw[:, 0]
        return np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")

    def transform_with_scaler(self, df: pd.DataFrame) -> np.ndarray:
        assert self.scaler is not None, "Call fit_scaler_on_train() first."
        X_raw = self._df_to_X(df)
        X_scaled = self.scaler.transform(X_raw)
        X_scaled[:, 0] = X_raw[:, 0]
        return np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


class Trainer:
    """엔드투엔드 학습 파이프라인(로드→정규화→split→fit/transform→train/eval/report)"""
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.loader = DataTableLoader()
        self.normalizer = FeatureTableNormalizer()
        self.dataset = DatasetBuilder()
        self.clf = Classifier(cfg)

    def train_from_features(
        self,
        feature_csv: str,
        test_size: float = 0.4,
        val_size_from_test: float = 0.5,
        resume_ckpt: Optional[str] = None,
    ) -> None:
        # 1) 파일 로딩(Excel/CSV 자동)
        df = self.loader.load(feature_csv)
        # 2) 컬럼 구조 정규화
        df = self.normalizer.normalize(df)

        # 3) Stratified split by label BEFORE fitting enc/scale
        y_all = df.iloc[:, -1]
        df_train, df_tmp, _, y_tmp = train_test_split(
            df, y_all, test_size=test_size, random_state=self.cfg.seed, stratify=y_all
        )
        y_tmp2 = df_tmp.iloc[:, -1]
        df_test, df_val, _, _ = train_test_split(
            df_tmp, y_tmp2, test_size=val_size_from_test, random_state=self.cfg.seed, stratify=y_tmp2
        )

        # 4) Fit ONLY on train
        y_train = self.dataset.fit_label_encoder(df_train)
        X_train = self.dataset.fit_scaler_on_train(df_train)

        # 5) Transform val/test
        y_val = self.dataset.transform_labels(df_val)
        X_val = self.dataset.transform_with_scaler(df_val)
        y_test = self.dataset.transform_labels(df_test)
        X_test = self.dataset.transform_with_scaler(df_test)

        # 6) Train / Eval
        self.clf.train(X_train, y_train, X_val, y_val, resume_ckpt=resume_ckpt)
        self.clf.evaluate(X_val, y_val, title="Validation")
        self.clf.evaluate(X_test, y_test, title="Test")

        # 7) Report
        y_pred = self.clf.predict(X_test)
        print("\n[Confusion Matrix]\n", confusion_matrix(y_test, y_pred))
        classes = list(self.dataset.encoder.classes_) if self.dataset.encoder is not None else None
        if classes:
            print("\n[Classification Report]\n", classification_report(y_test, y_pred, target_names=classes))

'''

# === train.py 변경본 시작 ===
from typing import Optional, List
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from model import Classifier


class DataTableLoader:
    """CSV/TSV/Excel(xlsx/xls)을 자동 감지해서 DataFrame으로 로드"""
    def load(self, path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".xlsx", ".xls"):
            # 엑셀: 첫 시트 로드
            return pd.read_excel(path)
        else:
            # 텍스트: 구분자/인코딩 자동 추정
            last_err = None
            for enc in ("utf-8", "cp949", "euc-kr", "latin1"):
                try:
                    return pd.read_csv(path, sep=None, engine="python", encoding=enc)
                except UnicodeDecodeError as e:
                    last_err = e
                    continue
                except Exception:
                    raise
            raise RuntimeError(f"텍스트 파일 인코딩을 추정할 수 없습니다: {path}\n{last_err}")


class FeatureTableNormalizer:
    """
    파이프라인이 기대하는 컬럼 구조로 정리:
    [filename, gender] + (수치 features...) + [label]
    """
    _GENDER_MAP = {"male": -1, "female": +1, "other": 0, "남": -1, "여": +1}

    @staticmethod
    def _findcol(df: pd.DataFrame, name_lower: str) -> Optional[str]:
        for c in df.columns:
            if str(c).strip().lower() == name_lower:
                return c
        return None

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # 완전히 빈 컬럼/Unnamed 제거, 공백 제거
        df = df.dropna(axis=1, how="all")
        df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
        df.columns = [str(c).strip() for c in df.columns]

        # 필수 컬럼 탐색(대소문자/공백 무시)
        filename_col = self._findcol(df, "filename")
        gender_col   = self._findcol(df, "gender")
        label_col    = self._findcol(df, "label")

        if filename_col is None:
            raise ValueError("'filename' 컬럼이 없습니다.")
        if gender_col is None:
            raise ValueError("'gender' 컬럼이 없습니다. {-1,0,+1} 또는 male/female/other")
        if label_col is None:
            alts = [c for c in df.columns if str(c).strip().lower() == "label"]
            if alts:
                df = df.rename(columns={alts[0]: "label"})
                label_col = "label"
            else:
                raise ValueError("'label' 컬럼이 없습니다.")

        # gender 문자열 → 코드화
        if df[gender_col].dtype == object:
            df[gender_col] = df[gender_col].map(
                lambda x: self._GENDER_MAP.get(str(x).strip().lower(), np.nan)
            )
            if df[gender_col].isna().any():
                bad = df.loc[df[gender_col].isna(), gender_col].unique()[:5]
                raise ValueError(f"gender 매핑 실패 값 존재: {bad}  허용: {list(self._GENDER_MAP.keys())}")
        df[gender_col] = df[gender_col].astype(float)

        # 특징 컬럼들: 숫자로 강제 변환
        middle_cols = [c for c in df.columns if c not in [filename_col, gender_col, label_col]]
        for c in middle_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df[middle_cols] = df[middle_cols].fillna(0.0)

        # 최종 순서로 재정렬
        ordered = [filename_col, gender_col] + middle_cols + [label_col]
        return df[ordered]


class DatasetBuilder:
    """LabelEncoder/StandardScaler 수명 주기 관리 (train에만 fit)"""
    def __init__(self) -> None:
        self.encoder: Optional[LabelEncoder] = None
        self.scaler: Optional[StandardScaler] = None

    @staticmethod
    def _df_to_X(df: pd.DataFrame) -> np.ndarray:
        return np.array(df.iloc[:, 1:-1], dtype=float)

    def fit_label_encoder(self, df_train: pd.DataFrame) -> np.ndarray:
        self.encoder = LabelEncoder()
        return self.encoder.fit_transform(df_train.iloc[:, -1])

    def transform_labels(self, df: pd.DataFrame) -> np.ndarray:
        assert self.encoder is not None, "Call fit_label_encoder() first."
        return self.encoder.transform(df.iloc[:, -1])

    def fit_scaler_on_train(self, df_train: pd.DataFrame) -> np.ndarray:
        X_raw = self._df_to_X(df_train)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_raw)
        # 첫 열(gender 코딩)은 원값 유지
        X_scaled[:, 0] = X_raw[:, 0]
        return np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")

    def transform_with_scaler(self, df: pd.DataFrame) -> np.ndarray:
        assert self.scaler is not None, "Call fit_scaler_on_train() first."
        X_raw = self._df_to_X(df)
        X_scaled = self.scaler.transform(X_raw)
        X_scaled[:, 0] = X_raw[:, 0]
        return np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


class FeatureImportanceAnalyzer:
    """
    퍼뮤테이션 중요도 기반 특성 영향도 분석기.
    - 기준 정확도에서, 특정 컬럼을 셔플했을 때 정확도 하락량(Δacc)의 평균/표준편차를 측정.
    - 장점: 모델 불문(Model-agnostic), 스케일 영향 적음, 해석 직관적.
    """
    def __init__(self, clf: Classifier, feature_names: List[str], verbose: int = 1, random_state: int = 42) -> None:
        self.clf = clf
        self.feature_names = list(feature_names)
        self.verbose = verbose
        self.rng = np.random.default_rng(random_state)

    def _score(self, X: np.ndarray, y: np.ndarray) -> float:
        loss, acc = self.clf.model.evaluate(X, y, verbose=0)
        return float(acc)

    def permutation_importance(self, X_val: np.ndarray, y_val: np.ndarray, repeats: int = 5) -> pd.DataFrame:
        assert self.clf.model is not None, "Model is not trained."
        baseline = self._score(X_val, y_val)

        n_feat = X_val.shape[1]
        drops_mean = np.zeros(n_feat, dtype=float)
        drops_std  = np.zeros(n_feat, dtype=float)

        for j in range(n_feat):
            scores = []
            for _ in range(repeats):
                Xp = X_val.copy()
                idx = self.rng.permutation(Xp.shape[0])
                Xp[:, j] = Xp[idx, j]  # 컬럼 j만 셔플
                acc = self._score(Xp, y_val)
                scores.append(baseline - acc)
            drops_mean[j] = float(np.mean(scores))
            drops_std[j]  = float(np.std(scores))

            if self.verbose and (j % 20 == 0 or j == n_feat - 1):
                print(f"[IMP] {j+1}/{n_feat} done", flush=True)

        df_imp = pd.DataFrame({
            "feature": self.feature_names,
            "drop_mean": drops_mean,
            "drop_std": drops_std,
            "baseline_acc": baseline,
            "shuffled_acc_est": baseline - drops_mean,
        }).sort_values("drop_mean", ascending=False).reset_index(drop=True)
        return df_imp

    def explain(self, feature: str) -> str:
        f = feature.lower()
        if f == "gender":
            return "성별 코드 자체(−1/0/+1). 라벨이 Gender_Age면 직접적 힌트라 영향도가 매우 큼."
        if f.startswith("f0_"):
            return "기본 주파수(F0) 통계. 남성/여성의 평균 피치 차이와 안정성(진동/jitter)이 성별 분류에 강력."
        if "centroid" in f:
            return "스펙트럴 중심(밝기). 고주파 비중이 높을수록 값↑. 성대 길이/포만트 차이와 연계."
        if "rolloff" in f:
            return "에너지 대부분이 누적되는 상한 주파수. 고주파 에너지의 꼬리 길이를 반영."
        if "bandwidth" in f:
            return "스펙트럼 퍼짐 정도. 잡음/자음 성분과 관련."
        if f.startswith("mfcc1_") or f.startswith("mfcc2_"):
            return "낮은 차수 MFCC는 스펙트럼 기울기/포만트 대략을 잡음. 음색·연령대 차이에 기여."
        if f.startswith("mfcc"):
            return "MFCC 계수 통계. 조음·공명 패턴을 요약하여 연령/성별에 유의미."
        if f.startswith("delta_mfcc"):
            return "ΔMFCC: 시간 변화율. 발화 다이내믹/과도구간 특성이 반영."
        if f.startswith("rms_"):
            return "RMS 에너지 통계. 발화 세기 수준·변동·왜도/첨도가 프로소디 차이를 포착."
        return "음향 통계 특징. 스펙트럼/에너지/시간 변화가 발화 특성 차이를 반영."

    def annotate_topk(self, df_imp: pd.DataFrame, k: int = 20) -> pd.DataFrame:
        top = df_imp.head(k).copy()
        top["why"] = top["feature"].map(self.explain)
        return top


class Trainer:
    """엔드투엔드 학습 + 중요도 분석 오케스트레이션"""
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.loader = DataTableLoader()
        self.normalizer = FeatureTableNormalizer()
        self.dataset = DatasetBuilder()
        self.clf = Classifier(cfg)

        # 분석용 캐시
        self._feature_names: Optional[List[str]] = None
        self._X_val: Optional[np.ndarray] = None
        self._y_val: Optional[np.ndarray] = None

    def train_from_features(
        self,
        feature_csv: str,
        test_size: float = 0.4,
        val_size_from_test: float = 0.5,
        resume_ckpt: Optional[str] = None,
    ) -> None:
        # 1) 파일 로딩(Excel/CSV 자동)
        df = self.loader.load(feature_csv)
        # 2) 컬럼 구조 정규화
        df = self.normalizer.normalize(df)

        # 3) stratified split
        y_all = df.iloc[:, -1]
        df_train, df_tmp, _, y_tmp = train_test_split(
            df, y_all, test_size=test_size, random_state=self.cfg.seed, stratify=y_all
        )
        y_tmp2 = df_tmp.iloc[:, -1]
        df_test, df_val, _, _ = train_test_split(
            df_tmp, y_tmp2, test_size=val_size_from_test, random_state=self.cfg.seed, stratify=y_tmp2
        )

        # 4) Fit ONLY on train
        y_train = self.dataset.fit_label_encoder(df_train)
        X_train = self.dataset.fit_scaler_on_train(df_train)

        # 5) Transform val/test
        y_val = self.dataset.transform_labels(df_val)
        X_val = self.dataset.transform_with_scaler(df_val)
        y_test = self.dataset.transform_labels(df_test)
        X_test = self.dataset.transform_with_scaler(df_test)

        # --- 분석용 캐시 저장 ---
        self._feature_names = list(df.columns[1:-1])  # filename/gender 제외 → [gender 포함한 첫 수치열부터, label 전까지]
        self._X_val = X_val
        self._y_val = y_val

        # 6) Train / Eval
        self.clf.train(X_train, y_train, X_val, y_val, resume_ckpt=resume_ckpt)
        self.clf.evaluate(X_val, y_val, title="Validation")
        self.clf.evaluate(X_test, y_test, title="Test")

        # 7) 보고
        y_pred = self.clf.predict(X_test)
        print("\n[Confusion Matrix]\n", confusion_matrix(y_test, y_pred))
        classes = list(self.dataset.encoder.classes_) if self.dataset.encoder is not None else None
        if classes:
            print("\n[Classification Report]\n", classification_report(y_test, y_pred, target_names=classes))

    def report_feature_importance(
        self,
        top_k: int = 20,
        repeats: int = 5,
        exclude_gender: bool = False,
    ) -> pd.DataFrame:
        """
        학습이 끝난 후, 검증셋 기준 퍼뮤테이션 중요도 리포트 출력.
        - exclude_gender=True면 'gender' 컬럼을 제외하고 순수 음향 피처의 영향을 본다.
        - 반환: 중요도 DataFrame(상위 k개, 자동 설명 포함)
        """
        assert self._X_val is not None and self._y_val is not None and self._feature_names is not None, \
            "먼저 train_from_features를 실행하세요."

        # (선택) gender 제외
        X_val = self._X_val
        feature_names = self._feature_names
        if exclude_gender and feature_names and feature_names[0].lower() == "gender":
            X_val = X_val[:, 1:]
            feature_names = feature_names[1:]

        analyzer = FeatureImportanceAnalyzer(self.clf, feature_names, verbose=self.cfg.verbose)
        df_imp = analyzer.permutation_importance(X_val, self._y_val, repeats=repeats)
        top = analyzer.annotate_topk(df_imp, k=top_k)

        # 콘솔 출력(상위 k개)
        print("\n[Permutation Importance: top-{}]".format(top_k))
        for _, r in top.iterrows():
            print(f"- {r['feature']:>22s} | Δacc={r['drop_mean']:.4f} (±{r['drop_std']:.4f}) | {r['why']}")

        return top
# === train.py 변경본 끝 ===
