# model.py 전체 교체본
from typing import Optional
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

class Classifier:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.model: Optional[models.Model] = None
        self.history = None

    def build(self, input_dim: int, num_classes: int) -> models.Model:
        m = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dense(1024, activation="relu"),
            layers.BatchNormalization(), layers.Dropout(0.2),
            layers.Dense(1024, activation="relu"),
            layers.BatchNormalization(), layers.Dropout(0.1),
            layers.Dense(1024, activation="relu"),
            layers.BatchNormalization(), layers.Dropout(0.2),
            layers.Dense(1024, activation="relu"),
            layers.BatchNormalization(), layers.Dropout(0.1),
            layers.Dense(1024, activation="relu"),
            layers.BatchNormalization(), layers.Dropout(0.2),
            layers.Dense(1024, activation="relu"),
            layers.BatchNormalization(),
            layers.Dense(num_classes, activation="softmax"),
        ])
        m.compile(
            optimizer=optimizers.Adam(learning_rate=self.cfg.lr, clipnorm=self.cfg.clipnorm),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.model = m
        return m

    @staticmethod
    def plot_training(history) -> None:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"], label="Train")
        plt.plot(history.history["val_accuracy"], label="Val")
        plt.xlabel("Epochs"); plt.ylabel("Accuracy"); plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"], label="Train")
        plt.plot(history.history["val_loss"], label="Val")
        plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.legend()
        plt.show()

    def train(self, X_train, y_train, X_val, y_val, resume_ckpt: Optional[str] = None) -> models.Model:
        np.random.seed(self.cfg.seed)
        tf.random.set_seed(self.cfg.seed)
        K.clear_session()
        tf.config.optimizer.set_jit(False)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        num_classes = len(np.unique(y_train))
        model = self.build(X_train.shape[1], num_classes)

        # (추가) 체크포인트 폴더 보장
        ckpt_dir = os.path.dirname(self.cfg.best_model_path)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)

        if resume_ckpt and os.path.isfile(resume_ckpt):
            model.load_weights(resume_ckpt)
            if self.cfg.verbose:
                print(f"[RESUME] Loaded weights from {resume_ckpt}")

        cbs = [
            ModelCheckpoint(self.cfg.best_model_path, save_best_only=True, verbose=self.cfg.verbose),
            EarlyStopping(monitor="val_loss", patience=self.cfg.patience_es, restore_best_weights=True, verbose=self.cfg.verbose),
            ReduceLROnPlateau(monitor="val_loss", factor=self.cfg.rlr_factor, patience=self.cfg.patience_rlr, min_lr=self.cfg.min_lr, verbose=self.cfg.verbose),
        ]

        print(tf.__version__, tf.config.list_physical_devices("GPU"))
        self.history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.cfg.epochs, batch_size=self.cfg.batch_size,
            callbacks=cbs, verbose=self.cfg.verbose
        )

        if self.cfg.plot_training:
            self.plot_training(self.history)
        return model

    def evaluate(self, X, y, title: str = "Eval"):
        assert self.model is not None
        loss, acc = self.model.evaluate(X, y, verbose=0)
        print(f"[{title}] loss={loss:.4f} acc={acc:.4f}")
        return loss, acc

    def predict(self, X):
        assert self.model is not None
        return np.argmax(self.model.predict(X, verbose=0), axis=1)

