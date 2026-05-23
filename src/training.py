"""
training.py — Esqueleto de entrenamiento común para el proyecto RNN
Proyecto Final Deep Learning · Maestría en IA · Pontificia Universidad Javeriana · 2026

Autor: Daniel Ruiz
Uso:  from src.training import train, set_seed

La función train() es agnóstica al modelo: recibe cualquier nn.Module
que acepte tensores de índices de tokens y devuelva logits (B, n_classes).
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ──────────────────────────────────────────────────────────────────────────────
# Semilla global
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Fija la semilla en random, numpy y torch (CPU + GPU) para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Para operaciones deterministas en cuDNN (puede reducir un poco la velocidad)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ──────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Detiene el entrenamiento si val_loss no mejora tras `patience` épocas.

    Args:
        patience (int): Número de épocas sin mejora antes de detener.
        min_delta (float): Mejora mínima que cuenta como mejora real.
        checkpoint_path (str | Path): Ruta donde guardar el mejor modelo.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4,
                 checkpoint_path: str | Path = "results/best_model.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        self.best_loss = float("inf")
        self.best_epoch = 0
        self.counter = 0
        self.stopped = False

    def step(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        """
        Evalúa si hay mejora y guarda checkpoint si es necesario.

        Returns:
            True si se debe detener el entrenamiento, False si continúa.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            print(f"    ✓ Checkpoint guardado (val_loss={val_loss:.4f}) → {self.checkpoint_path}")
        else:
            self.counter += 1
            print(f"    Early stopping: {self.counter}/{self.patience} sin mejora.")
            if self.counter >= self.patience:
                self.stopped = True
                return True
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Epoch helpers
# ──────────────────────────────────────────────────────────────────────────────

def _train_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip_grad: float | None = 1.0,
) -> tuple[float, float]:
    """
    Ejecuta una época de entrenamiento.

    Returns:
        (avg_loss, accuracy) sobre el set de entrenamiento.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        # Se asume que el DataLoader devuelve (input_ids, labels).
        # input_ids puede ser un tensor 2-D (B, seq_len).
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs)           # (B, n_classes)
        loss = criterion(logits, labels)
        loss.backward()

        if clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def _eval_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evalúa el modelo en un loader (val o test) sin actualizar gradientes.

    Returns:
        (avg_loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


# ──────────────────────────────────────────────────────────────────────────────
# Función principal: train()
# ──────────────────────────────────────────────────────────────────────────────

def train(
    model: nn.Module,
    train_loader,
    val_loader,
    config: dict,
) -> dict:
    """
    Entrena un modelo de clasificación y devuelve el diccionario de métricas
    en el formato JSON estandarizado del equipo.

    Args:
        model       : nn.Module que recibe (B, seq_len) y devuelve logits (B, C).
        train_loader: DataLoader de entrenamiento.
        val_loader  : DataLoader de validación.
        config      : Diccionario con los parámetros de entrenamiento y metadatos.
                      Claves obligatorias:
                        model_name  (str)
                        owner       (str)
                        track       (str)  "PP" o "TI"
                        n_epochs    (int)
                        lr          (float)
                        patience    (int)   early stopping
                        checkpoint_path (str)
                      Claves opcionales:
                        clip_grad       (float, default 1.0)
                        weight_decay    (float, default 0.0)
                        use_lr_scheduler(bool,  default True)
                        class_weights   (Tensor, default None)
                        embedding_dim   (int)
                        hidden_size     (int)
                        dropout         (float)
                        use_class_weights (bool)
                        seed            (int, default 42)

    Returns:
        metrics_dict: Diccionario con el formato JSON estandarizado del equipo.
                      (sin métricas de test — esas las agrega evaluate())
    """
    # ── Semilla ──────────────────────────────────────────────────────────────
    set_seed(config.get("seed", 42))

    # ── Dispositivo ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    model = model.to(device)

    # ── Conteo de parámetros ─────────────────────────────────────────────────
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros entrenables: {n_params:,}")

    # ── Loss function ─────────────────────────────────────────────────────────
    class_weights = config.get("class_weights", None)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimizador ───────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0.0),
    )

    # ── LR Scheduler (opcional) ───────────────────────────────────────────────
    scheduler = None
    if config.get("use_lr_scheduler", True):
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

    # ── Early Stopping ────────────────────────────────────────────────────────
    early_stopper = EarlyStopping(
        patience=config["patience"],
        checkpoint_path=config["checkpoint_path"],
    )

    # ── Historial ─────────────────────────────────────────────────────────────
    loss_history     = []
    val_loss_history = []
    acc_history      = []
    val_acc_history  = []

    # ── Bucle de entrenamiento ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Entrenando: {config['model_name']}  |  Owner: {config['owner']}")
    print(f"{'='*60}")

    start_time = time.time()

    for epoch in range(1, config["n_epochs"] + 1):
        train_loss, train_acc = _train_epoch(
            model, train_loader, criterion, optimizer, device,
            clip_grad=config.get("clip_grad", 1.0),
        )
        val_loss, val_acc = _eval_epoch(model, val_loader, criterion, device)

        loss_history.append(round(train_loss, 6))
        val_loss_history.append(round(val_loss, 6))
        acc_history.append(round(train_acc, 6))
        val_acc_history.append(round(val_acc, 6))

        print(
            f"Época {epoch:3d}/{config['n_epochs']} | "
            f"loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}"
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        if early_stopper.step(val_loss, model, epoch):
            print(f"\n  Early stopping activado en época {epoch}.")
            break

    elapsed = round(time.time() - start_time, 1)

    print(f"\n  Entrenamiento completado en {elapsed}s | Mejor época: {early_stopper.best_epoch}")

    # ── Construir metrics_dict (sin métricas de test todavía) ─────────────────
    metrics_dict = {
        "model_name": config["model_name"],
        "owner":      config["owner"],
        "track":      config["track"],
        "config": {
            "embedding_dim":     config.get("embedding_dim",     None),
            "hidden_size":       config.get("hidden_size",       None),
            "dropout":           config.get("dropout",           None),
            "use_class_weights": config.get("use_class_weights", False),
            "n_params":          n_params,
        },
        "metrics": {
            # Se llenan con evaluate() sobre el test set al final
            "accuracy":          None,
            "precision_macro":   None,
            "recall_macro":      None,
            "f1_macro":          None,
            "f1_per_class":      None,
            "confusion_matrix":  None,
        },
        "training": {
            "epochs_run":              len(loss_history),
            "best_epoch":              early_stopper.best_epoch,
            "training_time_seconds":   elapsed,
            "loss_history":            loss_history,
            "val_loss_history":        val_loss_history,
            "acc_history":             acc_history,
            "val_acc_history":         val_acc_history,
        },
    }

    return metrics_dict


# ──────────────────────────────────────────────────────────────────────────────
# Función de evaluación sobre test set
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    model: nn.Module,
    test_loader,
    checkpoint_path: str | Path,
    config: dict,
) -> tuple[list[int], list[int]]:
    """
    Carga el mejor checkpoint y devuelve las predicciones sobre test_loader.

    IMPORTANTE: llamar esta función UNA SOLA VEZ por modelo, al final.
    Nunca mirar el test set durante el tuning de hiperparámetros.

    Args:
        model           : nn.Module con la misma arquitectura con la que se entrenó.
        test_loader     : DataLoader del test set.
        checkpoint_path : Ruta al archivo .pt guardado por EarlyStopping.
        config          : Diccionario de configuración (usado para el dispositivo).

    Returns:
        (y_true, y_pred): Listas de enteros con las etiquetas reales y predichas.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            logits = model(inputs)
            preds = logits.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())

    return y_true, y_pred
