"""
Module de Transfer Learning avec dataset supplementaire.

Ce module implemente un entrainement en deux phases:
1. Pre-entrainement sur STL-10 (dataset supplementaire avec images 96x96)
2. Fine-tuning sur CIFAR-10 pour ameliorer les performances
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
import torchvision.transforms as transforms
from typing import Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from pathlib import Path
import time
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.models.architecture import create_model, CIFAR10ResNet
from src.utils.config import (
    NUM_CLASSES,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    DATA_DIR,
    RANDOM_SEED,
    CIFAR10_MEAN,
    CIFAR10_STD
)


# Mapping des classes STL-10 vers CIFAR-10
# STL-10: airplane(0), bird(1), car(2), cat(3), deer(4), dog(5), horse(6), monkey(7), ship(8), truck(9)
# CIFAR-10: airplane(0), automobile(1), bird(2), cat(3), deer(4), dog(5), frog(6), horse(7), ship(8), truck(9)
STL10_TO_CIFAR10 = {
    0: 0,  # airplane -> airplane
    1: 2,  # bird -> bird
    2: 1,  # car -> automobile
    3: 3,  # cat -> cat
    4: 4,  # deer -> deer
    5: 5,  # dog -> dog
    6: 7,  # horse -> horse
    8: 8,  # ship -> ship
    9: 9,  # truck -> truck
}

# Classes valides de STL-10 (excluant monkey=7)
VALID_STL10_CLASSES = [0, 1, 2, 3, 4, 5, 6, 8, 9]


class MixUp:
    """Implementation de MixUp pour l'augmentation des donnees."""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(
        self,
        images: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Applique MixUp sur un batch.

        Returns:
            (images mixees, targets1, targets2, lambda)
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = images.size(0)
        index = torch.randperm(batch_size)

        mixed_images = lam * images + (1 - lam) * images[index, :]
        targets_a, targets_b = targets, targets[index]

        return mixed_images, targets_a, targets_b, lam


class CutMix:
    """Implementation de CutMix pour l'augmentation des donnees."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(
        self,
        images: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Applique CutMix sur un batch."""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = images.size(0)
        index = torch.randperm(batch_size)

        bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]

        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))

        return images, targets, targets[index], lam

    def _rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2


def get_stl10_transforms() -> transforms.Compose:
    """Transformations pour STL-10 (redimensionne de 96x96 a 32x32)."""
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        transforms.RandomErasing(p=0.1)
    ])


def get_stl10_test_transforms() -> transforms.Compose:
    """Transformations de test pour STL-10."""
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])


class FilteredSTL10(torch.utils.data.Dataset):
    """Dataset STL-10 filtre pour correspondre aux classes CIFAR-10."""

    def __init__(self, root: str, split: str = 'train', transform=None, download: bool = True):
        self.stl10 = torchvision.datasets.STL10(
            root=root,
            split=split,
            transform=transform,
            download=download
        )

        # Filtrer les indices valides (exclure monkey)
        self.valid_indices = []
        for idx in range(len(self.stl10)):
            _, label = self.stl10[idx] if isinstance(self.stl10[idx], tuple) else (None, self.stl10.labels[idx])
            if label in VALID_STL10_CLASSES:
                self.valid_indices.append(idx)

        print(f"STL-10 {split}: {len(self.valid_indices)} images valides sur {len(self.stl10)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        image, label = self.stl10[real_idx]

        # Mapper vers les classes CIFAR-10
        cifar_label = STL10_TO_CIFAR10[label]

        return image, cifar_label


def load_stl10_dataset(download: bool = True) -> Tuple[FilteredSTL10, FilteredSTL10]:
    """Charge le dataset STL-10 filtre."""
    train_dataset = FilteredSTL10(
        root=str(DATA_DIR),
        split='train',
        transform=get_stl10_transforms(),
        download=download
    )

    test_dataset = FilteredSTL10(
        root=str(DATA_DIR),
        split='test',
        transform=get_stl10_test_transforms(),
        download=download
    )

    return train_dataset, test_dataset


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Critere de perte pour MixUp/CutMix."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def set_seed(seed: int = RANDOM_SEED):
    """Fixe les seeds pour la reproductibilite."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch_advanced(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    mixup: Optional[MixUp] = None,
    cutmix: Optional[CutMix] = None,
    use_mixup_prob: float = 0.5
) -> Tuple[float, float]:
    """Entrainement avance avec MixUp et CutMix."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoque {epoch+1} [Train]")

    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        # Appliquer MixUp ou CutMix aleatoirement
        use_mix = np.random.random() < use_mixup_prob

        if use_mix and (mixup is not None or cutmix is not None):
            if np.random.random() < 0.5 and mixup is not None:
                inputs, targets_a, targets_b, lam = mixup(inputs, targets)
            elif cutmix is not None:
                inputs, targets_a, targets_b, lam = cutmix(inputs, targets)
            else:
                use_mix = False
        else:
            use_mix = False

        optimizer.zero_grad()
        outputs = model(inputs)

        if use_mix:
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss = criterion(outputs, targets)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        if scheduler is not None and isinstance(scheduler, OneCycleLR):
            scheduler.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)

        if not use_mix:
            correct += predicted.eq(targets).sum().item()
        else:
            correct += (lam * predicted.eq(targets_a).sum().item()
                       + (1 - lam) * predicted.eq(targets_b).sum().item())

        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def validate_model(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """Validation du modele."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoque {epoch+1} [Valid]")

        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / len(val_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def transfer_learning_training(
    num_epochs_stl10: int = 10,
    num_epochs_cifar10: int = 30,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    use_mlflow: bool = True
) -> Dict:
    """
    Entrainement complet avec transfer learning.

    Phase 1: Pre-entrainement sur STL-10
    Phase 2: Fine-tuning sur CIFAR-10

    Args:
        num_epochs_stl10: Nombre d'epoques pour STL-10
        num_epochs_cifar10: Nombre d'epoques pour CIFAR-10
        batch_size: Taille des batches
        learning_rate: Taux d'apprentissage
        use_mlflow: Activer le tracking MLflow

    Returns:
        Historique d'entrainement
    """
    set_seed()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du device: {device}")

    # Creer le modele
    model = create_model(pretrained=True, freeze_backbone=False)
    model = model.to(device)

    total_params, trainable_params = model.count_parameters()
    print(f"Parametres totaux: {total_params:,}")
    print(f"Parametres entrainables: {trainable_params:,}")

    # Critere avec label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Augmentations avancees
    mixup = MixUp(alpha=0.2)
    cutmix = CutMix(alpha=1.0)

    history = {
        'phase1_train_loss': [],
        'phase1_train_acc': [],
        'phase1_val_loss': [],
        'phase1_val_acc': [],
        'phase2_train_loss': [],
        'phase2_train_acc': [],
        'phase2_val_loss': [],
        'phase2_val_acc': []
    }

    # Configuration MLflow
    if use_mlflow:
        # Corriger l'URI pour Windows (utiliser file:/// ou chemin relatif)
        mlflow_uri = MLFLOW_TRACKING_URI.replace("\\", "/")
        if not mlflow_uri.startswith("file:"):
            mlflow_uri = "file:///" + mlflow_uri
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(EXPERIMENT_NAME + "_TransferLearning")
        mlflow.start_run(run_name="transfer_learning_stl10_cifar10")

        mlflow.log_params({
            'model_name': model.model_name,
            'num_epochs_stl10': num_epochs_stl10,
            'num_epochs_cifar10': num_epochs_cifar10,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'label_smoothing': 0.1,
            'mixup_alpha': 0.2,
            'cutmix_alpha': 1.0,
            'optimizer': 'AdamW',
            'dataset_supplementaire': 'STL-10'
        })

    best_val_acc = 0.0
    start_time = time.time()

    try:
        # =====================================================================
        # PHASE 1: Pre-entrainement sur STL-10
        # =====================================================================
        print("\n" + "="*60)
        print("PHASE 1: Pre-entrainement sur STL-10")
        print("="*60 + "\n")

        # Charger STL-10
        print("Chargement du dataset STL-10...")
        stl10_train, stl10_test = load_stl10_dataset()

        stl10_train_loader = DataLoader(
            stl10_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        stl10_val_loader = DataLoader(
            stl10_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        # Optimiseur pour Phase 1
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=WEIGHT_DECAY
        )

        scheduler = OneCycleLR(
            optimizer,
            max_lr=learning_rate * 10,
            epochs=num_epochs_stl10,
            steps_per_epoch=len(stl10_train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )

        for epoch in range(num_epochs_stl10):
            train_loss, train_acc = train_one_epoch_advanced(
                model, stl10_train_loader, criterion, optimizer, scheduler,
                device, epoch, mixup, cutmix
            )
            val_loss, val_acc = validate_model(
                model, stl10_val_loader, criterion, device, epoch
            )

            history['phase1_train_loss'].append(train_loss)
            history['phase1_train_acc'].append(train_acc)
            history['phase1_val_loss'].append(val_loss)
            history['phase1_val_acc'].append(val_acc)

            print(f"\nPhase 1 - Epoque {epoch+1}/{num_epochs_stl10}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Valid - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

            if use_mlflow:
                mlflow.log_metrics({
                    'phase1_train_loss': train_loss,
                    'phase1_train_acc': train_acc,
                    'phase1_val_loss': val_loss,
                    'phase1_val_acc': val_acc
                }, step=epoch)

        # Sauvegarder le modele apres Phase 1
        phase1_model_path = MODELS_DIR / "model_phase1_stl10.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'phase': 1,
            'val_acc': val_acc
        }, phase1_model_path)
        print(f"\nModele Phase 1 sauvegarde: {phase1_model_path}")

        # =====================================================================
        # PHASE 2: Fine-tuning sur CIFAR-10
        # =====================================================================
        print("\n" + "="*60)
        print("PHASE 2: Fine-tuning sur CIFAR-10")
        print("="*60 + "\n")

        # Charger CIFAR-10
        print("Chargement du dataset CIFAR-10...")
        from src.data.dataset import load_cifar10_dataset, create_data_loaders

        cifar10_train, cifar10_test = load_cifar10_dataset()
        cifar10_train_loader, cifar10_val_loader, cifar10_test_loader = create_data_loaders(
            cifar10_train, cifar10_test, batch_size=batch_size
        )

        # Degeler le backbone pour un fine-tuning complet
        model.unfreeze_backbone()

        # Optimiseur pour Phase 2 avec learning rate plus faible
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate * 0.1,  # LR plus faible pour fine-tuning
            weight_decay=WEIGHT_DECAY
        )

        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )

        for epoch in range(num_epochs_cifar10):
            train_loss, train_acc = train_one_epoch_advanced(
                model, cifar10_train_loader, criterion, optimizer, None,
                device, epoch, mixup, cutmix
            )
            scheduler.step()

            val_loss, val_acc = validate_model(
                model, cifar10_val_loader, criterion, device, epoch
            )

            history['phase2_train_loss'].append(train_loss)
            history['phase2_train_acc'].append(train_acc)
            history['phase2_val_loss'].append(val_loss)
            history['phase2_val_acc'].append(val_acc)

            print(f"\nPhase 2 - Epoque {epoch+1}/{num_epochs_cifar10}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Valid - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

            if use_mlflow:
                mlflow.log_metrics({
                    'phase2_train_loss': train_loss,
                    'phase2_train_acc': train_acc,
                    'phase2_val_loss': val_loss,
                    'phase2_val_acc': val_acc
                }, step=num_epochs_stl10 + epoch)

            # Sauvegarder le meilleur modele
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = MODELS_DIR / "best_model_transfer_learning.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, best_model_path)
                print(f"  Meilleur modele sauvegarde (acc: {val_acc:.2f}%)")

        # Evaluation finale sur le test set
        print("\n" + "="*60)
        print("EVALUATION FINALE SUR LE TEST SET")
        print("="*60 + "\n")

        test_loss, test_acc = validate_model(
            model, cifar10_test_loader, criterion, device, 0
        )

        print(f"Test - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")

        if use_mlflow:
            mlflow.log_metrics({
                'test_loss': test_loss,
                'test_accuracy': test_acc,
                'best_val_accuracy': best_val_acc
            })

            # Logger le modele final
            mlflow.pytorch.log_model(model, "model_transfer_learning")
            mlflow.log_artifact(str(best_model_path))

    finally:
        total_time = time.time() - start_time

        if use_mlflow:
            mlflow.log_metric('total_training_time_seconds', total_time)
            mlflow.end_run()

    print(f"\n{'='*60}")
    print(f"Entrainement termine en {total_time/60:.2f} minutes")
    print(f"Meilleure accuracy de validation: {best_val_acc:.2f}%")
    print(f"Accuracy sur le test set: {test_acc:.2f}%")
    print(f"{'='*60}\n")

    return history


if __name__ == "__main__":
    print("="*60)
    print("TRANSFER LEARNING: STL-10 + CIFAR-10")
    print("="*60)

    history = transfer_learning_training(
        num_epochs_stl10=10,
        num_epochs_cifar10=30,
        batch_size=128,
        learning_rate=0.001,
        use_mlflow=True
    )

    print("\nEntrainement termine avec succes!")
