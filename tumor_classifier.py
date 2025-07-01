import torch
import torch.nn as nn

import torch
from sklearn.metrics import classification_report, confusion_matrix
import copy
from monai.metrics import ConfusionMatrixMetric
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, random_split,ConcatDataset
from monai.data import pad_list_data_collate
from torch.utils.data import Dataset
import torch.optim as optim
import numpy as np
import pandas as pd
import sys
import os
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import umap
from sklearn.metrics.pairwise import rbf_kernel
from monai.data import Dataset, DataLoader
from scipy.spatial import distance
import seaborn as sns
from monai.losses import FocalLoss
#imported model from MedicalNet
#from models import resnet18
from model import generate_model
from setting import parse_opts
from monai.networks.nets import DenseNet121, DenseNet169

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, RandFlipd, RandRotate90d, RandGaussianNoised, NormalizeIntensityd,
    ToTensord
)
train_transforms = Compose([
    # Don't use LoadImaged since data is already loaded
    EnsureChannelFirstd(keys=["image"],channel_dim=0),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
    RandRotate90d(keys=["image"], prob=0.5, max_k=3),
    RandGaussianNoised(keys="image", prob=0.2),
    ToTensord(keys=["image"]),
])

val_transforms = Compose([
    EnsureChannelFirstd(keys=['image'],channel_dim=0),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ToTensord(keys=['image'])
])



os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

tumor_to_idx = {
    "MyxofibroSarcomas": 0,
    "LeiomyoSarcomas": 1,
    "DTF": 2,
    "MyxoidlipoSarcoma": 3,
    "WDLPS": 4,
}



class ResNetWithClassifier(nn.Module):
    def __init__(self, base_model, in_channels=1, num_classes=5):  # change num_classes to match your setting
        super().__init__()
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])

        #might not be needed
        self.pool = nn.AdaptiveAvgPool3d(1)


        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        # print(f'Shape before pooling {x.shape}')
        # x = self.pool(x)
        # print(f'Shape after pooling {x.shape}')
        #pooled = self.pool(x)
        return self.classifier(x)

    def extract_features(self, x):
        x = self.encoder(x)
        #pooled = self.pool(x)
        return x.view(x.size(0), -1) #sahpe (B,512)

class QADataset(Dataset):
    def __init__(self, fold, preprocessed_dir, fold_paths, transform=None):
        """
        fold: str, e.g. 'fold_0'
        preprocessed_dir: base preprocessed path with .npz images
        pred_fold_paths: dict with predicted mask folder paths per fold
        fold_paths: dict with fold folder paths containing Dice scores & case IDs
        """
        self.fold = fold
        self.preprocessed_dir = preprocessed_dir
        self.fold_dir = fold_paths[fold]


        # Load Dice scores & case IDs from a CSV or JSON
        # Example CSV: case_id,dice

        self.metadata = pd.read_csv(self.fold_dir)

        # List of case_ids
        self.case_ids = self.metadata['case_id'].tolist()
        #self.dice_scores = self.metadata['dice'].tolist()
        self.subtypes = self.metadata['subtype'].tolist()
        #self.ds = nnUNetDatasetBlosc2(self.preprocessed_dir)

        self.transform = transform

        self.tumor_to_idx = tumor_to_idx

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        #dice_score = self.dice_scores[idx]
        subtype = self.subtypes[idx]
        subtype = subtype.strip()

        label_idx = self.tumor_to_idx[subtype]
        # Load preprocessed image (.npz)
        #Check if its not case_id_0000
        # data, seg, seg_prev, properties = self.ds.load_case(case_id)
        #print("Data shape:", data.shape)
        file = f'{case_id}_resized.pt'
        image = torch.load(os.path.join(self.preprocessed_dir, file))

        # file = f'{case_id}_features_roi.npz'
        # feat1 = np.load(os.path.join(self.preprocessed_dir, file))
        # image = feat1[feat1.files[0]]


        assert image.ndim == 4 and image.shape[0] == 1, f"Expected shape (1, H, W, D), but got {image.shape}"
        #image = np.asarray(image)
        #print(f'Image Shape {image.shape}')

        data_dict = {'image': image }
        # nnU-Net raw images usually have multiple channels; choose accordingly:
        # Here, just take channel 0 for simplicity:
        #input_image = np.stack([image[0], pred_mask], axis=0)  # (2, H, W, D)
        # Map dice score to category
        if self.transform:
            data_dict = self.transform(data_dict)

        image_tensor = data_dict['image']
        label_tensor = torch.tensor(label_idx).long()


        # print('Image tensor shape : ', image_tensor.shape)
        # print('Label tensor shape : ', label_tensor.shape)

        return {
            'input': image_tensor,  # shape (C_total, D, H, W)
            'label': label_tensor,  # scalar tensor
        }



def train_one_fold(model, preprocessed_dir, plot_dir, fold_paths, optimizer, scheduler, num_epochs, patience, device,fold):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    patience_counter = 0

    print(f"Training fold {fold} ...")

    # Initialize datasets for this fold
    train_fold_ids = [f"fold_{i}" for i in range(5) if i != fold]  # other folds for training
    val_fold_id = f"fold_{fold}"  # current fold for validation

    # Combine training folds datasets
    train_datasets = []
    for train_fold in train_fold_ids:
        ds = QADataset(
            fold=train_fold,
            preprocessed_dir=preprocessed_dir,
            fold_paths=fold_paths,
            transform=train_transforms
        )
        train_datasets.append(ds)
    train_dataset = ConcatDataset(train_datasets)
    del train_datasets

    val_dataset = QADataset(
        fold=val_fold_id,
        preprocessed_dir=preprocessed_dir,
        fold_paths=fold_paths,
        transform=val_transforms
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=True, num_workers=4, collate_fn=pad_list_data_collate)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, pin_memory=True, num_workers=4)

    class_counts = torch.tensor([
        46,  # MyxofibroSarcomas (idx 0)
        24,  # LeiomyoSarcomas    (idx 1)
        54,  # DTF                (idx 2)
        28,  # MyxoidlipoSarcoma  (idx 3)
        44,  # WDLPS              (idx 4)
    ], dtype=torch.float)

    class_weights = 1.0 / class_counts
    #class_weights = class_weights / class_weights.sum()

    loss_function = FlssocalLoss(
        to_onehot_y= True,
        use_softmax=True,
        gamma=3.0,
        weight=class_weights.to(device)  # alpha term
    )
    train_losses = []  # <-- add here, before the epoch loop
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        print(f"Epoch {epoch+1}/{num_epochs}")
        running_loss, correct, total = 0.0, 0, 0
        preds_list, labels_list = [], []

        scaler = GradScaler()

        for batch in train_loader:
            inputs = batch['input'].to(device)
            labels = batch['label'].to(device)
            #print("Input shape:", inputs.shape)

            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                preds = torch.argmax(outputs, dim=1)
                preds_cpu = preds.detach().cpu()
                labels_cpu = labels.detach().cpu()


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)

            if epoch % 5 == 0:
                preds_list.extend(preds_cpu.numpy())
                labels_list.extend(labels_cpu.numpy())



        epoch_train_loss = running_loss / total
        epoch_train_acc = correct.double() / total
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")

        train_losses.append(epoch_train_loss)

        if epoch % 5 == 0:
            idx_to_tumor = {v: k for k, v in tumor_to_idx.items()}
            pred_tumors = [idx_to_tumor[p] for p in preds_list]
            true_tumors = [idx_to_tumor[t] for t in labels_list]

            print(classification_report(true_tumors, pred_tumors, digits=4, zero_division=0))
        del inputs, outputs,labels, preds
        torch.cuda.empty_cache()

        # --- Validation phase ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_preds_list, val_labels_list = [], []

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                labels = batch['label'].to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                preds = torch.argmax(outputs, dim=1)
                preds_cpu = preds.detach().cpu()
                labels_cpu = labels.detach().cpu()

                val_loss += loss.item() * inputs.size(0)
                val_correct += torch.sum(preds == labels.data)
                val_total += labels.size(0)
                val_preds_list.extend(preds_cpu.numpy())
                val_labels_list.extend(labels_cpu.numpy())

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct.double() / val_total

        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        val_losses.append(epoch_val_loss)

        val_pred_tumors = [idx_to_tumor[p] for p in val_preds_list]
        val_true_tumors = [idx_to_tumor[t] for t in val_labels_list]
        print(classification_report(val_true_tumors, val_pred_tumors, digits=4, zero_division=0))

        warmup_epochs = 10
        base_lr = 1e-4
        if epoch < warmup_epochs:
            lr_scale = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr * lr_scale
        else:
            scheduler.step(epoch_val_loss)
        # if epoch > 10:
        #     scheduler.step(epoch_val_loss)

        # Log current learning rate(s)
        for i, param_group in enumerate(optimizer.param_groups):
            print(f"LR after scheduler step (param group {i}): {param_group['lr']:.6f}")


        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            best_report = classification_report(val_true_tumors, val_pred_tumors, digits=4, zero_division=0)
            cm = confusion_matrix(val_true_tumors,val_pred_tumors)
            #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(idx_to_tumor.values()))

            print(f"✅ New best model saved at epoch {epoch + 1} with val loss {epoch_val_loss:.4f}")

            torch.save(best_model_wts, f"best_model_fold_{fold}.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹️ Early stopping")
                model.load_state_dict(best_model_wts)

                labels = ["DTF", "LeiomyoSarcomas", "MyxofibroSarcomas", "MyxoidlipoSarcoma", "WDLPS"]
                plt.figure(figsize=(8, 6))  # Increase figure size
                sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", xticklabels=labels, yticklabels=labels)
                plt.title("Confusion Matrix - Fold 0", fontsize=14)
                plt.xlabel("Predicted Label", fontsize=12)
                plt.ylabel("True Label", fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()  # Ensures everything fits in the figure area
                plt.savefig(os.path.join(plot_dir, f"confusion_matrix_fold_{fold}.png"))
                plt.close()
                file = os.path.join(plot_dir, f"classification_report_fold_{fold}.txt")

                print('Best Report')
                print(best_report)

                with open(file, "w") as f:
                    f.write(f"Final Classification Report for Fold {fold}:\n")
                    f.write(best_report)
                return model, train_losses, val_losses


    #model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses

def plot_UMAP(train, y_train, neighbours, m, name, image_dir):
    print(f'feature shape {train.shape}')

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=neighbours,
        min_dist=0.1,
        metric=m,
        random_state=42
    )
    train_umap = reducer.fit_transform(train)  # (N, 2)
    # Apply UMAP transform to validation data
    # val_umap = reducer.transform(val)

    # Map labels back to names (optional)
    idx_to_tumor = {v: k for k, v in tumor_to_idx.items()}


    label_names_train = [idx_to_tumor[i] for i in y_train]

    # combined_umap = np.vstack([train_umap, val_umap])


    # all_subtypes= np.concatenate([y_train, y_val])
    unique_subtypes = sorted(set(y_train))

    # labels = np.array(['train'] * len(train_umap) + ['val'] * len(val_umap))
    markers = {'train': 'o', 'val': 's'}

    cmap = plt.cm.tab20
    color_lookup = {lab: cmap(i % 20) for i, lab in enumerate(unique_subtypes)}
    # 7. scatter plot
    plt.figure(figsize=(8, 6))
    # for marker_type in ['train', 'val']:
    for subtype in unique_subtypes:
        idx = [i for i, lab in enumerate(y_train) if lab == subtype]
        if not idx: continue
        plt.scatter(
            train_umap[idx, 0], train_umap[idx, 1],
            s=25,
            c=[color_lookup[subtype]] * len(idx),
            label=f"{idx_to_tumor[subtype]} ",
            alpha=0.8,
        )

    plt.xlabel("UMAP‑1")
    plt.ylabel("UMAP‑2")
    plt.title("ROI Feature Map Clusters by Subtype and Set")
    plt.legend(fontsize=8, loc='best', markerscale=1)
    plt.tight_layout()
    image_loc = os.path.join(image_dir, name)
    plt.savefig(image_loc, dpi=300)
    plt.show()

def intra_class_distance(X_train, y_train):

    #Intra-Class distance
    intra_class_dists_maha = {}
    intra_class_dists_euc = {}
    std_maha = {}
    std_euc = {}

    for subtype in np.unique(y_train):
        features = X_train[y_train == subtype]
        mean_vec = features.mean(axis=0)

        # Mahalanobis setup
        cov = np.cov(features.T)
        inv_covmat = np.linalg.pinv(cov)

        mahalanobis = [distance.mahalanobis(f, mean_vec, inv_covmat) for f in features]
        euclidean = np.linalg.norm(features - mean_vec, axis=1)

        intra_class_dists_maha[subtype] = np.mean(mahalanobis)
        intra_class_dists_euc[subtype] = np.mean(euclidean)

        std_maha[subtype] = np.std(mahalanobis)
        std_euc[subtype] = np.std(euclidean)

    return intra_class_dists_maha, intra_class_dists_euc, std_maha, std_euc

def plot_intra_class_distances(intra_class_dists_maha, intra_class_dists_euc, std_maha, std_euc,plot_dir):
    subtypes = list(intra_class_dists_maha.keys())
    maha_values = [intra_class_dists_maha[sub] for sub in subtypes]
    euc_values = [intra_class_dists_euc[sub] for sub in subtypes]
    maha_err = [std_maha[sub] for sub in subtypes]
    euc_err = [std_euc[sub] for sub in subtypes]

    x = np.arange(len(subtypes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, maha_values, width, yerr=maha_err, label='Mahalanobis', color='steelblue', capsize=5)
    ax.bar(x + width/2, euc_values, width, yerr=euc_err, label='Euclidean', color='orange', capsize=5)


    ax.set_ylabel("Average Intra-Class Distance")
    ax.set_xlabel("Subtype")
    ax.set_title("Intra-Class Distance per Tumor Subtype (with Std)")
    ax.set_xticks(x)

    ax.set_xticklabels(subtypes, rotation=45)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'intra_class_distances.png'))
    plt.close()

def compute_mmd(x, y, gamma=None):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two samples x and y using RBF kernel.
    Args:
        x: numpy array, shape (n_samples_x, n_features)
        y: numpy array, shape (n_samples_y, n_features)
        gamma: float or None, kernel parameter for RBF. If None, 1/n_features is used.
    Returns:
        mmd: float, squared MMD value between distributions of x and y
    """
    if gamma is None:
        gamma = 1.0 / x.shape[1]

    Kxx = rbf_kernel(x, x, gamma=gamma)
    Kyy = rbf_kernel(y, y, gamma=gamma)
    Kxy = rbf_kernel(x, y, gamma=gamma)

    mmd = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
    return mmd

def inter_class_distance(X_train, y_train, plot_dir):
    """
      Compute the Maximum Mean Discrepancy (MMD) between two samples x and y using RBF kernel.
      Args:
          x: numpy array, shape (n_samples_x, n_features)
          y: numpy array, shape (n_samples_y, n_features)
          gamma: float or None, kernel parameter for RBF. If None, 1/n_features is used.
      Returns:
          mmd: float, squared MMD value between distributions of x and y
      """
    sorted_tumors = sorted(tumor_to_idx.items(), key=lambda x: x[1])
    unique_subtypes = [tumor for tumor, _ in sorted_tumors]
    idx_to_tumor = {v: k for k, v in tumor_to_idx.items()}

    mmd_matrix = np.zeros((len(unique_subtypes), len(unique_subtypes)))

    # for i, subtype_i in enumerate(unique_subtypes):
    #     xi = X_train[y_train == subtype_i]
    #     for j, subtype_j in enumerate(unique_subtypes):
    #         xj = X_train[y_train == subtype_j]
    #         mmd_matrix[i, j] = compute_mmd(xi, xj)
    #
    for i, (_, idx_i) in enumerate(sorted_tumors):
        xi = X_train[y_train == idx_i]
        for j, (_, idx_j) in enumerate(sorted_tumors):
            xj = X_train[y_train == idx_j]
            mmd_matrix[i, j] = compute_mmd(xi, xj)

    # Create pretty labels with names and indices
    pretty_labels = [f"{tumor} ({tumor_to_idx[tumor]})" for tumor in unique_subtypes]#

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(mmd_matrix, xticklabels=pretty_labels, yticklabels=pretty_labels,
                cmap="viridis", annot=True, fmt=".3f")
    plt.title("MMD Distance Matrix Between Tumor Subtypes")
    plt.xlabel("Subtype")
    plt.ylabel("Subtype")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'MMD_distance.png'))
    plt.close()

    return mmd_matrix


def plot_mmd_diag_vs_offdiag(mmd_matrix, y_train, plot_dir):
    mmd_matrix = np.array(mmd_matrix)

    unique_subtypes = np.unique(y_train)
    # Diagonal values: intra-class distances (ideally close to 0)
    diag_values = np.diag(mmd_matrix)

    # Off-diagonal values: inter-class distances
    off_diag_values = mmd_matrix[~np.eye(mmd_matrix.shape[0], dtype=bool)]

    # Create boxplot data
    data = [
        diag_values,  # intra-class
        off_diag_values  # inter-class
    ]

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=data, palette=["#66c2a5", "#fc8d62"])
    plt.xticks([0, 1], ['Intra-Class (Diagonal)', 'Inter-Class (Off-Diagonal)'])
    plt.ylabel('MMD Distance')
    plt.title('Intra- vs Inter-Class MMD Comparison')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'MMD_diag_vs_offdiag.png'))
    plt.close()


def main(preprocessed_dir, plot_dir, fold_paths, device):
    for fold in range(1):
        #Loading MedicalNet model and weights
        from argparse import Namespace

        sets = Namespace(
            model='resnet',
            model_depth=18,
            resnet_shortcut='A',
            input_D=48,
            input_H=272,
            input_W=256,
            n_input_channels=1,
            n_seg_classes=5,
            gpu_id=[0],
            no_cuda=False,
            phase='train',
            pretrain_path='pretrain/resnet_18.pth',
            new_layer_names=['conv_seg'],
            manual_seed=1,
            learning_rate=0.001,
            batch_size=4,
            num_workers=4,
            resume_path='',
            save_intervals=10,
            n_epochs=200,
            data_root=preprocessed_dir,
            img_list='./data/train.txt',
            ci_test=False,
        )

        base_model, _ = generate_model(sets)

        # Load pretrained weights
        pretrained_dict = torch.load('/pretrain/resnet_18.pth')['state_dict']
        base_model.load_state_dict(pretrained_dict,strict=False)



        model = ResNetWithClassifier(base_model, in_channels =1, num_classes=5)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, min_lr=1e-6)
        #criterion = nn.CrossEntropyLoss()


        best_model, train_losses, val_losses= train_one_fold(model, preprocessed_dir, plot_dir,fold_paths,optimizer, scheduler,
                                    num_epochs=100, patience=20, device=device, fold=fold)

        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curves')
        plt.savefig(os.path.join(plot_dir, 'loss_curves.png'))

def extract_features(train_dir, fold_paths, device, plot_dir):
    model = TumorClassifier(model_depth=18, in_channels=1, num_classes=5)
    model.load_state_dict(torch.load("best_model_fold_0.pth", map_location=device))
    model.to(device)
    model.eval()
    # Combine training folds datasets
    train_fold_ids = [f"fold_{i}" for i in range(5)]
    train_datasets = []
    for train_fold in train_fold_ids:
        ds = QADataset(
            fold=train_fold,
            preprocessed_dir=train_dir,
            fold_paths=fold_paths,
        )
        train_datasets.append(ds)
    train_dataset = ConcatDataset(train_datasets)
    del train_datasets


    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=4)

    # val_dataset = QADataset(
    #     fold='ood_val',
    #     preprocessed_dir=val_dir,
    #     fold_paths=fold_paths
    # )
    # val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)

    # all_features_val = []
    # all_labels_val = []

    all_features_train = []
    all_labels_train = []

    # with torch.no_grad():
    #     for batch in val_loader:
    #         inputs = batch['input'].to(device)  # (B, C, D, H, W)
    #         labels = batch['label'].cpu().numpy()  # class indices
    #
    #         features = model.extract_features(inputs).cpu().numpy()  # (B, 512)
    #         all_features_val.append(features)
    #         all_labels_val.extend(labels)
    #
    # # Combine into arrays
    # X_val = np.concatenate(all_features_val, axis=0)
    # y_val = np.array(all_labels_val)

    with torch.no_grad():
        for batch in train_loader:
            inputs = batch['input'].to(device)  # (B, C, D, H, W)
            labels = batch['label'].cpu().numpy()  # class indices

            features = model.extract_features(inputs).cpu().numpy()  # (B, 512)
            all_features_train.append(features)
            all_labels_train.extend(labels)

    # Combine into arrays
    X_train = np.concatenate(all_features_train, axis=0)
    y_train = np.array(all_labels_train)

    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X_train)

    plot_UMAP(X_train, y_train, neighbours=5, m='cosine', name='UMAP_cosine_5n_fold0.png', image_dir=plot_dir)
    plot_UMAP(X_train,y_train,neighbours=10, m='cosine', name='UMAP_cosine_10n_fold0.png', image_dir =plot_dir)
    #plot_UMAP(X_train, y_train, neighbours=15, m='cosine', name='UMAP_cosine_15n_fold0.png', image_dir=plot_dir)


    plot_UMAP(X_scaled, y_train, neighbours=5, m='manhattan', name='UMAP_manh_5n_fold0.png', image_dir=plot_dir)
    plot_UMAP(X_scaled,y_train,neighbours=10, m='manhattan', name='UMAP_manh_10n_fold0.png', image_dir =plot_dir)
    #plot_UMAP(X_scaled, y_train, neighbours=15, m='manhattan', name='UMAP_manh_15n_fold0.png', image_dir=plot_dir)

    maha, euc, std_maha, std_euc = intra_class_distance(X_scaled, y_train)
    plot_intra_class_distances(maha,euc, std_maha, std_euc,plot_dir)
    mmd_matrix = inter_class_distance(X_scaled, y_train, plot_dir)
    plot_mmd_diag_vs_offdiag(mmd_matrix,y_train, plot_dir)





if __name__ == '__main__':
    fold_paths = {
        'fold_0': '/gpfs/home6/palfken/masters_thesis/fold_0',
        'fold_1': '/gpfs/home6/palfken/masters_thesis/fold_1',
        'fold_2': '/gpfs/home6/palfken/masters_thesis/fold_2',
        'fold_3': '/gpfs/home6/palfken/masters_thesis/fold_3',
        'fold_4': '/gpfs/home6/palfken/masters_thesis/fold_4',
    }
    preprocessed= sys.argv[1]
    plot_dir = sys.argv[2]


    main(preprocessed, plot_dir, fold_paths, device = 'cuda')
    #extract_features(preprocessed,fold_paths, device = 'cuda', plot_dir = plot_dir)

