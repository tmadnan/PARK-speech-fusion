import os
import copy
import pickle
import pandas as pd

from tqdm import tqdm

from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.metrics import auc, roc_auc_score, roc_curve, f1_score, accuracy_score, recall_score, precision_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from mlxtend.plotting import plot_confusion_matrix

import numpy as np
import matplotlib.pyplot as plt

import wandb
import random
import click
import json
from imblearn.over_sampling import SMOTE

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


FEATURES_FILE_ROOT = "../data/"

MODEL_TAG = "A"
MODEL_PATH = os.path.join(f"../models/fox_model_{MODEL_TAG}","predictive_model/model.pth")
SCALER_PATH = os.path.join(f"../models/fox_model_{MODEL_TAG}","scaler/scaler.pth")
MODEL_CONFIG_PATH = os.path.join(f"../models/fox_model_{MODEL_TAG}","predictive_model/model_config.json")
TRAIN_FILE = "../data/train_set_quick_brown_fox_task.csv"
TEST_FILE = "../data/test_set_quick_brown_fox_task.csv"
MODEL_BASE_PATH = f"../models/fox_model_{MODEL_TAG}"

DEV_ID_FILE = "../data/audio_projection_dev_set_participants.txt"
TEST_ID_FILE = "../data/audio_projection_test_set_participants.txt"


dev_ids = set()
test_ids = set()

'''
Load patient ids in dev and test sets
'''
with open(DEV_ID_FILE) as f:
    ids = f.readlines()
    dev_ids = set([x.strip() for x in ids])
    # print(len(dev_ids))

with open(TEST_ID_FILE) as f:
    ids = f.readlines()
    test_ids = set([x.strip() for x in ids])
    # print(len(test_ids))



'''
set-up device (for gpu support)
'''
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print(f"Running on {device} ...")

def load(feature_path, drop_correlated = True, corr_thr = 0.85):
    df = pd.read_csv(feature_path)

    '''
    Drop data point if any of the feature is null
    '''
    df = df.dropna(subset = df.columns.difference(['dob','time_mdsupdrs', 'age', 'ethnicity', 'gender', 'race']), how='any')
    
    '''
    Drop metadata columns to focus on features
    '''
    
    df_columns_to_keep = df[['Filename', 'pd']]
 
    
    df_features = df.drop(columns=['Filename', 'Participant_ID', 'gender','age','race', 'pd'])

    '''
    Drop columns (if set true) if it is correlated with another one with PCC>thr
    '''
    if drop_correlated:
        corr_matrix = df_features.corr()
        iters = range(len(corr_matrix.columns) - 1)
        drop_cols = []

        for i in iters:
            for j in range(i+1):
                item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
                col = item.columns
                row = item.index
                val = abs(item.values)

                if val >= corr_thr:
                    drop_cols.append(col.values[0])

        drops = set(drop_cols)
        
        # Drop features from both the main and the feature dataframe
        df.drop(drops, axis=1, inplace=True)
        df_features.drop(drops, axis=1, inplace=True)
    # end of drop correlated columns implementation

    features = df[df_features.columns.to_list()]
    
    columns = features.columns
    
    # now concat the df_features with the df_columns_to_keep 
    features = pd.concat([df_columns_to_keep, df_features], axis=1)
    
    # drop duplicates based on the Filename column
    features = features.drop_duplicates(subset=['Filename'])
 
    # make the Filename column the index
    #features.set_index('Filename', inplace=True)
    
    return features, columns
    
    

def merge_features(feature_file_list, feature_column_list):
    
    filename_pd_pairs = []
    for feature_file in feature_file_list:
        filename_pd_pairs.append(set(zip(feature_file['Filename'], feature_file['pd'])))
        
    intersection = filename_pd_pairs[0]
    for i in range(1, len(filename_pd_pairs)):
        intersection = intersection.intersection(filename_pd_pairs[i])
        
    intersection = list(intersection)
    
    # create a dataframe with the intersection
    intersection_df = pd.DataFrame(intersection, columns=['Filename', 'pd'])
    
    # sort the intersection_df by the Filename
    intersection_df = intersection_df.sort_values(by='Filename')
    
    # print(intersection_df.shape)
    
    intersection_df = intersection_df.drop_duplicates(subset='Filename')
    
    file_names = intersection_df
    
    feature_list = []
    IDs = []
    labels = []
    
    for i in range(len(feature_file_list)):
        # this time we will only keep the rows in the feature files that are in the intersection
        feature_file = feature_file_list[i]
        feature_file = feature_file[feature_file['Filename'].isin(intersection_df['Filename'])]
        # maintain the order of the intersection_df
        feature_file = feature_file.set_index('Filename')
        feature_file = feature_file.reindex(intersection_df['Filename'])
        feature_file = feature_file[feature_column_list[i]]
        feature_list.append(feature_file.to_numpy())
        
        
        

        
        
    IDs.extend(file_names['Filename'])
    labels.extend(file_names['pd'])
    return feature_list, labels, IDs
    # print(file_names.shape)
    


def parse_patient_id(name:str):
    if name.startswith("NIH"): [ID, *_] = name.split("-")
    elif name.endswith("quick_brown_fox.mp4") and '%' not in name : [*_, ID, _] = name.split("-")
    else: [*_, ID, _, _, _] = name.split("_")
    return ID

'''
Based on the universal train/test split, split the dataframe into train and test sets
'''
def train_test_split(feature_df_list, labels, ids):
    
    # train_ids = [parse_patient_id(Filename) for Filename in (list)(all_files_train)]
    # test_ids = [parse_patient_id(Filename) for Filename in (list)(all_files_test)]

    filenames = ids
    
    ids = [parse_patient_id(Filename) for Filename in ids]
    
    feature_train_list = []
    feature_test_list = []
    filenames_test = []
    
    
    for i in range(len(feature_df_list)):
        features_train = []
        labels_train = []
        ids_train = []
        
        features_test = []
        labels_test = []
        ids_test = []
    
        
        for x, l, pid, filename in zip(feature_df_list[i], labels, ids, filenames):
            if pid in test_ids:
                ids_test.append(pid)
                features_test.append(x)
                labels_test.append(l)
                filenames_test.append(filename)
                
            else:
                ids_train.append(pid)
                features_train.append(x)
                labels_train.append(l) 
        
        feature_train_list.append(features_train)
        feature_test_list.append(features_test)
        
    return feature_train_list, labels_train, ids_train, feature_test_list, labels_test, ids_test, filenames_test



def train_dev_split(ids, labels, feature_sets):
    
    feature_train_list = []
    feature_dev_list = []
    
    
    for i in range(len(feature_sets)):
        features_train = []
        labels_train = []
        ids_train = []
        
        features_dev = []
        labels_dev = []
        ids_dev = []
    
        
        for x, l, pid in zip(feature_sets[i], labels, ids):
            if pid in dev_ids:
                ids_dev.append(pid)
                features_dev.append(x)
                labels_dev.append(l)
                
            else:
                ids_train.append(pid)
                features_train.append(x)
                labels_train.append(l) 
        
        feature_train_list.append(features_train)
        feature_dev_list.append(features_dev)
        
    return feature_train_list, labels_train, ids_train, feature_dev_list, labels_dev, ids_dev


class Prjection_Dataset(Dataset):
    def __init__(self, feature_list, labels):
        self.feature_list = feature_list
        self.labels = labels
        self.n_modalities = len(feature_list)
        
    def __getitem__(self, index):
        features = [torch.Tensor(self.feature_list[i][index]) for i in range(self.n_modalities)]
        labels = torch.Tensor([self.labels[index]])
        return features, labels
    
    def __len__(self):
        return len(self.labels)
        


class Projection_Shallow_ANN(nn.Module):
    def __init__(self, dimensions, chosen_modality_index, normalization_dim=1, first_operation='normalize'):
        super().__init__()
        self.dimensions = dimensions
        self.n_modalities = len(dimensions)
        self.chosen_modality_index = chosen_modality_index
        # Use the dimension of the chosen modality as the projection dimension for all other modalities.
        self.projection_dim = dimensions[chosen_modality_index]
        self.normalization_dim = normalization_dim
        self.first_operation = first_operation
        
        # Initialize projection layers for all modalities except the chosen one.
        self.projections = nn.ModuleList()
        for i in range(self.n_modalities):
            if i != chosen_modality_index:
                self.projections.append(nn.Linear(dimensions[i], self.projection_dim))
            else:
                # Placeholder for the chosen modality (no projection needed).
                self.projections.append(None)
        
        # Initialize reconstruction layers for all modalities.
        self.reconstruction = nn.ModuleList([nn.Linear(self.projection_dim, dimensions[i]) for i in range(self.n_modalities)])
        
        # A single fully connected layer and sigmoid activation for processing the combined features.
        self.fc = nn.Linear(in_features=self.projection_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, features):
        projected_features = []
        for i in range(self.n_modalities):
            if i == self.chosen_modality_index:
                # Directly use the data from the chosen modality without projection.
                projected_features.append(features[i])
            else:
                # Project the data for all other modalities.
                projected_features.append(self.projections[i](features[i]))
        
        # Optionally normalize and then combine the projected features.
        if self.first_operation == 'normalize':
            normalized_projected_features = [F.normalize(f, p=2, dim=self.normalization_dim) for f in projected_features]
            processed_features = sum(normalized_projected_features)
        else:
            added_features = sum(projected_features)
            processed_features = F.normalize(added_features, p=2, dim=self.normalization_dim)
        
        # Process the combined features through the final layer and sigmoid activation.
        output = self.fc(processed_features)
        output = self.sigmoid(output)
        
        # Attempt to reconstruct the original dimensions for all modalities.
        reconstructed_features = [self.reconstruction[i](projected_features[i]) for i in range(self.n_modalities)]
        
        return projected_features, output, reconstructed_features

class Projection_ANN(nn.Module):
    def __init__(self, dimensions, chosen_modality_index, normalization_dim=1, first_operation='normalize'):
        super().__init__()
        self.dimensions = dimensions
        self.n_modalities = len(dimensions)
        self.chosen_modality_index = chosen_modality_index
        self.projection_dim = dimensions[chosen_modality_index]
        self.normalization_dim = normalization_dim
        self.first_operation = first_operation
        
        # Initialize projections for all modalities except the chosen one.
        self.projections = nn.ModuleList()
        for i in range(self.n_modalities):
            if i != chosen_modality_index:
                self.projections.append(nn.Linear(dimensions[i], self.projection_dim))
            else:
                # Placeholder for the chosen modality (no projection needed).
                self.projections.append(None)
        
        # Reconstruction layers for all modalities (including the chosen one).
        self.reconstruction = nn.ModuleList([nn.Linear(self.projection_dim, dimensions[i]) for i in range(self.n_modalities)])

        # Neural network layers for further processing.
        self.fc1 = nn.Linear(in_features=self.projection_dim, out_features=(int)(self.projection_dim/2))
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(in_features=(int)(self.projection_dim/2), out_features=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, features):
        projected_features = []
        for i in range(self.n_modalities):
            if i == self.chosen_modality_index:
                # Directly use the data from the chosen modality.
                projected_features.append(features[i])
            else:
                # Project the data for all other modalities.
                projected_features.append(self.projections[i](features[i]))
        
        # Optionally normalize and then combine the projected features.
        if self.first_operation == 'normalize':
            normalized_projected_features = [F.normalize(f, p=2, dim=self.normalization_dim) for f in projected_features]
            processed_features = sum(normalized_projected_features)
        else:
            added_features = sum(projected_features)
            processed_features = F.normalize(added_features, p=2, dim=self.normalization_dim)
        
        # Further process the combined features.
        output = self.fc1(processed_features)
        output = self.activation(output)
        output = self.fc2(output)
        output = self.sigmoid(output)
        
        # Reconstruct the original data dimensions for all modalities.
        reconstructed_features = [self.reconstruction[i](projected_features[i]) for i in range(self.n_modalities)]
        
        return projected_features, output, reconstructed_features


def cosine_similarity(projected_features):
    projected_features = [F.normalize(projected_features[i], p=2, dim=1) for i in range(len(projected_features))]
    cosine_similarity = torch.sum(projected_features[0]*projected_features[1], dim=1)
    return 1 - cosine_similarity.mean()

def calculate_reconstruction_loss(reconstructed_features, features, loss_category = 'mse'):
    if loss_category == 'mse':
        loss = [F.mse_loss(reconstructed_features[i], features[i], reduction='mean') for i in range(len(reconstructed_features))]
        return sum(loss)/len(loss)
    elif loss_category == 'l1':
        loss = [F.l1_loss(reconstructed_features[i], features[i], reduction='mean') for i in range(len(reconstructed_features))]
        return sum(loss)/len(loss)
    elif loss_category == 'huber':
        loss = [F.smooth_l1_loss(reconstructed_features[i], features[i], reduction='mean') for i in range(len(reconstructed_features))]
        return sum(loss)/len(loss)
    elif loss_category == 'kl':
        loss = [F.kl_div(reconstructed_features[i], features[i], reduction='mean') for i in range(len(reconstructed_features))]
        return sum(loss)/len(loss)
    else:
        raise ValueError("Invalid loss category")
    

def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0  # or some other default value or handling
    else:
        return numerator / denominator
    


def calculate_metrics_at_thresholds(all_labels, all_preds, thresholds, model_base_path=".", weights=None):
    """
    Calculate sensitivity, specificity, accuracy, and F1-score for a range of thresholds.

    Parameters:
    - all_labels: Ground truth labels (0/1).
    - all_preds: Model predictions (logits or probabilities).
    - thresholds: List or array of thresholds to evaluate.
    - model_base_path: Path to save predictions for debugging (optional).

    Returns:
    - metrics: Dictionary containing thresholds, sensitivity, specificity, accuracy, and F1-score values.
    """
    sensitivity = []
    specificity = []
    accuracy = []
    f1_scores = []
    precision = []
    recall = []
    composite_scores = []
    
    if weights is None:
        weights = {
            "sensitivity": 1.0,
            "specificity": 1.0,
            "accuracy": 1.0,
            "f1_scores": 1.0,
        }
        
    for threshold in thresholds:
        preds = (all_preds >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()

        sens = recall_score(all_labels, preds)
        spec = safe_divide(tn, tn + fp)
        acc = accuracy_score(all_labels, preds)
        f1 = f1_score(all_labels, preds)
        prec = precision_score(all_labels, preds, zero_division=0)
        rec = sens

        sensitivity.append(sens)
        specificity.append(spec)
        accuracy.append(acc)
        f1_scores.append(f1)
        precision.append(prec)
        recall.append(rec)
        
        composite_score = (
            weights["sensitivity"] * sens
            + weights["specificity"] * spec
            + weights["accuracy"] * acc
            + weights["f1_scores"] * f1
        )
        composite_scores.append(composite_score)

    return {
        "thresholds": thresholds,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "f1_scores": f1_scores,
        "precision": precision,
        "recall": recall,
        "composite_scores": composite_scores
    }

def find_optimal_threshold(metrics):
    """
    Find the optimal threshold based on maximizing F1-score.

    Parameters:
    - metrics: Dictionary containing thresholds, sensitivity, specificity, accuracy, and F1-score values.

    Returns:
    - optimal_threshold: Threshold that maximizes F1-score.
    """
    optimal_idx = np.argmax(metrics["composite_scores"])
    print(f'----------------- Maximum accuracy is {np.max(metrics["composite_scores"])} -----------------')
    return metrics["thresholds"][optimal_idx]

def plot_metrics(metrics):
    """
    Plot sensitivity, specificity, accuracy, and F1-score against thresholds.

    Parameters:
    - metrics: Dictionary containing thresholds, sensitivity, specificity, accuracy, and F1-score values.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot each metric
    plt.plot(metrics["thresholds"], metrics["sensitivity"], label="Sensitivity", marker="o")
    plt.plot(metrics["thresholds"], metrics["specificity"], label="Specificity", marker="o")
    plt.plot(metrics["thresholds"], metrics["accuracy"], label="Accuracy", marker="o")
    plt.plot(metrics["thresholds"], metrics["f1_scores"], label="F1-Score", marker="o")

    # Labeling
    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.title("Metrics vs Threshold")
    plt.legend()
    plt.grid(alpha=0.4)
    plt.savefig(f'threshold_analysis{random.randint(1, 100)}.png', dpi=600)
    plt.show()
    

def find_threshold_closest_sensitivity_specificity(metrics):
    """
    Find the threshold where sensitivity and specificity are closest to each other.

    Parameters:
    - metrics: Dictionary containing thresholds, sensitivity, and specificity values.

    Returns:
    - closest_threshold: Threshold where sensitivity and specificity are closest.
    - min_difference: The minimum difference between sensitivity and specificity.
    """
    differences = np.abs(np.array(metrics["sensitivity"]) - np.array(metrics["specificity"]))
    min_idx = np.argmin(differences)
    closest_threshold = metrics["thresholds"][min_idx]
    min_difference = differences[min_idx]
    return closest_threshold, min_difference


def compute_metrics(y_true, y_pred_scores, threshold = 0.5):
    if threshold != 0.5:
        print(f"---------------------- Threshold Changed to {threshold} ----------------------")
    labels = np.asarray(y_true).reshape(-1)
    pred_scores = np.asarray(y_pred_scores).reshape(-1)
    preds = pred_scores >= threshold
    
    # save labels, preds, and pred_scores to csv
    df = pd.DataFrame()
    df['labels'] = labels
    df['preds'] = preds
    df['pred_scores'] = pred_scores
    df.to_csv(os.path.join(MODEL_BASE_PATH,"predictions.csv"),index=False)
    
    metrics = {}
    metrics['accuracy'] = accuracy_score(labels, preds)
    metrics['average_precision'] = average_precision_score(labels, pred_scores)
    metrics['auroc'] = roc_auc_score(labels, pred_scores)
    metrics['f1_score'] = f1_score(labels, preds)
    metrics["TPR"] = accuracy_score(labels, preds)
    
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    #metrics["weighted_accuracy"] = ((tp/(tp+fp))+(tn/(tn+fn)))/2.0
    metrics["weighted_accuracy"] = (safe_divide(tp, tp + fp) + safe_divide(tn, tn + fn)) / 2.0
    '''
    True positive rate or recall or sensitivity: probability of identifying a positive case 
    (often called the power of a test)
    '''
    metrics['TPR'] = metrics['recall'] = metrics['sensitivity'] = recall_score(labels, preds)
    
    '''
    False positive rate: probability of falsely identifying someone as positive, who is actually negative
    '''
    metrics['FPR'] = fp / (fp+tn)
    
    '''
    Positive Predictive Value: probability that a patient with a positive test result 
    actually has the disease
    '''
    metrics['PPV'] = metrics['precision'] = precision_score(labels, preds , zero_division=0)
    
    '''
    Negative predictive value: probability that a patient with a negative test result 
    actually does not have the disease
    '''
    metrics['NPV'] = tn / (tn+fn)
    
    '''
    True negative rate or specificity: probability of a negative test result, 
    conditioned on the individual truly being negative
    '''
    metrics['TNR'] = metrics['specificity'] = tn / (tn+fp)
    
    return metrics

def evaluate(model, dataloader, reconstruction_loss_cfg, weights, thresholding_flag=False, threshold=0.5):
    all_preds = []
    all_labels = []
    results = {}

    total_loss = 0
    pred_loss_criterion = torch.nn.BCELoss()
    projection_loss_criterion = cosine_similarity
    reconstruction_loss_criterion = calculate_reconstruction_loss
    
    model.eval()

    with torch.no_grad():
        for i, (features, labels) in enumerate(dataloader):
            features = [feature.to(device) for feature in features]
            labels = labels.to(device)
            projected_features, preds, reconstructed_features = model(features)
            try:
                prediction_loss = pred_loss_criterion(preds, labels)
            except:
                
                # now we will write the classical features, their projections, and the reconstructed features to a file
                # the classical features are the first element of the features list
                
                print(features[0].shape)
                print(features[0][0].shape)
                print(str(features[0][0][0].item()))
                
                with open("classical_original_features.csv", "w") as f:
                    for i in range(features[0].shape[1]):
                        for j in range(features[0][i].shape[0]):
                            f.write(str(features[0][i][j].item()) + ",")
                        f.write("\n")
                        
                
                raise ValueError("Error in prediction loss")
                
            cosine_loss = projection_loss_criterion(projected_features)
            reconstruction_loss = reconstruction_loss_criterion(reconstructed_features, features, reconstruction_loss_cfg)
            
            # normalize the weights of the three losses
            weight_sum = weights["w_prediction_loss"] + weights["w_cosine_loss"] + weights["w_reconstruction_loss"]
            prediction_loss = weights["w_prediction_loss"]/weight_sum * prediction_loss
            cosine_loss = weights["w_cosine_loss"]/weight_sum * cosine_loss
            reconstruction_loss = weights["w_reconstruction_loss"]/weight_sum * reconstruction_loss
            
            
            total_loss = prediction_loss + cosine_loss + reconstruction_loss
            #total_loss = prediction_loss + cosine_loss
            all_preds.extend(preds.to('cpu').numpy())
            all_labels.extend(labels.to('cpu').numpy())

    if thresholding_flag:
        thresholds = np.linspace(0, 1, 101)  # Generate thresholds from 0 to 1
        
        weights = {
            "sensitivity": 1.0,
            "specificity": 1.0,
            "accuracy": 0,
            "f1_scores": 0,
        }
        
        metrics = calculate_metrics_at_thresholds(all_labels, all_preds, thresholds, weights=weights)
        optimal_threshold = find_optimal_threshold(metrics)
        
        closest_threshold, min_difference = find_threshold_closest_sensitivity_specificity(metrics)

        print(f"Threshold where sensitivity and specificity are closest: {closest_threshold}")
        print(f"Minimum difference between sensitivity and specificity: {min_difference}")
    

        print(f"Optimal Threshold: {optimal_threshold}")

        # Visualize the metrics
        plot_metrics(metrics)
        
        df_label_pred = pd.DataFrame({'all_labels': np.array(all_labels).flatten(), 'all_preds': np.array(all_preds).flatten()})
        df_label_pred.to_csv('all_labels_preds.csv', index=False)
        
        return closest_threshold

    results = compute_metrics(all_labels, all_preds, threshold=threshold)
    results["loss"] = total_loss.to('cpu').item() / len(dataloader.dataset)
    

    return results

@click.command()
@click.option("--model", default="ANN", help="Options: ANN, ShallowANN")
@click.option("--projection_dim", default=256, help="Projection dimension size")
@click.option("--normalization_dim", default=0, help="Normalization dimension; typically 0 or 1")
@click.option("--first_operation", default="normalize", help="First operation to perform on features")
@click.option("--reconstruction_loss", default="mse", help="Reconstruction loss; Options: mse, l1, huber, kl")
@click.option("--learning_rate", default=0.3674643450313223, help="Learning rate for classifier")
@click.option("--random_state", default=621, help="Random state for classifier")
@click.option("--seed", default=191, help="Seed for random number generation")
@click.option("--use_feature_scaling", default='yes', help="Apply feature scaling: yes or no")
@click.option("--scaling_method", default="StandardScaler", help="Method for feature scaling: StandardScaler, MinMaxScaler")
@click.option("--minority_oversample", default='no', help="Apply minority oversampling: yes or no")
@click.option("--batch_size", default=128, help="Batch size for training")
@click.option("--num_epochs", default=86, help="Number of epochs for training")
@click.option("--drop_correlated", default='yes', help="Drop correlated features: yes or no")
@click.option("--corr_thr", default=0.85, help="Threshold for dropping correlated features")
@click.option("--optimizer", default="SGD", help="Optimizer for training: SGD, AdamW")
@click.option("--beta1", default=0.9084719350261068, help="Beta1 value for optimizer")
@click.option("--beta2", default=0.9940871758715644, help="Beta2 value for optimizer")
@click.option("--weight_decay", default=0.0001, help="Weight decay rate for regularization")
@click.option("--momentum", default=0.8075456327084843, help="Momentum for optimizer")
@click.option("--use_scheduler", default='no', help="Use learning rate scheduler: yes or no")
@click.option("--scheduler", default="reduce", help="Type of scheduler: step, reduce")
@click.option("--step_size", default=17, help="Step size for step scheduler")
@click.option("--gamma", default=0.6033860204614545, help="Gamma value for scheduler")
@click.option("--patience", default=5, help="Patience for reduce scheduler")
@click.option("--w_prediction_loss", default=87, help="Weight for prediction loss")
@click.option("--w_cosine_loss", default=68, help="Weight for cosine loss")
@click.option("--w_reconstruction_loss", default=48, help="Weight for reconstruction loss")
@click.option("--FEATURES_FILE_1", default="imagebind_fox_features.csv", help="Path to the first feature file")
@click.option("--FEATURES_FILE_2", default="wavlm_fox_features.csv", help="Path to the second feature file")


def main(**cfg):
    ENABLE_WANDB = False
    if ENABLE_WANDB:
        wandb.init(project="multi_modal_audio", config=cfg, entity="roc-hci-audio-team")

    '''
    save the configurations obtained from wandb (or command line) into the model config file
    '''
    with open(MODEL_CONFIG_PATH,"w") as f:
        f.write(json.dumps(cfg))

    '''
    Ensure reproducibility of randomness
    '''
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"]) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    oversample = SMOTE(random_state = cfg['random_state'])

    if cfg["drop_correlated"]=='no':
        drop_correlated = False
    else:
        drop_correlated = True
    
    print(cfg)
    
    # load the dataset ==> For both feature files 
    features_1, columns_1 = load(FEATURES_FILE_ROOT + cfg['features_file_1'], drop_correlated=drop_correlated, corr_thr=cfg["corr_thr"])
    features_2, columns_2 = load(FEATURES_FILE_ROOT + cfg['features_file_2'], drop_correlated=drop_correlated, corr_thr=cfg["corr_thr"]) 
        
    feature_list, labels, IDs = merge_features([features_1, features_2], [columns_1, columns_2])
    
    print("Printing the shape of the feature list and the labels")
    print(feature_list[0].shape, feature_list[1].shape, len(labels), len(IDs))
    
    # Feature train, test split 
    feature_train_list, labels_train, ids_train, feature_test_list, labels_test, ids_test, filenames_test = train_test_split(feature_list, labels, IDs)
    
    pd.DataFrame(filenames_test, columns=['Filename']).to_csv(MODEL_BASE_PATH + "/test_filenames.csv", index=False)
    
    ids_test_df = pd.DataFrame(columns=['ID', 'pd'])
    ids_test_df['ID'] = ids_test
    ids_test_df['pd'] = labels_test
    ids_test_df.to_csv(MODEL_BASE_PATH + "/test_ids.csv", index=False)
    

    print("Printing the shape of the feature train and test sets")
    print(len(feature_train_list[0]), len(labels_train), len(ids_train), len(feature_test_list[0]), len(labels_test), len(ids_test), len(set(ids_test)))
    
    ids_train_full = copy.deepcopy(ids_train)

    train_sets, labels_train, ids_train_final, dev_sets, labels_dev, ids_dev = train_dev_split(ids_train_full, labels_train, feature_train_list)
    
    print("Printing the shape of the train and dev sets")
    print(len(train_sets[0]), len(ids_train_final), len(labels_train), len(dev_sets[0]), len(ids_dev), len(labels_dev), len(set(ids_dev)))
    
    assert set(ids_train_final).intersection(set(ids_dev)) == set()
    print("No overlap between train and dev sets")
    
    assert set(ids_train_final).intersection(set(ids_test)) == set()
    print("No overlap between train and test sets")
    
    assert set(ids_dev).intersection(set(ids_test)) == set()
    print("No overlap between dev and test sets")
    
    
    X_train, X_dev, X_test = train_sets, dev_sets, feature_test_list
    
    y_train, y_dev, y_test = labels_train, labels_dev, labels_test
    
    used_scaler = None

    n_modality = len(X_train)
    
    
    for i in range(n_modality):
        X_train[i] = np.asarray(X_train[i])
        X_dev[i] = np.asarray(X_dev[i])
        X_test[i] = np.asarray(X_test[i])

    y_train = np.asarray(y_train)

    assert len(X_train) == len(X_dev) == len(X_test)
    

    if cfg['use_feature_scaling']=='yes':
        if cfg['scaling_method'] == 'StandardScaler':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        for i in range(n_modality):
            X_train[i] = scaler.fit_transform(X_train[i])
            X_dev[i] = scaler.transform(X_dev[i])
            X_test[i] = scaler.transform(X_test[i])
        pickle.dump(scaler, open(SCALER_PATH,"wb"))
        used_scaler = pickle.load(open(SCALER_PATH,'rb'))
        

    print("Printing the shape of the train set before oversampling")
    print(X_train[0].shape, X_train[1].shape, len(y_train))
    
    print("Printing the 0/1 count of the train set before oversampling")
    print(np.unique(y_train, return_counts=True))
    
    if cfg['minority_oversample']=='yes':
        # concatenate the features 
        X_train_concat = np.concatenate(X_train, axis=1)
        
        print("Printing the shape of the train set cat before oversampling")
        print(X_train_concat.shape, len(y_train))
        
        X_train_concat_oversampled, y_train = oversample.fit_resample(X_train_concat, y_train)
        
        print("Printing the shape of the train set cat after oversampling")
        print(X_train_concat_oversampled.shape, len(y_train))
        
        # split the features back into the original different shapes
        X_train = [X_train_concat_oversampled[:, :X_train[i].shape[1]] for i in range(n_modality)]
        
        print("Printing the shape of the train set after oversampling")
        print(X_train[0].shape, X_train[1].shape, len(y_train))
    
    print("Printing the 0/1 count of the train set after oversampling")
    print(np.unique(y_train, return_counts=True))
    
    for i in range(n_modality):
        X_train[i] = np.asarray(X_train[i])
        X_dev[i] = np.asarray(X_dev[i])
        X_test[i] = np.asarray(X_test[i])

    y_train = np.asarray(y_train)

    train_dataset = Prjection_Dataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)    
    
    train_dataset = Prjection_Dataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    dev_dataset = Prjection_Dataset(X_dev, y_dev)
    dev_loader = DataLoader(dev_dataset, batch_size=cfg["batch_size"])
    test_dataset = Prjection_Dataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size = cfg['batch_size'])
    
    model = None
    dimensions = [X_train[i].shape[1] for i in range(n_modality)]

    print("Printing the dimensions of the features")
    print(dimensions)
    
    if cfg['model']=="ANN":
        model = Projection_ANN(dimensions, 0)
    elif cfg['model']=="ShallowANN":
        model = Projection_Shallow_ANN(dimensions, 0)
    else:
        raise ValueError("Invalid model")
    
    
    model = model.to(device)
    
    # print(model)
    
    if cfg["optimizer"]=="AdamW":
        optimizer = torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],betas=(cfg['beta1'],cfg['beta2']),weight_decay=cfg['weight_decay'])
    elif cfg["optimizer"]=="SGD":
        optimizer = torch.optim.SGD(model.parameters(),lr=cfg['learning_rate'],momentum=cfg['momentum'],weight_decay=cfg['weight_decay'])
    else:
        raise ValueError("Invalid optimizer")


    if cfg["use_scheduler"]=="yes":
        if cfg['scheduler']=="step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'], gamma=cfg['gamma'])
        elif cfg['scheduler']=="reduce":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=cfg['gamma'], patience = cfg['patience'])
        else:
            raise ValueError("Invalid scheduler")

    # we will perform the training only for a single data point
    # to verify that the model is working as expected

    # set the loss criterion
    pred_loss_criterion = nn.BCELoss()
    projection_loss_criterion = cosine_similarity
    reconstruction_loss_criterion = calculate_reconstruction_loss

    best_dev_loss = np.inf
    best_model = copy.deepcopy(model)


    list_predicition_loss = []
    list_cosine_loss = []
    list_reconstruction_loss = []
    list_total_loss = []
    list_train_loss = []
    list_dev_loss = []

    for epoch in tqdm(range(cfg['num_epochs'])):
        model.train()
        for index, (features, label) in enumerate(train_loader):
            
            features = [feature.to(device) for feature in features]
            label = label.to(device)
            projected_features, output, reconstructed_features = model(features)
            # calculate the loss 
            # there will three types of loss
            # 1. BCE loss for the prediction and the target
            # 2. Cosine similarity loss for the projected features
            # 3. reconstruction loss for the original features
            optimizer.zero_grad()
            prediction_loss = pred_loss_criterion(output, label)
            cosine_loss = projection_loss_criterion(projected_features)
            reconstruction_loss = reconstruction_loss_criterion(reconstructed_features, features, cfg['reconstruction_loss'])
            
            # normalize the weights of the three losses
            weight_sum = cfg["w_prediction_loss"] + cfg["w_cosine_loss"] + cfg["w_reconstruction_loss"]
            w_pred = cfg["w_prediction_loss"]/weight_sum
            w_cosine = cfg["w_cosine_loss"]/weight_sum
            w_recon = cfg["w_reconstruction_loss"]/weight_sum
            # print(w_pred, w_cosine, w_recon, weight_sum)
            prediction_loss = w_pred * prediction_loss
            cosine_loss = w_cosine * cosine_loss
            reconstruction_loss = w_recon * reconstruction_loss
            
            # print(prediction_loss.item(), cosine_loss.item(), reconstruction_loss.item())
    
            total_loss = prediction_loss + cosine_loss + reconstruction_loss
            #total_loss = prediction_loss + cosine_loss
            total_loss.backward()
            optimizer.step()
            
            list_predicition_loss.append(prediction_loss.item())
            list_cosine_loss.append(cosine_loss.item())
            list_reconstruction_loss.append(reconstruction_loss.item())
            list_total_loss.append(total_loss.item())
            
            
            if ENABLE_WANDB:
                wandb.log({"Training Loss": total_loss.to("cpu").item(), "Prediction Loss": prediction_loss.to("cpu").item(), "Cosine Loss": cosine_loss.to("cpu").item(), "Reconstruction Loss": reconstruction_loss.to("cpu").item()})
            
            weights = {
                "w_prediction_loss":cfg["w_prediction_loss"],
                "w_cosine_loss":cfg["w_cosine_loss"],
                "w_reconstruction_loss":cfg["w_reconstruction_loss"]
            }
        dev_metrics = evaluate(model, dev_loader, cfg['reconstruction_loss'], weights=weights)
        dev_loss = dev_metrics['loss']
        dev_accuracy = dev_metrics['accuracy']
        dev_balanced_accuracy = dev_metrics['weighted_accuracy']
        dev_auroc = dev_metrics['auroc']
        dev_f1 = dev_metrics['f1_score']
        list_dev_loss.append(dev_loss)

        if cfg['use_scheduler']=="yes":
            if cfg['scheduler']=='step':
                scheduler.step()
            else:
                scheduler.step(dev_loss)
            
        if dev_loss < best_dev_loss:
            best_model = copy.deepcopy(model)
            best_dev_loss = dev_loss
            best_dev_accuracy = dev_accuracy
            best_dev_balanced_accuracy = dev_balanced_accuracy
            best_dev_auroc = dev_auroc
            best_dev_f1 = dev_f1

    # optimal_threshold = evaluate(best_model, dev_loader, cfg['reconstruction_loss'], weights=weights, thresholding_flag=True)
    
    # results = evaluate(best_model, test_loader, cfg['reconstruction_loss'], weights=weights, threshold=?optimal_threshold+0.01)
    
    results = evaluate(best_model, test_loader, cfg['reconstruction_loss'], weights=weights)

    # evaluate(best_model, test_loader, cfg['reconstruction_loss'], weights=weights, thresholding_flag=True)

    # plot all the losses
    plt.plot(list_predicition_loss, label='Prediction Loss')
    plt.plot(list_cosine_loss, label='Cosine Loss')
    plt.plot(list_reconstruction_loss, label='Reconstruction Loss')
    plt.plot(list_total_loss, label='Total Loss')
    plt.plot(list_dev_loss, label='Dev Loss')
    plt.legend()
    # plt.show()

    # if wandb is enabled, log the plot
    if ENABLE_WANDB:
        wandb.log({"Plot": wandb.Image(plt)})
    
    print(results)

    if ENABLE_WANDB:
        wandb.log(results)
        wandb.log({"dev_accuracy":best_dev_accuracy, "dev_balanced_accuracy":best_dev_balanced_accuracy, "dev_loss":best_dev_loss, "dev_auroc":best_dev_auroc, "dev_f1":best_dev_f1})


    '''
    Save best model
    '''
    # torch.save(best_model.to('cpu').state_dict(),MODEL_PATH)

    # dimensions = [X_test[i].shape[1] for i in range(n_modality)]

    '''
    Test whether the model can be loaded successfully
    '''
    # if cfg['model']=="ANN":
    #     loaded_model = Projection_ANN(dimensions, cfg['projection_dim'])
    # elif cfg['model']=="ShallowANN":
    #     loaded_model = Projection_Shallow_ANN(dimensions, cfg['projection_dim'])
    # else:
    #     raise ValueError("Invalid model")

    # loaded_model.load_state_dict(torch.load(MODEL_PATH))
    # loaded_model = loaded_model.to(device)
    # print(evaluate(loaded_model,test_loader, cfg['reconstruction_loss']))
    # print(cfg)
    # print(loaded_model)

if __name__ == "__main__":
    main()