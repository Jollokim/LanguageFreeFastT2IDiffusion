from pytorch_fid.fid_score import save_fid_stats, calculate_fid_given_paths
import torch
import os
import numpy as np


def create_ref(in_dataset: str, out_file: str, batch_size: int, inception_features: int = 2048, device: str = 'cpu', num_workers: int = 1):
    """
    Generate reference fid statistics (μ, σ) for a dataset and save them to an output npz file.

    Parameters:
    in_dataset (str): Path to the input dataset.
    out_file (str): Path to the output file where fid stats will be saved.
    batch_size (int): Batch size for processing the dataset.
    inception_features (int, optional): Number of features expected from the Inception model (default is 2048).
    device (str, optional): Device to run the feature extraction (default is 'cpu').
    num_workers (int, optional): Number of worker processes for data loading (default is 1).

    Returns:
    None

    """
    save_fid_stats([in_dataset, out_file], batch_size, device, inception_features, num_workers)


    


if __name__ == '__main__':
    # set up number of workers and devices
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    try:
        num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        # os.sched_getaffinity is not available under Windows, use
        # os.cpu_count instead (which may not return the *available* number
        # of CPUs).
        num_cpus = os.cpu_count()

    num_workers = min(num_cpus, 8) if num_cpus is not None else 0

    batch_size = 16
    inception_features = 2048

    in_dataset = 'data/MM_CelebA_HQ/images/faces'
    out_file = 'data/MM_CelebA_HQ/all_fid_ref.npz'
    
    create_ref(in_dataset, out_file, batch_size, inception_features, device, num_workers)