#%%imports
import argparse
import torch

#%%args

def get_args():
    
    parser = argparse.ArgumentParser(description='Arguments for image classification')

    parser.add_argument('-data_dir', type=str, help= 'Directory for downloaded dataset',
        default='./../../data/external'
    )

    parser.add_argument('-batch_size', type=int, help= 'Batch size',
        default=20
    )

    parser.add_argument('-num_workers', type=int, help= 'Number of workers',
        default=2
    )