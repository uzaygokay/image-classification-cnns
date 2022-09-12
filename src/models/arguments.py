#%%imports
import argparse
import torch

#%%args

def get_args():
    
    parser = argparse.ArgumentParser(description='Arguments for image classification')

    #data module related
    parser.add_argument('-data_dir', type=str, help= 'Directory for downloaded dataset',
        default='./../../data/external'
    )

    parser.add_argument('-batch_size', type=int, help= 'Batch size',
        default=64
    )

    parser.add_argument('-num_workers', type=int, help= 'Number of workers',
        default=4
    )

    parser.add_argument('-max_epochs', type=int, help= 'Number of workers',
        default=3
    )

    #model related
    parser.add_argument('-hidden_size', type=int, help= 'Hidden size of network',
        default=64
    )

    parser.add_argument('-learning_rate', type=float, help= 'Learning rate',
        default=2e-4
    )

    parser.add_argument('-weight_decay', type=float, help= 'Weight Decay',
        default=1e-5
    )

    parser.add_argument('-dropout', type=float, help= 'Dropout rate',
        default=0.1
    )
    
    parser.add_argument("-gpus", type = int, help = "Number of GPUs", 
        default = torch.cuda.device_count()
    )

    args = parser.parse_args()

    return args