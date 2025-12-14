from dotenv import load_dotenv
from neuralop.models import FNO 
from neuralop.data.datasets import load_darcy_flow_small 
from neuralop.training import Trainer
import torch
import torch.optim as optim
import torch.nn as nn 
import os

f"""
Training Data:
- Shape: (300, 26, 101, 101) in form of (trajectory, frames, x, y)


Task: 

1. Train an FNO to use 5 timeframes as input and output 3 timeframes
2. Plot a qualitative plot with three visual plots: ground truth, prediction, absolute error
3. Plots from left to right represent phase field model time evolutions

"""

model = FNO(
    n_models=(32,32),
    hidden_channels=64,
    in_channels=2,
    out_channels=1
)

model.save_checkpoint(save_folder='./checkpoints', save_name='example_fno')

train_loader, test_loaders, data_processor = load_darcy_flow_small(
    n_train=1000,
    batch_size=32,
    n_tests=[100],
    test_resolutions=[32],
    test_batch_sizes=[32]
)

trainer = Trainer(
    model=model,
    n_epochs=3,
    data_processor=data_processor,
    verbose=True
)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

train_loss = nn.MSELoss()

eval_losses = {
    'mse': nn.MSELoss(),
    'mae': nn.L1Loss()
}

trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses
)