"""The nnU-Net trainers of work package R3 (runs/20261111-r3/plan.md, section N): NN's nnUNetTrainer_250epochs_NoMirroring (nnunetv2 2.8.1,
variants/training_length/nnUNetTrainer_Xepochs_NoMirroring.py; why no mirroring: experiments/nn/nn_common.py) with the seed of the initial weights fixed.

Why a seed (the decorrelation of the two models of one fold): nnU-Net seeds nothing.  Its data stream is drawn in augmentation workers that seed themselves from the OS (NonDetMultiThreadedAugmenter,
seeds=None), and torch's default generator, from which initialize() builds the network, is seeded anew at every process start (torch 2.5.1 in the environments sci310 and nnunet: torch.initial_seed()
differs between two starts, checked 11-11).  So two runs of the stock trainer are already decorrelated -- in their initial weights and their data -- but nothing of it is recorded, and two starts are not
guaranteed to differ.  Here torch, numpy and random are seeded right before the network is built: the initial weights of the four models come from four distinct seeds (0, 1 for fold A; 2, 3 for fold B;
r3_common.NN_SEEDS), recorded in the class name and the training log; the data stream stays NN's (OS-seeded workers).  The class name carries the seed, so each model has its own results directory
(<trainer>__nnUNetPlans__3d_fullres/fold_<k>); --c (continue) seeds again and then loads the checkpoint, which replaces the weights.

nnUNet_extTrainer points at this directory (experiments/r3/nn_r3.py nn_env); nothing in the environment nnunet is changed.  Runs only in that environment.
"""
import random

import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs_NoMirroring import nnUNetTrainer_250epochs_NoMirroring


class _Seeded(nnUNetTrainer_250epochs_NoMirroring):
    SEED = None

    def initialize(self):
        if not self.was_initialized:
            random.seed(self.SEED)
            np.random.seed(self.SEED)
            torch.manual_seed(self.SEED)
            torch.cuda.manual_seed_all(self.SEED)
            self.print_to_log_file(f'R3: torch, numpy and random seeded with {self.SEED} before the network is built (experiments/r3/ext_trainers/nn_trainers_r3.py)')
        super().initialize()


class nnUNetTrainer_250epochs_NoMirroring_r3s0(_Seeded):
    SEED = 0


class nnUNetTrainer_250epochs_NoMirroring_r3s1(_Seeded):
    SEED = 1


class nnUNetTrainer_250epochs_NoMirroring_r3s2(_Seeded):
    SEED = 2


class nnUNetTrainer_250epochs_NoMirroring_r3s3(_Seeded):
    SEED = 3
