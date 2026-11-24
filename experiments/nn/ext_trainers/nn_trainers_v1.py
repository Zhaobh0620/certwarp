"""The one trainer of work package NN that nnunetv2 2.8.1 does not have (nnUNet_extTrainer points at this directory; nothing in the environment nnunet is changed). Written as nnUNetTrainer_250epochs_NoMirroring is, in nnunetv2/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_Xepochs_NoMirroring.py, with 100 epochs. Why no mirroring: experiments/nn/nn_common.py."""
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_100epochs_NoMirroring(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes
