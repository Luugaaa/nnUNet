
"""
nnUNetTrainerWandb trainer class for nnUNet that integrates with Weights & Biases (wandb).
This class extends the nnUNetTrainer class to log training and validation metrics, images, and
other relevant information to wandb for visualization and tracking.

Inspired from https://github.com/MIC-DKFZ/nnUNet/issues/2733#issuecomment-2744210810
"""
from datetime import datetime
import matplotlib.pyplot as plt 
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from contextlib import nullcontext as dummy_context
import os
import torch
import wandb

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from torch._dynamo import OptimizedModule
from pathlib import Path

class nnUNetTrainerWandb(nnUNetTrainer):
    _example_input_size = None
    def run_training(self):
        
        self.on_train_start()
        output_path = os.path.join("output_path", str(datetime.now().date()) +"_" +str(datetime.now().time()))
        os.makedirs(output_path, exist_ok=True)

        wandb.init(project=f'model_seg_sc-gm-lesion_human_ms_exvivo_t2star',  dir=output_path)

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train),batch_id))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()
        wandb.finish()  
        self.on_train_end()

    def train_step(self, batch: dict, batch_id: int) -> dict:
        data = batch['data']
        target = batch['target']
        
        if self._example_input_size is None:
            self._example_input_size = list(data.shape)
            self.print_to_log_file(f"Captured example input size: {self._example_input_size}")


        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # del data
            l = self.loss(output, target)

            if batch_id == 0: 
                train_image= data[0].detach().cpu().squeeze().float().numpy()
                
                # Deal with region-based training
                if target[0].shape[1] == 1:
                    # Not region-based, just squeeze the class dimension
                    train_gt = target[0].detach().cpu().squeeze().float().numpy()[0]
                    train_pred = np.argmax(output[0].detach().cpu().squeeze().numpy(), axis=1)[0]                
                else:
                    # Region-based, we need to sum all classes along the class dimension
                    train_gt = target[0].detach().cpu().squeeze().float().numpy()[0]
                    train_gt = np.squeeze(np.sum(train_gt, axis=0))  # Sum across classes to get a single mask and squeeze it
                    train_pred = output[0].detach().cpu().squeeze().float().numpy()[0]
                    train_pred = np.squeeze(np.sum(train_pred, axis=0))  # Sum across classes to get a single mask and squeeze it
                fig = plot_single_slice(combined=train_image, gt=train_gt, pred=train_pred)
                wandb.log({"training images": wandb.Image(fig)})
                plt.close(fig)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}
    
    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                if self.is_ddp:
                    mod = self.network.module
                else:
                    mod = self.network
                if isinstance(mod, OptimizedModule):
                    mod = mod._orig_mod

                checkpoint = {
                    'network_weights': mod.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema,
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.my_init_kwargs,
                    'trainer_name': self.__class__.__name__,
                    'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                }
                torch.save(checkpoint, filename)
                
                if self._example_input_size is not None:
                    try:
                        dummy_input_size = [1] + self._example_input_size[1:]
                        dummy_input = torch.rand(dummy_input_size).to(next(mod.parameters()).device)
                        
                        traced_model = torch.jit.trace(mod, dummy_input)

                        p = Path(filename)
                        script_filename = p.with_name(f"{p.stem}_script.pt")
                        
                        traced_model.save(str(script_filename))
                        self.print_to_log_file(f'Saved TorchScript model to {script_filename}')

                    except Exception as e:
                        self.print_to_log_file(f'WARNING: Could not save TorchScript model. Error: {e}')
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')



def plot_single_slice(combined, gt, pred, debug=False):
    """
    Plot the image, ground truth, and prediction for a single slice.
    Assumes 2D inputs (H, W).
    """
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle('Image | Ground Truth | Prediction')

    axs[0].imshow(combined.T, cmap='gray')
    axs[0].set_title("Image")
    axs[0].axis('off')

    axs[1].imshow(gt.T, cmap='Reds')
    axs[1].set_title("Ground Truth")
    axs[1].axis('off')

    axs[2].imshow(pred.T, cmap='Blues')
    axs[2].set_title("Prediction")
    axs[2].axis('off')

    plt.tight_layout()
    return fig