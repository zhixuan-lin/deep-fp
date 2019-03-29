import os
import torch

class Checkpointer:
    """
    Checkpointer that save and load model, optimizer, scheduler states, and any
    other arguments
    """
    def __init__(
            self,
            model,
            optimizer=None,
            scheduler=None,
            save_dir="",
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        
    def save(self, name, **kargs):
        """
        Save model, optimizer, scheduler and all kargs to "name.pth"
        """
        if not self.save_dir:
            return
        
        # save model, optimizer, scheduler and other arguments
        data = {}
        data['model'] = self.model.state_dict()
        if self.optimizer is not None:
            data['optimizer'] = self.optimizer
        if self.scheduler is not None:
            data['scheduler'] = self.scheduler
        # save any other arguments
        data.update(kargs)
        
        save_file = os.path.join(self.save_dir, '{}.pth'.format(name))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)
    
    def load(self, f=None):
        if self.has_checkpoint():
            # there is a checkpoint
            f = self.get_checkpoint_file()
        if not f:
            print("No checkpoint found.")
            return {}
        print("Loading checkpoint from {}".format(f))
        # load the checkpoint dictionary
        checkpoint = self._load_file(f)
        
        # load model weight to the model
        self._load_model(checkpoint)
        if 'optimizer' in checkpoint and self.optimizer:
            # if there is a optimizer, load state dict into the optimizer
            print("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop('optimizer'))
        if 'scheduler' in checkpoint and self.scheduler:
            # if there is a optimizer, load state dict into it
            print("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop('scheduler'))
            
        # there might be other arguments to be saved
        return checkpoint
    
    def has_checkpoint(self):
        # if there is at least one checkpoint, the file 'last_checkpoint' will
        # exist
        save_file = os.path.join(self.save_dir, 'last_checkpoint')
        return os.path.exists(save_file)
    
    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, 'last_checkpoint')
        # return last checkpoint name
        return open(save_file).read().strip()
    
    def tag_last_checkpoint(self, last_filename):
        """
        Tag the last checkpoint file 'last_filename' to a text file
        'last_checkpoint'
        """
        save_file = os.path.join(self.save_dir, 'last_checkpoint')
        with open(save_file, 'w') as f:
            f.write(last_filename)
            
    def _load_file(self, f):
        return torch.load(f, map_location=torch.device('cpu'))
    
    def _load_model(self, checkpoint):
        torch.load_state_dict(self.model, checkpoint)
        
    
        
    
