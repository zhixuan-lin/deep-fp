from collections import defaultdict
from collections import deque

import torch

class SmoothedValue:
    """
    Track a series of values and provide access to smoothed values over a window
    or the global series average.
    """
    
    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0
        
    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value
        
    @property
    def median(self):
        """
        Return the median of current window
        """
        d = torch.tensor(list(self.deque))
        return d.median().item()
    
    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()
    
    @property
    def global_avg(self):
        return self.total / self.count
    
class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        
    def update(self, **kargs):
        for k, v in kargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)
            
    def state_dict(self):
        state = {}
        for key in self.meters:
            state[key] = self.meters[key].median
        
        return state
    
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribue '{}'".format(
            type(self).__name__, attr))
    
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f}".format(name, meter.median)
            )
            
        return self.delimiter.join(loss_str)
