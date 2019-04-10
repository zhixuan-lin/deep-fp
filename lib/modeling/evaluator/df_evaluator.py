import torch
import numpy as np

class DFEvaluator:
    def __init__(self):
        self.correct = 0
        self.total = 0
    
    def evaluate(self, data, targets, results):
        """
        DF evaluator. We will keep running average
        :param data: (B, D)
        :param targets: (B,), long tensor
        :param results: (B, C), where C is the number of classes
        """
        with torch.no_grad():
            # (B, N)
            predicted = torch.argmax(results, dim=1)
            correct = predicted == targets
            self.total += correct.size()[0]
            self.correct += torch.sum(correct).item()
    
    def clear(self):
        self.correct = 0
        self.total = 0
    
    def results(self):
        return self.correct / self.total if self.total else 0
    
class PREvaluator:
    """
    Evaluate precision and recall. The last class will be consider "false", and
    all other will be consider "true"
    """
    def __init__(self, num_class):
        self.num_class = num_class
        self.y_truth = np.array([])
        self.y_score = np.array([])
    
    def evaluate(self, data, targets, results):
        """
        DF evaluator. We will keep running average
        :param data: (B, D)
        :param targets: (B,), long tensor
        :param results: (B, C), where C is the number of classes
        """
        
        with torch.no_grad():
            # (B, ), predicted class
            predicted = torch.argmax(results, dim=1)
            y_score = results[:, predicted]
            # change unmonitored to 0, (B, )
            y_score[predicted == (self.num_class - 1)] = 0.0
            # (B, )
            y_truth = targets != (self.num_class - 1)
            
            self.y_score = np.hstack([self.y_score, y_score])
            self.y_truth = np.hstack([self.y_truth, y_truth])
            
    def results(self):
        return {
            'precision': self.tp / (self.tp + self.fp) if self.tp + self.fp else 0,
            'recall': self.tp / (self.tp + self.fn) if self.tp + self.fn else 0
        }
        
