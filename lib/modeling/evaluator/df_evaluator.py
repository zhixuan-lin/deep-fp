import torch


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
