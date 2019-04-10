import torch
from tqdm import tqdm

def test(model, device, dataloader, evaluator):
    """
    Test engine.
    
    :param model: model(data, targets) -> results
    :param dataloader: next(iter(dataloader)) -> data, targets
    :param evaluator: evaluator.evaluate(data, targets, results),
                      evaluator.clear()
                      evaluator.get_results() -> results
    :return: the results returned by evaluator
    """
    
    model = model.to(device)
    model.eval()
    pbar = tqdm(dataloader)
    for (data, targets) in pbar:
        data = data.to(device)
        targets = targets.to(device)
        results = model(data)
        evaluator.evaluate(data, targets, results)
        results = evaluator.results()
        if isinstance(results, float):
            pbar.set_description('Acc: {:.3f}'.format(evaluator.results()))
        else:
            pbar.set_description('prec: {:.3f}, recall: {:.3f}'.format(results['precision'], results['recall']))
            
        
    return evaluator.results()
    
    
