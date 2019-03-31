import torch

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
    i = 0
    for (data, targets) in dataloader:
        print(i)
        data = data.to(device)
        targets = targets.to(device)
        results = model(data)
        evaluator.evaluate(data, targets, results)
        i += 1
        if i > 50:
            break
        
    return evaluator.results()
    
    
