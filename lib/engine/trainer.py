import datetime
import logging
import time

import torch
import torch.distributed as dist
from ..utils.metric_logger import MetricLogger


def train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    max_epochs,
    arguments,
):
    print("Start training")
    meters = MetricLogger(", ")
    # max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    max_iter = max_epochs * len(data_loader)
    model.train()
    
    start_training_time = time.time()
    # end: the end time of last iteration
    end = time.time()
    
    
    for iteration, (data, targets) in enumerate(data_loader, start_iter):
        # compute epoch num
        epoch = iteration // len(data_loader) + 1
        if epoch > max_epochs:
            checkpointer.save("model_final", **arguments)
            break
        # time used for loading data
        data_time = time.time() - end
        iteration = iteration + 1
        
        # iteration should be kept in the checkpointer
        arguments["iteration"] = iteration
        
        # step learning rate scheduler
        if scheduler:
            scheduler.step()
        
        # batch training
        # put data to device
        data = data.to(device)
        targets = targets.to(device)
        
        # get losses
        loss = model(data, targets)
        
        # sum all losses for bp
        # losses = sum(loss for loss in loss_dict.values())
        meters.update(loss=loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # time for one iteration
        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        
        # estimated seconds is number of iterations left / time per iteration
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        
        if iteration % 20 == 0 or iteration == max_iter:
            print(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "epoch: {epoch}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        # "max mem: {memeory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    epoch=epoch,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    # memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                )
            )
            
        # save model, optimizer, scheduler, and other arguments
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
            
        
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    
    print("Total training time: {} ({:.4f} s /it)".format(
        total_time_str, total_training_time / (max_iter)
    ))
