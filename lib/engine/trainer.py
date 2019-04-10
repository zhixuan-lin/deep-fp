import datetime
import logging
import time

import torch
import torch.distributed as dist
from ..utils.metric_logger import MetricLogger
from .eval import test


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
    tensorboard,
    
    dataloader_val=None,
    evaluator=None
):
    print("Start training")
    meters = MetricLogger(", ")
    start_epoch = arguments['epoch']
    start_iter = arguments['iteration']
    max_iter = max_epochs * len(data_loader)
    
    start_training_time = time.time()
    # end: the end time of last iteration
    end = time.time()

    first = True
    for epoch in range(start_epoch, max_epochs):
        epoch = epoch + 1
        model.train()
        
        # starting from where we drop
        # enumerator = enumerate(data_loader, start_iter if first else 0)
        for iteration, (data, targets) in enumerate(data_loader):
            # this is necessary for ensure the right number of epochs
            if iteration >= max_iter:
                break
                
            # compute epoch num
            # time used for loading data
            data_time = time.time() - end
            iteration = iteration + 1
            
            global_step = (epoch - 1) * len(data_loader) + iteration
            
            # iteration should be kept in the checkpointer
            # arguments['iteration'] = iteration
            
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
            eta_seconds = meters.time.global_avg * (max_iter - global_step)
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
                if not tensorboard is None:
                    
                    metric_dict = meters.state_dict()
                    tensorboard.update(**metric_dict)
                    tensorboard.add('train', global_step)

                
            # save model, optimizer, scheduler, and other arguments
            # global_step = epoch * len(data_loader) + iteration
        # if global_step % checkpoint_period == 0:
        # save model after each epoch
        arguments['epoch'] = epoch
        checkpointer.save("model_{:04d}".format(epoch), **arguments)
        # checkpointer.save("model_{:05d}_{:07d}".format(epoch, iteration), **arguments)
        
        # evaluate result after each epoch
        if not evaluator is None and not dataloader_val is None:
            results = test(model, device, dataloader_val, evaluator)
            print('Final accuracy: {}'.format(results))
            tensorboard.update(accuracy=results)
            
            
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    
    print("Total training time: {} ({:.4f} s /it)".format(
        total_time_str, total_training_time / (max_iter)
    ))
