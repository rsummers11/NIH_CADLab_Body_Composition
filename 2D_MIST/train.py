import os
import sys
import argparse
import torch
from torch import autocast, nn
from train_validation_loader import get_train_validate_loader, get_train_loader
from normalization import compute_dataset_statistics
from importlib import import_module
from helpers import empty_cache, dummy_context
from unet_res_mixed import resunet_mixed
import json
from datetime import date
from datetime import datetime
from logger import NetLogger
from loader import load_segmentation_model, load_checkpoint, save_checkpoint
from polylr import PolyLRScheduler
from dice import get_tp_fp_fn_tn
from loss import MyLoss
from time import time, sleep
from collate_outputs import collate_outputs
import math
import nibabel as nib
import shutil
import torch
import numpy as np
import scipy.ndimage
from tqdm import tqdm

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch, consistency = 0.1, consistency_rampup = 200.0):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * sigmoid_rampup(epoch, consistency_rampup)

def print_to_log_file(log_file_name, *args, also_print_to_console=True, add_timestamp=True):
    timestamp = time()
    dt_object = datetime.fromtimestamp(timestamp)

    if add_timestamp:
        args = (f"{dt_object}:", *args)

    successful = False
    max_attempts = 5
    ctr = 0
    while not successful and ctr < max_attempts:
        try:
            with open(log_file_name, 'a+') as f:
                for a in args:
                    f.write(str(a))
                    f.write(" ")
                f.write("\n")
            successful = True
        except IOError:
            print(f"{datetime.fromtimestamp(timestamp)}: failed to log: ", sys.exc_info())
            sleep(0.5)
            ctr += 1
    if also_print_to_console:
        print(*args)

def on_epoch_start(logger, current_epoch):
    logger.log('epoch_start_timestamps', time(), current_epoch)

def on_train_epoch_start(model, lr_scheduler, logger, log_file_name, optimizer, current_epoch):
    model.train()
    lr_scheduler.step(current_epoch)
    print_to_log_file(log_file_name, '')
    print_to_log_file(log_file_name, f'Epoch {current_epoch}')
    print_to_log_file(log_file_name, f"Current learning rate: {np.round(optimizer.param_groups[0]['lr'], decimals=5)}")
    # lrs are the same for all workers so we don't need to gather them in case of DDP training
    logger.log('lrs', optimizer.param_groups[0]['lr'], current_epoch)

def on_train_epoch_end(logger, train_outputs, current_epoch):
    outputs = collate_outputs(train_outputs)

    loss_here = np.mean(outputs['loss'])
    if np.isnan(loss_here):
        print('found problem')
    logger.log('train_losses', loss_here, current_epoch)

def on_validation_epoch_end(logger, current_epoch, val_outputs):
    outputs_collated = collate_outputs(val_outputs)
    tp = np.sum(outputs_collated['tp_hard'], 0)
    fp = np.sum(outputs_collated['fp_hard'], 0)
    fn = np.sum(outputs_collated['fn_hard'], 0)

    loss_here = np.mean(outputs_collated['loss'])

    global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
    mean_fg_dice = np.nanmean(global_dc_per_class)
    logger.log('mean_fg_dice', mean_fg_dice, current_epoch)
    logger.log('dice_per_class_or_region', global_dc_per_class, current_epoch)
    logger.log('val_losses', loss_here, current_epoch)

def on_epoch_end(logger, log_file_name, current_epoch, save_every, num_epochs, best_miou, model, optimizer, model_dir):
    logger.log('epoch_end_timestamps', time(), current_epoch)

    # todo find a solution for this stupid shit
    print_to_log_file(log_file_name, 'train_loss', np.round(logger.my_fantastic_logging['train_losses'][-1], decimals=4))
    print_to_log_file(log_file_name, 'val_loss', np.round(logger.my_fantastic_logging['val_losses'][-1], decimals=4))
    print_to_log_file(log_file_name, 'Pseudo dice', [np.round(i, decimals=4) for i in logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
    print_to_log_file(log_file_name, f"Epoch time: {np.round(logger.my_fantastic_logging['epoch_end_timestamps'][-1] - logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")


    # handling periodic checkpointing
    if (current_epoch + 1) % save_every == 0 and current_epoch != (num_epochs - 1):
        save_checkpoint(os.path.join(model_dir, 'checkpoint_latest.pth'), current_epoch,
                        model, optimizer, logger, best_miou)

    # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
    if best_miou is None or logger.my_fantastic_logging['ema_fg_dice'][-1] > best_miou:
        best_miou = logger.my_fantastic_logging['ema_fg_dice'][-1]
        print_to_log_file(log_file_name, f"Yayy! New best EMA pseudo Dice: {np.round(best_miou, decimals=4)}")
        save_checkpoint(os.path.join(model_dir, 'checkpoint_best.pth'), current_epoch,
                        model, optimizer, logger, best_miou)

    logger.plot_progress_png(model_dir)
    return best_miou

def on_train_end(model_dir, current_epoch, model, optimizer, logger, best_miou, device):

    # dirty hack because on_epoch_end increments the epoch counter and this is executed afterwards.
    # This will lead to the wrong current epoch to be stored
    save_checkpoint(os.path.join(model_dir, "checkpoint_final.pth"), current_epoch,
                    model, optimizer, logger, best_miou)

    # now we can delete latest
    if os.path.isfile(os.path.join(model_dir, "checkpoint_latest.pth")):
        os.remove(os.path.join(model_dir, "checkpoint_latest.pth"))

    empty_cache(device)
    print("Training done.")

def train_step(strong_data, weak_data, model, optimizer, loss_fn, epoch, device):
    strong_images = strong_data['img'].float()
    strong_masks = strong_data['seg']
    strong_images = strong_images.to(device, non_blocking=True)
    if isinstance(strong_masks, list):
        strong_masks = [i.to(device, non_blocking=True) for i in strong_masks]
    else:
        strong_masks = strong_masks.to(device, non_blocking=True)

    weak_images = weak_data['img'].float()
    weak_masks = weak_data['seg']
    weak_images = weak_images.to(device, non_blocking=True)
    if isinstance(weak_masks, list):
        weak_masks = [i.to(device, non_blocking=True) for i in weak_masks]
    else:
        weak_masks = weak_masks.to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    # Autocast is a little bitch.
    # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
    # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
    # So autocast will only be active if we have a cuda device.
    # with autocast(device.type, enabled=True) if device.type == 'cuda' else dummy_context():
    strong_outputs = model(strong_images)
    weak_outputs = model(weak_images)
    loss = loss_fn(outputA=strong_outputs[0], outputB_F=strong_outputs[1],
                   outputB_W=weak_outputs[1], label_F=strong_masks,
                   label_W=weak_masks)
    weak_weight = get_current_consistency_weight(epoch//10)
    l = loss[0] + weak_weight * (0.01 * loss[1] + 0.01 * loss[2] + 1 * loss[3] + 1 * loss[4])
    # print('\n loss_0 {}, loss_1 {}, loss_2 {}, loss_3 {}, loss_4 {}'.format(loss[0], loss[1], loss[2],
    #                                                                         loss[3], loss[4]))
    if math.isnan(l):
        print('found NAN value')

    l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
    optimizer.step()
    # print('training loss {}'.format(l.detach().cpu().numpy()))
    return {'loss': l.detach().cpu().numpy()}

def validation_step(strong_data, weak_data, model, loss_fn, n_classes, ignore_label,
                    epoch, device):
    strong_images = strong_data['img'].float()
    strong_masks = strong_data['seg']
    strong_images = strong_images.to(device, non_blocking=True)
    if isinstance(strong_masks, list):
        strong_masks = [i.to(device, non_blocking=True) for i in strong_masks]
    else:
        strong_masks = strong_masks.to(device, non_blocking=True)

    weak_images = weak_data['img'].float()
    weak_masks = weak_data['seg']
    weak_images = weak_images.to(device, non_blocking=True)
    if isinstance(weak_masks, list):
        weak_masks = [i.to(device, non_blocking=True) for i in weak_masks]
    else:
        weak_masks = weak_masks.to(device, non_blocking=True)

    # Autocast is a little bitch.
    # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
    # If the device_type is 'mps' then it will complain that mps is not implemented,
    # even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
    # So autocast will only be active if we have a cuda device.
    # with autocast(device.type, enabled=True) if device.type == 'cuda' else dummy_context():
    strong_outputs = model(strong_images)
    weak_outputs = model(weak_images)
    del strong_images
    del weak_images
    loss = loss_fn(outputA=strong_outputs[0], outputB_F=strong_outputs[1],
                   outputB_W=weak_outputs[1], label_F=strong_masks,
                   label_W=weak_masks)
    weak_weight = get_current_consistency_weight(epoch // 10)
    l = loss[0] + weak_weight * (0.01 * loss[1] + 0.01 * loss[2] + 1 * loss[3] +
                                 1 * loss[4])

    # we only need the output with the highest output resolution
    # output = output[0]
    # labels = labels[0]

    # the following is needed for online evaluation. Fake dice (green line)
    axes = [0] + list(range(2, strong_outputs[0].ndim))

    if n_classes == 1:
        predicted_segmentation_onehot = (torch.sigmoid(strong_outputs[0]) > 0.5).long()
    else:
        # no need for softmax
        # output_seg = output.argmax(1)[:, None]
        # output_seg= torch.argmax(torch.softmax(output, 1), dim=1)
        # predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
        # predicted_segmentation_onehot.scatter_(1, output_seg, 1)
        predicted_segmentation_onehot = torch.softmax(strong_outputs[0], 1)
        # del output_seg

    if ignore_label is not None:
        if n_classes == 1:
            label = (strong_outputs[0] != ignore_label).float()
            # CAREFUL that you don't rely on target after this line!
            strong_outputs[0][strong_outputs[0] == ignore_label] = 0
        else:
            label = 1 - strong_outputs[0][:, -1:]
            # CAREFUL that you don't rely on target after this line!
            strong_outputs[0] = strong_outputs[0][:, :-1]
    else:
        label = None

    tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, strong_masks, axes=axes, mask=label)

    tp_hard = tp.detach().cpu().numpy()
    fp_hard = fp.detach().cpu().numpy()
    fn_hard = fn.detach().cpu().numpy()
    if n_classes != 1:
        # if we train with regions all segmentation heads predict some kind of foreground. In conventional
        # (softmax training) there needs tobe one output for the background. We are not interested in the
        # background Dice
        # [1:] in order to remove background
        tp_hard = tp_hard[1:]
        fp_hard = fp_hard[1:]
        fn_hard = fn_hard[1:]

    # print('\nloss {}, tp_hard {}, fp_hard {}, fn_hard {}'.format(l.detach().cpu().numpy(), tp_hard, fp_hard, fn_hard))
    return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

def create_weak_loader(weak_input_dir, start_epoch, config, data_statistics, retrain_flag):
    original_input_dir = weak_input_dir
    parent_path = os.path.dirname(weak_input_dir)
    dir_name = os.path.basename(os.path.normpath(weak_input_dir))
    updated_weak_dir_list = [d for d in os.listdir(parent_path)
                             if dir_name in d and dir_name != d]
    updated_weak_dir_list.sort(reverse=True)
    if (start_epoch + 1) >= config.get('updated_interval', 10) and not retrain_flag:
        if len(updated_weak_dir_list) != 0:
            weak_input_dir = updated_weak_dir_list[0]
        # updated_dir = os.path.join(parent_path, dir_name + '_updated')
    else:
        for d in updated_weak_dir_list:
            shutil.rmtree(os.path.join(parent_path, d))

    if len(os.listdir(weak_input_dir)) == 0:
        weak_input_dir = original_input_dir

    return get_train_validate_loader(root_dir=weak_input_dir,
                                     img_statistics=data_statistics,
                                     scale_factor=config.get('scale', 1),
                                     num_classes=config.get('num_classes', 1),
                                     batch_size=config.get('batch_size', 1),
                                     validation_ratio=config.get('val_ratio', 0.1),
                                     shuffle=True, num_workers=0, pin_memory=False)

def compute_dice_coefficent(prediction1, prediction2):
    axes = list(range(2, len(prediction1.shape)))
    intersect = (prediction1 * prediction2).sum(axes)
    sum_pred = prediction1.sum(axes)
    sum_gt = prediction2.sum(axes)
    dc = (2 * intersect + 1.0) / (torch.clip(sum_gt + sum_pred + 1.0, 1e-8))
    dc = dc.mean()
    return dc

def update_weak_labels(weak_input_dir, data_statistics, model, epoch, config, append_str='', removing_ratio = 0.2):
    parent_path = os.path.dirname(os.path.normpath(weak_input_dir))
    if 'subparent_dirname' in config.keys():
        parent_path = os.path.join(parent_path, config.get('subparent_dirname')+append_str)
        os.makedirs(parent_path, exist_ok=True)
    # print('weak_input_dir {} \n and \n parent_dir {}'.format(weak_input_dir, parent_path))
    dir_name = os.path.basename(os.path.normpath(weak_input_dir))

    updated_dir = os.path.join(parent_path, dir_name + '_epochs_' + "{:05d}".format(epoch))
    # print('updated_dir {}'.format(updated_dir))
    os.makedirs(updated_dir, exist_ok=True)

    #     remove all subfolds
    for c in os.listdir(updated_dir):
        full_path = os.path.join(updated_dir, c)
        if os.path.isfile(full_path):
            os.remove(full_path)
        else:
            shutil.rmtree(full_path)

    os.makedirs(os.path.join(updated_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(updated_dir, 'masks'), exist_ok=True)

    weak_train_loader  = get_train_loader(root_dir=weak_input_dir, img_statistics=data_statistics,
                                            scale_factor=config.get('scale', 1),
                                            batch_size=1,
                                            num_workers=0, pin_memory=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dice_dict = {}

    model.eval()
    with torch.no_grad():
        for (weak_data, weak_img_paths, weak_mask_paths) in weak_train_loader:
            weak_images = weak_data['img'].float().to(device)
            weak_outputs = model(weak_images)

            weak_outputs[0] = torch.softmax(weak_outputs[0], dim=1)
            weak_outputs[1] = torch.softmax(weak_outputs[1], dim=1)
            dice = compute_dice_coefficent(weak_outputs[0], weak_outputs[1])
            # dice_dict[weak_mask_paths[0]] = np.asscalar(dice.cpu().numpy())
            dice_dict[weak_mask_paths[0]] = dice.cpu().numpy().item()

    if 'removing_ratio' in config.keys():
        removing_rate = config.get('removing_ratio', 0.2)
    else:
        removing_rate = removing_ratio

    dice_dict = dict(sorted(dice_dict.items(), key=lambda item: item[1], reverse=True))
    removing_num = int(len(dice_dict) * removing_rate)
    for i in range(removing_num):
        dice_dict.popitem()

    # save prediction results..
    with torch.no_grad():
        for (weak_data, weak_img_paths, weak_mask_paths) in weak_train_loader:
            weak_images = weak_data['img'].float().to(device)
            # weakly_masks = weakly_data['seg'].float().to(device)
            weak_outputs = model(weak_images)

            weak_outputs[0] = torch.softmax(weak_outputs[0], dim=1)
            channel_num = weak_outputs[0].shape[1]
            weak_outputs[0] = torch.argmax(weak_outputs[0], dim=1).cpu().numpy()

            for i in range(weak_outputs[0].shape[0]):
                if weak_mask_paths[i] in dice_dict.keys():
                    # copy images..
                    image_base_name = os.path.basename(weak_img_paths[i])
                    shutil.copyfile(weak_img_paths[i], os.path.join(updated_dir, 'images', image_base_name))
                    mask_base_name = os.path.basename(weak_mask_paths[i])

                    ctNIB = nib.load(weak_img_paths[i])
                    ct = ctNIB.get_fdata()
                    affine = ctNIB.affine

                    tmp_img = scipy.ndimage.zoom(weak_outputs[0][i, ...], (1.0/config.get('scale', 1),
                                                                           1.0/config.get('scale', 1)), order=3)
                    tmp_img[tmp_img>=channel_num] = channel_num - 1
                    tmp_img[tmp_img < 0] = 0
                    seg_res_img = nib.Nifti1Image(tmp_img, affine, ctNIB.header)
                    nib.save(seg_res_img, os.path.join(updated_dir, 'masks', mask_base_name))

    return updated_dir

def train_dual_branch(strong_input_dir, weak_input_dir, config, model_dir):
    if not os.path.isdir(strong_input_dir) or not os.path.isdir(weak_input_dir):
        print('Either of {} and {} is not existed'.format(strong_input_dir, weak_input_dir))
        return
    else:
        print(weak_input_dir, strong_input_dir)

    print(strong_input_dir, weak_input_dir)
    # no re-training anymore because the network is too complicated...

    config_file_name = os.path.splitext(config)[0]
    config = getattr(import_module('config.' + config), 'config')
    print(config)

    data_statistics = compute_dataset_statistics(os.path.join(strong_input_dir, 'images'),
                                                 os.path.join(strong_input_dir, 'masks'))
    with open(os.path.join(model_dir, 'training_data_statistics.txt'), 'w') as convert_file:
        convert_file.write(json.dumps(data_statistics))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    empty_cache(device)
    model = resunet_mixed(in_class=1, out_class=config.get('num_classes', 1))
    model = model.to(device)

    # if torch.cuda.is_available():
    #     model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    print(model)

    timestamp = datetime.now()
    os.makedirs(model_dir, exist_ok=True)
    log_file_name = os.path.join(model_dir, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                                 (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                                  timestamp.second))
    logger = NetLogger()

    best_miou = None
    initial_lr = 1e-3
    weight_decay = 3e-5
    # optimizer = torch.optim.SGD(model.parameters(), initial_lr, weight_decay=weight_decay,
    #                             momentum=0.99, nesterov=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)

    start_epoch = 0
    weak_train_loader, weak_valid_loader, weak_sample_num = create_weak_loader(weak_input_dir=weak_input_dir,
                                                                               start_epoch=start_epoch,
                                                                               config=config,
                                                                               data_statistics=data_statistics,
                                                                               retrain_flag=True)

    strong_train_loader, strong_valid_loader, _ = get_train_validate_loader(
        root_dir=strong_input_dir, img_statistics=data_statistics, scale_factor=config.get('scale', 1),
        num_classes=config.get('num_classes', 1), batch_size=config.get('batch_size', 1),
        validation_ratio=config.get('val_ratio', 0.1),
        shuffle=True, num_workers=0, full_img_num=weak_sample_num, pin_memory=False)

    epochs = config.get('num_epochs', 1000)
    lr_scheduler = PolyLRScheduler(optimizer, initial_lr, epochs)
    loss_fn = MyLoss(n_classes=config.get('num_classes', 1))
    frozen_iter = config.get('frozen_iter', 100)
    assert epochs > frozen_iter, 'The iteration number to froze the network ' \
                                 'is no less than the total of training epochs'

    # train the model with both encoder and decoder
    for epoch in range(start_epoch, frozen_iter):
        on_epoch_start(logger, epoch)

        ##Training
        on_train_epoch_start(model, lr_scheduler, logger, log_file_name, optimizer, epoch)
        train_outputs = []

        pbar = tqdm(desc='Training batch progress: ', total=len(weak_train_loader), ncols=100)
        for (strong_data, weak_data) in zip(strong_train_loader, weak_train_loader):
            train_outputs.append(train_step(strong_data, weak_data, model, optimizer, loss_fn, epoch, device))
            pbar.update(1)
        pbar.close()
        on_train_epoch_end(logger, train_outputs, epoch)

        with torch.no_grad():
            model.eval()
            val_outputs = []
            pbar = tqdm(desc='Validation batch progress: ', total=len(weak_valid_loader), ncols=100)
            for (strong_data, weak_data) in zip(strong_valid_loader, weak_valid_loader):
                val_outputs.append(validation_step(strong_data, weak_data, model, loss_fn,
                                                   config.get('num_classes', 1), None,
                                                   epoch, device))
                pbar.update(1)
            pbar.close()

            on_validation_epoch_end(logger, epoch, val_outputs)

        best_miou = on_epoch_end(logger, log_file_name, epoch, config.get('save_every', 5), epochs,
                                 best_miou, model, optimizer, model_dir)
        if (epoch + 1) % config.get('updated_interval', 10) == 0:
            updated_weak_dir = update_weak_labels(weak_input_dir, data_statistics, model, epoch, config)
            weak_train_loader, weak_valid_loader, weak_sample_num = get_train_validate_loader(
                root_dir=updated_weak_dir, img_statistics=data_statistics, scale_factor=config.get('scale', 1),
                num_classes=config.get('num_classes', 1), batch_size=config.get('batch_size', 1),
                validation_ratio=config.get('val_ratio', 0.1), shuffle=True, num_workers=0, pin_memory=False)
            strong_train_loader, strong_valid_loader, _ = get_train_validate_loader(
                root_dir=strong_input_dir, img_statistics=data_statistics, scale_factor=config.get('scale', 1),
                num_classes=config.get('num_classes', 1), batch_size=config.get('batch_size', 1),
                validation_ratio=config.get('val_ratio', 0.1), shuffle=True, num_workers=0,
                full_img_num=weak_sample_num, pin_memory=False)

    # OK, let's do transfer learning and freeze the encoder..
    model.froze_network()
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)

    # optimizer = torch.optim.Adam('adam')(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    # Transfer learning here...
    for epoch in range(frozen_iter, epochs):
        on_epoch_start(logger, epoch)

        ##Training
        on_train_epoch_start(model, lr_scheduler, logger, log_file_name, optimizer, epoch)
        train_outputs = []

        pbar = tqdm(desc='Training batch progress: ', total=len(weak_train_loader), ncols=100)
        for (strong_data, weak_data) in zip(strong_train_loader, weak_train_loader):
            train_outputs.append(train_step(strong_data, weak_data, model, optimizer, loss_fn, epoch, device))
            pbar.update(1)
        pbar.close()
        on_train_epoch_end(logger, train_outputs, epoch)

        with torch.no_grad():
            model.eval()
            val_outputs = []
            pbar = tqdm(desc='Validation batch progress: ', total=len(weak_valid_loader), ncols=100)
            for (strong_data, weak_data) in zip(strong_valid_loader, weak_valid_loader):
                val_outputs.append(validation_step(strong_data, weak_data, model, loss_fn,
                                                   config.get('num_classes', 1),
                                                   None, epoch, device))
                pbar.update(1)
            pbar.close()

            on_validation_epoch_end(logger, epoch, val_outputs)

        best_miou = on_epoch_end(logger, log_file_name, epoch, config.get('save_every', 5), epochs,
                                 best_miou, model, optimizer, model_dir)
        if (epoch + 1) % config.get('updated_interval', 10) == 0:
            updated_weak_dir = update_weak_labels(weak_input_dir, data_statistics, model, epoch, config)
            weak_train_loader, weak_valid_loader, weak_sample_num = get_train_validate_loader(
                root_dir=updated_weak_dir, img_statistics=data_statistics, scale_factor=config.get('scale', 1),
                num_classes=config.get('num_classes', 1), batch_size=config.get('batch_size', 1),
                validation_ratio=config.get('val_ratio', 0.1),
                shuffle=True, num_workers=0, pin_memory=False)
            strong_train_loader, strong_valid_loader, _ = get_train_validate_loader(
                root_dir=strong_input_dir, img_statistics=data_statistics, scale_factor=config.get('scale', 1),
                num_classes=config.get('num_classes', 1), batch_size=config.get('batch_size', 1),
                validation_ratio=config.get('val_ratio', 0.1),
                shuffle=True, num_workers=0, full_img_num=weak_sample_num, pin_memory=False)
    on_train_end(model_dir, epoch, model, optimizer, logger, best_miou, device)

def main():
    # ------------------------------- argument parsing --------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('-si', '--strong_input_dir', type=str,
                        help='Input directory containing fully-annotated training data in nii.gz format',
                        required=True)
    parser.add_argument('-wi', '--weak_input_dir', type=str,
                        help='Input directory containing automatically-generated training data in nii.gz format',
                        required=True)
    parser.add_argument('-c', '--config_file', type=str,
                        help='The configuration file name with training parameter settings',
                        required=True)
    parser.add_argument('-o', '--model_dir', type=str,
                        help='Model directory that saves model files',
                        required=True)
    args = parser.parse_args()
    train_dual_branch(args.strong_input_dir, args.weak_input_dir, args.config_file, args.model_dir)

if __name__ == '__main__':
    main()