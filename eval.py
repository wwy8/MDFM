#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file function to evaluate a model

If you find any bug or have some suggestion, please, email me.
"""

import torch
import torch.nn as nn
import torch.nn.functional as nnF
from metrics import Metrics, accuracy

from classification_metrics import AVGMetrics
from tqdm import tqdm
from Mymodel import mymodel
import config
def metrics_for_eval (model, data_loader, device, loss_fn, topk=2, get_balanced_acc=False, get_auc=False):
    """
        This function returns accuracy, topk accuracy and loss for the evaluation partition

        :param model (nn.Model): the model you' want to evaluate
        :param data_loader (DataLoader): the DataLoader containing the data partition
        :param checkpoint_path(string, optional): string with a checkpoint to load the model. If None, none checkpoint is
        loaded. Default is None.
        :param loss_fn (nn.Loss): the loss function used in the training
        :param get_balanced_acc (bool, optional); if you want to compute the balanced accuracy for the eval partition
        set it as True. The default is False because it may impact in the training phase.
        :param get_auc (bool, optional); if you want to compute the AUC for the eval partition
        set it as True. The default is False because it may impact in the training phase.
        :return: a instance of the classe metrics
    """

    # setting the model to evaluation mode
    model.eval()

    print ("\nEvaluating...")
    # Setting tqdm to show some information on the screen
    with tqdm(total=len(data_loader), ascii=True, ncols=100) as t:

        # Setting require_grad=False in order to dimiss the gradient computation in the graph
        with torch.no_grad():

            loss_avg = AVGMetrics()
            acc_avg = AVGMetrics()
            topk_avg = AVGMetrics()

            opt_metrics = list()
            if not get_balanced_acc and not get_auc:
                metrics = None
            else:                
                if get_balanced_acc:
                    opt_metrics.append('balanced_accuracy')
                if get_auc:
                    opt_metrics.append('auc')                    
                metrics = Metrics(opt_metrics)


            print("Testing...")
            for data in data_loader:

                # In data we may have imgs, labels and extra info. If extra info is [], it means we don't have it
                # for the this training case. Imgs came in data[0], labels in data[1] and extra info in data[2]
                try:
                    images_batch, labels_batch, meta_data_batch, _ = data
                except ValueError:
                    images_batch, labels_batch = data
                    meta_data_batch = []
                batch_size = images_batch.size(0)
                if len(meta_data_batch):
                    # Moving the data to the deviced that we set above
                    images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)
                    meta_data_batch = meta_data_batch.to(device)
                    meta_data_batch = meta_data_batch.float()

                    # Doing the forward pass using meta_data
                    pred_batch = model(images_batch, meta_data_batch)
                    #_, pred_batch = torch.max(pred_batch, 1)
                else:
                    # Moving the data to the device that we set above
                    images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)

                    # Doing the forward pass without using meta-data
                    pred_batch, _ = model(images_batch)
                    #_, pred_batch = torch.max(pred_batch, 1)


                # Computing the loss
                L = loss_fn(pred_batch, labels_batch)
                # Computing the accuracy
                #acc = float(test_correct) / total
                acc, topk_acc = accuracy(pred_batch, labels_batch, topk=(1, topk))

                loss_avg.update(L.item())
                acc_avg.update(acc.item())
                topk_avg.update(topk_acc.item())

                if metrics is not None:
                    # Moving the data to CPU and converting it to numpy in order to compute the metrics
                    pred_batch_np = pred_batch.cpu().numpy()#nnF.softmax(pred_batch, dim=1).cpu().numpy()
                    labels_batch_np = labels_batch.cpu().numpy()
                    # updating the scores
                    metrics.update_scores(labels_batch_np, pred_batch_np)

                # Updating tqdm
                t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
                t.update()

    bal_acc, auc = None, None
    if metrics is not None:
        # Getting the metrics
        metrics.compute_metrics()
        try:
            bal_acc =  metrics.metrics_values['balanced_accuracy']
        except KeyError:
            bal_acc = None
        try:
            auc =  metrics.metrics_values['auc']
        except KeyError:
            auc = None

    return {"loss": loss_avg(), "accuracy": acc_avg(), "topk_acc": topk_avg(), "balanced_accuracy": bal_acc, 'auc': auc}


# Testing the model
def test_model (model, data_loader, checkpoint_path=None, loss_fn=None, device=None, save_pred=False,
                    partition_name='Test', metrics_to_comp=('accuracy'), class_names=None, metrics_options=None,
                    apply_softmax=False, verbose=True, full_path_pred=None):
    """
    This function evaluates a given model for a given data_loader

    :param model (nn.Model): the model you'd like to evaluate
    :param data_loader (DataLoader): the DataLoader containing the data partition
    :param checkpoint_path(string, optional): string with a checkpoint to load the model. If None, no checkpoint is
    loaded. Default is None.
    :param loss_fn (nn.Loss): the loss function used in the training
    :param partition_name (string): the partition name
    :param metrics (list, tuple or string): it's the variable that receives the metrics you'd like to compute.
        IMPORTANT: if metrics is None, only the prediction.csv will be generated. Default is only the accuracy.
    :param class_names (list, tuple): a list or tuple containing the classes names in the same order you use in the
        label. For ex: ['C1', 'C2']. For more information about the options, please, refers to
        jedy.pytorch.model.metrics.py
    :param metrics_ options (dict): this is a dict containing some options to compute the metrics. Default is None.
    For more information about the options, please, refers to jedy.pytorch.model.metrics.py
    :param device (torch.device, optional): the device to use. If None, the code will look for a device. Default is
    None. For more information about the options, please, refers to jedy.pytorch.model.metrics.py
    :param verbose (bool, optional): if you'd like to print information o the screen. Default is True
    True. Default is False.

    :return: a instance of the classe metrics
    """

    # setting the model to evaluation mode
    model.eval()

    def _get_predictions (model, images_batch, meta_data_batch=None):
        with torch.no_grad():
            if meta_data_batch is None:
                pred_batch = model(images_batch)
            else:
                pred_batch = model(images_batch, meta_data_batch)
        return pred_batch

    net = mymodel()
    ckpt = torch.load(config.resume)
    net.load_state_dict(state_dict=ckpt['net_state_dict'])
    model = net.cuda()

    if device is None:
        # Setting the device
        if torch.cuda.is_available():
            device = torch.device("cuda:" + str(torch.cuda.current_device()))
        else:
            device = torch.device("cpu")

    # Moving the model to the given device
    model.to(device)

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    # Setting the metrics object
    metrics = Metrics (metrics_to_comp, class_names, metrics_options)

    # Setting tqdm to show some information on the screen
    with tqdm(total=len(data_loader), ascii=True, ncols=100) as t:

        loss_avg = AVGMetrics()

        for data in data_loader:
            # In data we may have images, labels, meta-data, and image ID. If meta-data is [], it means we don't have it
            # for the this training case. Images came in data[0], labels in data[1], meta_data in data[2], and image_id
            # in data[3]
            try:
                images_batch, labels_batch, meta_data_batch, img_id = data#根据你的数据更改
            except ValueError:
                images_batch, labels_batch = data
                meta_data_batch = []
                img_id = None
            batch_size = images_batch.size(0)
            if len(meta_data_batch):
                # Moving the data to the device that we set above
                images_batch = images_batch.to(device)
                if len(labels_batch):
                    labels_batch = labels_batch.to(device)
                meta_data_batch = meta_data_batch.to(device)
                meta_data_batch = meta_data_batch.float()

                # Doing the forward pass using meta-data
                pred_batch = _get_predictions (model, images_batch, meta_data_batch)
                _, concat_predict = torch.max(pred_batch, 1)


            elif len(labels_batch):
                # Moving the data to the device that we set above
                images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)

                # Doing the forward pass without using meta-data
                pred_batch = _get_predictions(model, images_batch)
            else:
                # Moving the data to the deviced that we set above
                images_batch = images_batch.to(device)

                # Doing the forward pass without using meta_data
                pred_batch = _get_predictions(model, images_batch)

            # Computing the loss
            if len(labels_batch):
                L = loss_fn(pred_batch, labels_batch)
                loss_avg.update(L.item())
                labels_batch_np = labels_batch.cpu().numpy()
            else:
                labels_batch_np = None
                loss_avg.update(0)

            # Moving the data to CPU and converting it to numpy in order to compute the metrics
            if apply_softmax:
                pred_batch_np = nnF.softmax(pred_batch,dim=1).cpu().numpy()
            else:
                pred_batch_np = pred_batch.cpu().numpy()

            # updating the scores
            #print(pred_batch_np)
            metrics.update_scores(labels_batch_np, pred_batch_np, img_id)

            # Updating tqdm
            if metrics.metrics_names is None:
                t.set_postfix(loss='{:05.3f}'.format(0.0))
            else:
                t.set_postfix(loss='{:05.3f}'.format(loss_avg()))

            t.update()

    # Adding loss into the metric values
    metrics.add_metric_value("loss", loss_avg())

    # Getting the metrics
    metrics.compute_metrics()

    if save_pred or metrics.metrics_names is None:
        if full_path_pred is None:
            metrics.save_scores()
        else:
            _spt = full_path_pred.split('/')
            _folder = "/".join(_spt[0:-1])
            _p = _spt[-1]
            metrics.save_scores(folder_path=_folder, pred_name=_p)

    if verbose:
        print('- {} metrics:'.format(partition_name))
        metrics.print()


    return metrics.metrics_values

def get_labels_frequency (data_csv, col_target, col_single_id, verbose=False):
    """
    This function returns the frequency of each label in the dataset
    :param data_csv (string or pd.DataFrame): the path for a csv or a dataframe already loaded
    :param col_target (string): the name of the target/label column
    :param col_single_id (string): the name any column that is present for all rows in the dataframe
    :param verbose (boolean): a boolean to print or not the frequencies
    return (pd.DataFrame): a dataframe containing the frequency of each label
    """

    # Loading the data_csv
    if isinstance(data_csv, str):
        data_csv = pd.read_csv(data_csv)

    data_ = data_csv.groupby([col_target])[col_single_id].count()
    if (verbose):
        print('### Data summary: ###')
        print(data_)
        print(">> Total samples: {} <<".format(data_.sum()))

    return data_

import os
from torch.autograd import Variable
import torch.utils.data
from torch.nn import DataParallel
from loader import get_loader
from Mymodel import mymodel
from utils import init_log, progress_bar
import config
import pandas as pd
# Loading the csv file
_folder = 1
_base_path = config.base_path  # 这里都是加载数据的过程  换成你自己的
_csv_path_train = os.path.join(_base_path, "pad-ufes-20_parsed_folders.csv")
csv_all_folders = pd.read_csv(_csv_path_train)

print("-" * 50)
print("- Loading validation data...")
val_csv_folder = csv_all_folders[ (csv_all_folders['folder'] == _folder) ]
train_csv_folder = csv_all_folders[ csv_all_folders['folder'] != _folder ]
ser_lab_freq = get_labels_frequency(train_csv_folder, "diagnostic", "img_id")
_labels_name = ser_lab_freq.index.values
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
trainloader, testloader = get_loader()#这里改成你自己的dataloader  从第312行到这里都是加载数据 都需要换成你自己的
model = mymodel()#这里换成你的model
ckpt = torch.load(config.resume)#这里换成加载你pth文件的代码
model.load_state_dict(state_dict=ckpt['net_state_dict'])#这里换成加载你pth文件的代码
model = model.cuda()
_save_folder = config.metrics_dir#设置一个你保存指标的路径
os.makedirs(_save_folder)
_metric_options = {
    'save_all_path': os.path.join(_save_folder, "best_metrics"),
    'pred_name_scores': 'predictions_best_test.csv',
    'normalize_conf_matrix': True}
test_model (model, testloader, checkpoint_path=None, loss_fn=None, save_pred=True,
            partition_name='eval', metrics_to_comp='all', class_names=_labels_name, metrics_options=_metric_options,
            apply_softmax=True, verbose=False)

