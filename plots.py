import csv
import sys
import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import traceback


def get_csv_data(csv_file, labels, space_separated=False):
    data, all_labels = get_csv_data_and_labels(csv_file,
                                               space_separated=space_separated)
    # # Uncomment to print the labels
    # print(csv_file)
    # for label in all_labels:
    #     print(label)
    # print('***\n'*3)
    n_data = data.shape[0]

    new_data = np.zeros((len(labels), n_data))

    # # Uncomment for debugging
    # print(all_labels)

    for ll, name in enumerate(labels):
        if name in all_labels:
            idx = all_labels.index(name)
            try:
                new_data[ll, :] = data[:, idx]
            except Exception:
                print(traceback.format_exc())
                print("Error with data in %s" % csv_file)
                sys.exit(1)
        else:
            raise ValueError("Label '%s' not available in file '%s'"
                             % (name, csv_file))

    return new_data


def get_csv_data_and_labels(csv_file, space_separated=False):
    # Read from CSV file
    try:
        if space_separated:
            series = pd.read_csv(csv_file, delim_whitespace=True)
        else:
            series = pd.read_csv(csv_file)
    except Exception:
        print(traceback.format_exc())
        print("Error reading %s" % csv_file)
        sys.exit(1)

    data = series.values
    labels = list(series)

    return data, labels


def plot_eval_returns(csv_file, block=False):
    labels = ['Test Returns Mean']

    data = get_csv_data(csv_file, labels)

    fig, axs = plt.subplots(1)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    fig.subplots_adjust(hspace=0)
    fig.suptitle('Average Return', fontweight='bold')

    for aa, ax in enumerate(axs):
        ax.plot(data[aa])
        ax.set_ylabel(labels[aa])
        plt.setp(ax.get_xticklabels(), visible=False)

    axs[-1].set_xlabel('Episodes')
    plt.setp(axs[-1].get_xticklabels(), visible=True)

    print('total_iters:', len(data[-1]))
    plt.show(block=block)


def plot_action_means(csv_file,  block=False):
    labels = []
    adim = get_max_action_idx(csv_file, "Mean Action ") + 1

    for ii in range(adim):
        labels.append('Mean Action %02d' % ii)

    data = get_csv_data(csv_file, labels)

    fig, axs = plt.subplots(adim)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    fig.subplots_adjust(hspace=0)
    fig.suptitle('Mean Action', fontweight='bold')

    for aa, ax in enumerate(axs):
        ax.plot(data[aa])
        ax.set_ylabel('Action %02d' % aa)
        plt.setp(ax.get_xticklabels(), visible=False)

    axs[-1].set_xlabel('Episodes')
    plt.setp(axs[-1].get_xticklabels(), visible=True)
    plt.show(block=block)


def plot_action_stds(csv_file, block=False):
    labels = []
    adim = get_max_action_idx(csv_file, "Mean Action ") + 1
    for ii in range(adim):
        labels.append('Std Action %02d' % ii)

    data = get_csv_data(csv_file, labels)

    fig, axs = plt.subplots(adim)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    fig.subplots_adjust(hspace=0)
    fig.suptitle('Std Action', fontweight='bold')

    for aa, ax in enumerate(axs):
        ax.plot(data[aa])
        ax.set_ylabel('Action %02d' % aa)
        plt.setp(ax.get_xticklabels(), visible=False)

    axs[-1].set_xlabel('Episodes')
    plt.setp(axs[-1].get_xticklabels(), visible=True)
    plt.show(block=block)


def plot_policy_info(csv_file, block=False):
    labels = [
        'Alpha',
        'Entropy',
    ]
    data = get_csv_data(csv_file, labels)

    fig, axs = plt.subplots(len(labels))
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    fig.subplots_adjust(hspace=0)
    fig.suptitle('Policy Info', fontweight='bold')

    for aa, ax in enumerate(axs):
        ax.plot(data[aa])
        ax.set_ylabel(labels[aa])
        plt.setp(ax.get_xticklabels(), visible=False)

    axs[-1].set_xlabel('Episodes')
    plt.setp(axs[-1].get_xticklabels(), visible=True)

    print('total_iters:', len(data[-1]))
    plt.show(block=block)


def get_headers(csv_file):
    with open(csv_file, 'r') as f:
        d_reader = csv.DictReader(f)

        # get fieldnames from DictReader object and store in list
        headers = d_reader.fieldnames

    return headers


def get_max_action_idx(csv_file, header_label=None):
    if header_label is None:
        header_label = 'Mean Action '
    max_value = max([int(header.split(header_label, 1)[1])
                     for header in get_headers(csv_file)
                     if header.startswith(header_label)])
    return max_value
