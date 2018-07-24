# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

import data
import model

parser = argparse.ArgumentParser(description='PyTorch dialog segmentation Language Model')
parser.add_argument('--data', type=str, default='./data/movie_dialogs',
                    help='location of the data corpus')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--model', type=str, default='model.pt',
                    help='load saved model')
parser.add_argument('--seg_batch_size', type=int, default=1, 
                    help='batch size for segmentation')
parser.add_argument('--bptt', type=int, default=5,
                    help='sequence length')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
args = parser.parse_args()



ori_file = "test.txt"
segmented_file = "segmented_dialog.txt"
BREAK = "<BRK>"

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

corpus = data.Corpus(args.data)

"""TODO: don't need this function if batch size is 1"""
def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

test_data = batchify(corpus.test, args.seg_batch_size)

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def load_model():
    with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()
    print(model)
    return model

def compare(model, line):
    model.eval()
    hidden = model.init_hidden(args.seg_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)

def prepare_data(dictionary, line):
    words = line.split() + ['<eos>']
    ids = torch.LongTensor(len(words))
    for word in words:
        if words in dictionary:
            ids[token] = dictionary.word2idx[word]
        else:
            ids[token] = dictionary.word2idx["<unk>"]
        token += 1
    ids = batchify(ids, args.seg_batch_size)
    return ids

def to_word(idx_list, dictionary):
    return " ".join(dictionary.idx2word[i] for i in idx_list)

def segment():
    model = load_model()
    model.rnn.flatten_parameters()
    # Tokenize file content
    with open(path, 'r', encoding="utf8") as f:
        model.eval()
        token = 0
        dictionary = model.dictionary
        ntokens = len(dictionary)
        for line in f:
            if len(line.split()) < 4:
                return_line = line
            else:
                data = prepare_data(dictionary, line)
                hidden = model.init_hidden(args.seg_batch_size)
                result_list = [data[0]]
                with torch.no_grad():
                    output, hidden = model(data, hidden)
                    output_flat = output.view(-1, ntokens)
                    for i in range(len(output_flat)-1):
                        result_list.append(data[i+1])
                        if output_flat(data[i+1]) < output_flat(dictionary.word2idx[BREAK]):
                            result_list.apend(dictionary.word2idx[BREAK])
                return_line = to_word(result_list)
            print("original line")
            print(line)
            print("processed")
            print(return_line)








