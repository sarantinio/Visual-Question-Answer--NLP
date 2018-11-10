# coding: utf-8

"""
Deep CBOW (with minibatching)

Based on Graham Neubig's DyNet code examples:
  https://github.com/neubig/nn4nlp2017-code
  http://phontron.com/class/nn4nlp2017/

"""

import gzip
import json
import numpy as np
import h5py
import parameters
import random
import time
from collections import defaultdict
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sys

torch.manual_seed(1)
random.seed(1)


CUDA = torch.cuda.is_available()
print("CUDA: %s" % CUDA)


# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]
PAD = w2i["<pad>"]

# One data point
Example = namedtuple("Example", ["words", "tag", "type", "img_feat"])

def read_dataset(questions_path, annotations_path, image_features_path, img_features2id_path, imgid2imginfo_path):

    with open(imgid2imginfo_path, 'r') as file:
        imgid2info = json.load(file)

    # load image features from hdf5 file and convert it to numpy array
    img_features = np.asarray(h5py.File(image_features_path, 'r')['img_features'])

    # load mapping file
    with open(img_features2id_path, 'r') as f:
        visual_feat_mapping = json.load(f)['VQA_imgid2id']

    with gzip.GzipFile(questions_path, 'r') as file:
        questions = json.loads(file.read())

    with gzip.GzipFile(annotations_path, 'r') as file:
        annotations = json.loads(file.read())

    for line in range(len(questions['questions'])):
        words = questions['questions'][line]['question'].lower().strip()
        tag = annotations['annotations'][line]['multiple_choice_answer']
        type = annotations['annotations'][line]['answer_type']

        img_id = questions['questions'][line]['image_id']
        h5_id = visual_feat_mapping[str(img_id)]
        img_feat = img_features[h5_id]
        yield Example(words=[w2i[x] for x in words.split(" ")],
                      tag=t2i[tag],
                      type=type,
                      img_feat=img_feat)


# Read in the data
train = list(read_dataset( "data/vqa_questions_train.gzip",
                           "data/vqa_annotatons_train.gzip",
                            parameters.image_features_path,
                            parameters.img_features2id_path,
                            parameters.imgid2imginfo_path))

w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("data/vqa_questions_valid.gzip",
                        "data/vqa_annotatons_valid.gzip",
                        parameters.image_features_path,
                        parameters.img_features2id_path,
                        parameters.imgid2imginfo_path))

test = list(read_dataset("data/vqa_questions_test.gzip",
                        "data/vqa_annotatons_test.gzip",
                        parameters.image_features_path,
                        parameters.img_features2id_path,
                        parameters.imgid2imginfo_path))

nwords = len(w2i)
ntags = len(t2i)

class DeepLSTM(nn.Module):
    """
    Deep CBOW model
    """

    def __init__(self, vocab_size, embedding_dim, img_features_dim, output_dim, hidden_dims=[], activation_functions=[]):
        """
        :param vocab_size: Vocabulary size of the training set.
        :param embedding_dim: The word embedding dimension.
        :param output_dim: The output dimension, ie the number of classes.
        :param hidden_dims: A list of hidden layer sizes. Default: []
        :param transformations: A list of transformation functions.
        """
        super(DeepLSTM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=parameters.lstm_dim, num_layers=1, batch_first=True)
        self.hidden_num = len(hidden_dims)
        self.linears = {}
        if (self.hidden_num == 0):
            self.linear1 = nn.Linear(img_features_dim + parameters.lstm_dim, output_dim)
        else:
            self.linear1 = nn.Linear(img_features_dim + parameters.lstm_dim, hidden_dims[0])
            for i in range(1, self.hidden_num):
                name = "linear" + str(i + 1)
                self.add_module(name, nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            name = "linear" + str(self.hidden_num + 1)
            self.add_module(name, nn.Linear(hidden_dims[self.hidden_num - 1], output_dim))
        self.named_modules = dict(self.named_children())
        self.F = activation_functions

    def forward(self, words, image):
        embeds = self.embeddings(words)
        _, (_,c) = self.lstm(embeds)
        h = c.squeeze(0)
        h = torch.cat([image, h], dim=1)
        if(self.hidden_num == 0):
            h = self.linear1(h)
        else:
            for i in range(self.hidden_num):
                name = self.F[i]
                matrix = self.named_modules["linear"+str(i+1)](h)
                h = self.func_map(name, matrix)
            h = self.named_modules["linear"+str(self.hidden_num+1)](h)
        h = F.log_softmax(h)
        return h

    def func_map(self, name, matrix):
        if name == "relu":
            return F.relu(matrix)
        else:
            # Default
            return F.relu(matrix)

model = DeepLSTM(nwords,
                 parameters.embedding_dim,
                 parameters.img_features_dim,
                 ntags,
                 parameters.hidden_dims,
                 parameters.transformations)

if CUDA:
    model.cuda()

print(model)

def minibatch(data, batch_size=32):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]


def evaluate(model, data):
    """Evaluate a model on a data set."""

    correct_all = 0.0
    correct_yesno = 0.0
    correct_number = 0.0
    correct_other = 0.0

    size_yesno = 0
    size_number = 0
    size_other = 0

    for batch in minibatch(data, parameters.batch_size):

        seqs, tags, types, image = preprocess(batch)
        scores = model(get_variable(seqs), get_image(image))
        _, predictions = torch.max(scores.data, 1)
        targets = get_variable(tags)
        correct = torch.eq(predictions, targets)
        for i in range(len(correct)):
            if types[i] == 'yes/no':
                size_yesno+=1
                if correct.data[i] == 1:
                    correct_yesno += 1
            elif types[i] == 'number':
                size_number+=1
                if correct.data[i] == 1:
                    correct_number += 1
            elif types[i] == 'other':
                size_other+=1
                if correct.data[i]:
                    correct_other+=1

        correct_all += torch.eq(predictions, targets).sum().data[0]

    return correct_all, len(data), correct_all/len(data), correct_yesno/size_yesno, correct_number/size_number, correct_other/size_other


def get_variable(x):
    """Get a Variable given indices x"""
    tensor = torch.cuda.LongTensor(x) if CUDA else torch.LongTensor(x)
    return Variable(tensor)
def get_image(x):
    tensor = torch.cuda.FloatTensor(x) if CUDA else torch.FloatTensor(x)
    return Variable(tensor)


def preprocess(batch):
    """ Add zero-padding to a batch. """

    tags = [example.tag for example in batch]
    types = [example.type for example in batch]

    # add zero-padding to make all sequences equally long
    seqs = [example.words for example in batch]
    max_length = max(map(len, seqs))
    seqs = [seq + [PAD] * (max_length - len(seq)) for seq in seqs]
    img = [example.img_feat.tolist() for example in batch]
    return seqs, tags, types, img


optimizer = optim.Adam(model.parameters(), parameters.lr)

for ITER in range(parameters.epochs):

    random.shuffle(train)
    train_loss = 0.0
    start = time.time()
    updates = 0

    for batch in minibatch(train, parameters.batch_size):

        updates += 1

        # pad data with zeros
        seqs, tags, types, image = preprocess(batch)

        # forward pass
        scores = model(get_variable(seqs), get_image(image))
        targets = get_variable(tags)
        loss = nn.NLLLoss()
        # loss = nn.CrossEntropyLoss()
        output = loss(scores, targets)
        train_loss += output.data[0]

        # backward pass
        model.zero_grad()
        output.backward()

        # update weights
        optimizer.step()

    print("iter %r: avg train loss=%.4f, time=%.2fs" %
          (ITER, train_loss/updates, time.time()-start))

    # evaluate
    _, _, train_all, train_yesno, train_number, train_other = evaluate(model, train)
    _, _, dev_all, dev_yesno, dev_number, dev_other = evaluate(model, dev)
    print("iter %r: train all=%.4f yesno=%.4f number=%.4f other=%.4f" % (ITER, train_all, train_yesno, train_number, train_other))
    print("iter %r: valid all=%.4f yesno=%.4f number=%.4f other=%.4f" % (ITER, dev_all, dev_yesno, dev_number, dev_other))

_, _, test_all, test_yesno, test_number, test_other = evaluate(model, test)
print("test all=%.4f yesno=%.4f number=%.4f other=%.4f" % (test_all, test_yesno, test_number, test_other))
