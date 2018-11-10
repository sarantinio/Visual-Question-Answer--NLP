import gzip
from collections import defaultdict

import h5py
import numpy as np
import json
import os
from collections import defaultdict
import time
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim


current_path = os.getcwd()
path_to_h5_file   = 'download/VQA_image_features.h5'
path_to_json_file = 'download/VQA_img_features2id.json'

w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]
CUDA = torch.cuda.is_available()

def read_dataset(annotations_path, questions_path):
    with open(current_path + '/data/imgid2imginfo.json', 'r') as file:
        imgid2info = json.load(file)

    # load image features from hdf5 file and convert it to numpy array
    img_features = np.asarray(h5py.File(path_to_h5_file, 'r')['img_features'])

    # load mapping file
    with open(path_to_json_file, 'r') as f:
        visual_feat_mapping = json.load(f)['VQA_imgid2id']

    with gzip.GzipFile(annotations_path, 'r') as file:
        annotations = json.loads(file.read())

    with gzip.GzipFile(questions_path, 'r') as file:
        questions = json.loads(file.read())

    for line in range(len(questions['questions'])):
        most_common_answer = annotations['annotations'][line]['multiple_choice_answer']
        words = questions['questions'][line]['question'].lower().strip()
        img_id = questions['questions'][line]['image_id']
        h5_id = visual_feat_mapping[str(img_id)]
        img_feat = img_features[h5_id]
        yield ([w2i[x] for x in words.split(" ")], t2i[most_common_answer], img_feat)

# Read in the data
train = list(read_dataset("data/vqa_annotatons_train.gzip", "data/vqa_questions_train.gzip"))
w2i = defaultdict(lambda: UNK, w2i)
test = list(read_dataset("data/vqa_annotatons_test.gzip", "data/vqa_questions_test.gzip"))
nwords = len(w2i)
nanswers = len(t2i)


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, feature_dim, output_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(feature_dim+embedding_dim, output_dim)

    def forward(self, words, image):
        embeds = self.embeddings(words)
        bow = torch.sum(embeds, 1)
        bow = torch.cat([image, bow], dim=1)
        logits = self.linear(bow)
        return logits


model = CBOW(nwords, 64, 2048, nanswers)
print(model)


def evaluate(model, data):
    """Evaluate a model on a data set."""
    correct = 0.0

    for words, tag, img_feat in data:
        image = img_feat.tolist()
        image_features = Variable(torch.FloatTensor([image]))
        lookup_tensor = Variable(torch.LongTensor([words]))
        scores = model(lookup_tensor, image_features)
        predict = scores.data.numpy().argmax(axis=1)[0]

        if predict == tag:
            correct += 1

    return correct, len(data), correct / len(data)



optimizer = optim.SGD(model.parameters(), lr=0.01)
print(CUDA)
if CUDA :
    model.cuda()
for ITER in range(100):

    random.shuffle(train)
    train_loss = 0.0
    start = time.time()

    for words, tag, img_feat in train:
        image = img_feat.tolist()
        # forward pass
        lookup_tensor = Variable(torch.LongTensor([words]))
        image_features = Variable(torch.FloatTensor([image]))
        scores = model(lookup_tensor, image_features)
        loss = nn.CrossEntropyLoss()
        target = Variable(torch.LongTensor([tag]))
        output = loss(scores, target)
        train_loss += output.data[0]

        # backward pass
        model.zero_grad()
        output.backward()

        # update weights
        optimizer.step()

    print("iter %r: train loss/sent=%.4f, time=%.2fs" %
          (ITER, train_loss/len(train), time.time()-start))

    # evaluate
    _, _, acc = evaluate(model, test)
    print("iter %r: test acc=%.4f" % (ITER, acc))
