image_features_path = './download/VQA_image_features.h5'
img_features2id_path = './download/VQA_img_features2id.json'
imgid2imginfo_path = './data/imgid2imginfo.json'

# General parameters
batch_size = 128
lr = 0.001
epochs = 100

# Embedding parameters
embedding_dim = 64

# LSTM parameters
lstm_dim = 128

# Image feature parameters
img_features_dim = 2048

# Classifier parameters
hidden_dims = [100]
transformations = ["relu"]

#max number of iterations in order to get the dev accuracy imrpoved again , or the loss dev (?) decreased again.
threshold_val = 5
