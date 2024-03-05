from model_TF import define_model
from load_dataset import load_captions, extract_features, clean_captions, create_word_set, captions_to_tokens, data_generator
from sklearn.model_selection import train_test_split
import os
from pickle import dump, load
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dataset_not_ready = False


def custom_loss(predictions, targets):
    epsilon = 0# 1e-5
    loss = -np.log(predictions + epsilon) * targets
    summed_loss = np.sum(loss, axis=1)
    mean_loss = np.mean(summed_loss)
    return mean_loss


if dataset_not_ready:
  dataset_path = '/home/jakub/HOME/Datasets/Flickr8k/captions.txt'
  dataset = load_captions(dataset_path)

  directory = '/home/jakub/HOME/Datasets/Flickr8k/Images'
  full_dataset = extract_features(directory, dataset)

  clean_captions(full_dataset)

  keys = list(full_dataset.keys())
  train_keys, test_keys = train_test_split(keys, test_size=0.1, random_state=42)
  train_dataset = {key: full_dataset[key] for key in train_keys}
  test_dataset = {key: full_dataset[key] for key in test_keys}
  print(f'Train set: {len(train_dataset)}, Test set: {len(test_dataset)}')
  
  tokenizer = captions_to_tokens(train_dataset)
  dump(tokenizer, open('tokenizer.pkl', 'wb'))
  dump(train_dataset, open('train_dataset.pkl', 'wb'))
  dump(test_dataset, open('test_dataset.pkl', 'wb'))

else:
    tokenizer = load(open('tokenizer.pkl', 'rb'))
    train_dataset = load(open('train_dataset.pkl', 'rb'))
    test_dataset = load(open('test_dataset.pkl', 'rb'))

word_set, max_length, number_of_captions = create_word_set(train_dataset)
vocab_size = len(tokenizer.word_index) + 1
generator = data_generator(train_dataset, tokenizer, max_length, vocab_size)
model = define_model(vocab_size, max_length)
epochs = 100
steps = number_of_captions

for i in range(epochs):
  data_in, output = next(generator) 
  model.fit(data_in, output, epochs=1, steps_per_epoch=1, verbose=1)
  pred = model.predict(data_in)
  model.save('TF_models\\model_' + str(i) + '.h5')