from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from tqdm import tqdm
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np


def load_captions(file_path):
  dataset = {}
  with open(file_path, 'r') as file:
    lines = file.readlines()[1:]
    for line in lines:
      name, caption = line.strip().split(',', 1)
      if name not in dataset:
        dataset[name] = {'captions': [], 'features': []}
      dataset[name]['captions'].append(caption)
  return dataset


def extract_features(directory, dataset):
  model = VGG16()
  model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
  print(model.summary())
  for name in tqdm(dataset.keys(), desc="Extracting features", unit="image"):
    filename = directory + '/' + name
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    feature = model.predict(image, verbose=0)
    dataset[name]['features'] = feature
  return dataset


def show_captions(dataset):
  for key in dataset.keys():
    for caption in dataset[key]['captions']:
      print(caption)


def clean_captions(dataset):
  translator = str.maketrans('', '', string.punctuation)
  for key in dataset.keys():
    for i, caption in enumerate(dataset[key]['captions']):
      text = caption
      text_without_punctuation = text.translate(translator)
      lower_text_without_punctuation = text_without_punctuation.lower()
      words = lower_text_without_punctuation.split()
      filtered_words = [word for word in words if len(word) > 1]
      filtered_words_2 = [word for word in filtered_words if word.isalpha()]
      processed_text = 'startseq ' + ' '.join(filtered_words_2) + ' endseq '
      dataset[key]['captions'][i] = processed_text


def create_word_set(dataset):
  word_set = set()
  max_length = 0
  number_of_captions = 0
  for key in dataset.keys():
    for caption in dataset[key]['captions']:
      number_of_captions +=1
      words = caption.split()
      word_set.update(words)
      if len(words)>max_length:
        max_length = len(words)
  return word_set, max_length, number_of_captions


def captions_to_tokens(dataset):
  tokenizer = Tokenizer()
  for key in dataset.keys():
    if 'tokens' not in dataset[key].keys():
      dataset[key]['tokens'] = []
    lines = dataset[key]['captions']
    tokenizer.fit_on_texts(lines)
  return tokenizer


def data_generator(dataset, tokenizer, max_length, vocab_size):
  while 1:
    for key in dataset.keys():
      photo = dataset[key]['features'][0]
      captions_list = dataset[key]['captions']
      for caption in captions_list:
        in_img, in_seq, out_word = create_sequences(tokenizer, max_length, caption, photo, vocab_size)
        yield [in_img, in_seq], out_word


def create_sequences(tokenizer, max_length, caption, photo, vocab_size):
  X1, X2, y = list(), list(), list()
  seq = tokenizer.texts_to_sequences([caption])[0]
  for i in range(1, len(seq)):
    in_seq, out_seq = seq[:i], seq[i]
    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
    X1.append(photo)
    X2.append(in_seq)
    y.append(out_seq)
  return np.array(X1), np.array(X2), np.array(y)