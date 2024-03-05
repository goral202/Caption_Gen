import numpy as np
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
import os 
from sklearn.model_selection import train_test_split
from model_Torch import CustomModel
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None


def generate_caption(model, tokenizer, photo, max_length):
	in_text = 'startseq'
	for i in range(max_length):
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		sequence = pad_sequences([sequence], maxlen=max_length)
		photo = torch.tensor(photo, dtype=torch.float).to(device)
		sequence = torch.tensor(sequence, dtype=torch.long).to(device)
		yhat = model(photo,sequence).detach().numpy()
		yhat = np.argmax(yhat)
		word = word_for_id(yhat, tokenizer)
		if word is None:
			break
		in_text += ' ' + word
		if word == 'endseq':
			break
	return in_text


def evaluate_model(model, dataset, tokenizer, max_length):
	actual, predicted = list(), list()
	for key in dataset.keys():
		photo = dataset[key]['features']
		captions_list = dataset[key]['captions']
		yhat = generate_caption(model, tokenizer, photo, max_length)
		# store actual and predicted
		references = [d.split() for d in captions_list]
		actual.append(references)
		predicted.append(yhat.split())
		print(yhat)
	# calculate BLEU score
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
	


tokenizer = load(open('tokenizer.pkl', 'rb'))
test_dataset = load(open('test_dataset.pkl', 'rb'))
folder_path = 'Torch_models_4'
max_length = 34
filename = os.path.join(folder_path, f'model_49.pth')
wagi = torch.load(filename,map_location=torch.device(device))
model = CustomModel(8572, 34)
model.load_state_dict(wagi)
model.eval()
evaluate_model(model, test_dataset, tokenizer, max_length)
