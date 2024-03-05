import os 
from pickle import load
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
		yhat = model.predict([photo, sequence], verbose=0)
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
		references = [d.split() for d in captions_list]
		actual.append(references)
		predicted.append(yhat.split())
		print(yhat)
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
	


tokenizer = load(open('tokenizer.pkl', 'rb'))
test_dataset = load(open('test_dataset.pkl', 'rb'))
max_length = 34

folder_path = 'TF_models'
filename = os.path.join(folder_path, f'model_1.pth')

model = load_model(filename)
evaluate_model(model, test_dataset, tokenizer, max_length)
