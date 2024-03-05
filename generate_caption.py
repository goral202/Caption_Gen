from pickle import load
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from model_Torch import CustomModel
import torch
import os
import numpy as np
import cv2


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def extract_features(filename):
	model = VGG16()
	model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
	image = load_img(filename, target_size=(224, 224))
	image = img_to_array(image)
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	image = preprocess_input(image)
	feature = model.predict(image, verbose=0)
	return feature


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


max_length = 34
photo_dict = '/home/jakub/HOME/Datasets/Flickr8k/Images'

folder_path = 'Torch_models_3'
filename = os.path.join(folder_path, f'model_29.pth')

tokenizer = load(open('tokenizer.pkl', 'rb'))
test_dataset = load(open('test_dataset.pkl', 'rb'))

wagi = torch.load(filename,map_location=torch.device(device))
model = CustomModel(8572, 34)
model.load_state_dict(wagi)
model.eval()


for image_name in test_dataset.keys():
	photo_path = os.path.join(photo_dict, image_name)
	photo = test_dataset[image_name]['features']
	caption = generate_caption(model, tokenizer, photo, max_length)
	caption = caption.replace('startseq', '').replace('endseq', '')
	image = cv2.imread(photo_path)
	cv2.imshow(caption, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows() 