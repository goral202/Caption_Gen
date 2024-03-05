from load_dataset import load_captions, extract_features, clean_captions, create_word_set, captions_to_tokens, data_generator
from sklearn.model_selection import train_test_split
from model_Torch import CustomModel, CustomLoss
import torch.optim as optim
import torch
from pickle import dump, load
from tqdm import tqdm
import os


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset_not_ready = False


if dataset_not_ready:
    dataset_path = '/home/jakub/HOME/Datasets/Flickr8k/captions.txt'
    dataset = load_captions(dataset_path)

    directory = '/home/jakub/HOME/Datasets/Flickr8k/Images'
    full_dataset = extract_features(directory, dataset)

    clean_captions(full_dataset)

    keys = list(full_dataset.keys())

    train_keys, test_keys = train_test_split(keys, test_size=0.2, random_state=42)
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

    folder_path = 'Torch_models_4'
    word_set, max_length, number_of_captions = create_word_set(train_dataset)
    vocab_size = len(tokenizer.word_index) + 1
    train_generator = data_generator(train_dataset, tokenizer, max_length, vocab_size)

    model = CustomModel(vocab_size, max_length).to(device=device)
    criterion = CustomLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 50
    steps_per_epoch = number_of_captions

for epoch in range(num_epochs):
    loss_sum = 0.0
    progress_bar = tqdm(total=steps_per_epoch, desc=f'Epoch {epoch+1}/{num_epochs}', position=0, leave=True)
    for i in range(steps_per_epoch):
        [image_features, captions], targets = next(train_generator)
        image_features = torch.tensor(image_features, dtype=torch.float32).to(device=device)
        captions = torch.tensor(captions, dtype=torch.long).to(device=device)
        targets = torch.tensor(targets, dtype=torch.float32).to(device=device)
        optimizer.zero_grad()
        outputs = model(image_features, captions)
        loss = criterion(outputs, targets)
        loss.backward()
        loss_sum += loss.item()
        train_loss = loss_sum/(i+1)
        optimizer.step()
        progress_bar.update(1)
        if i % 15 == 0:
            progress_bar.set_postfix(loss=train_loss)
    torch.save(model.state_dict(), os.path.join(folder_path, f'model_{epoch}.pth'))
progress_bar.close()

