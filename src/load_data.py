import pandas as pd
import os
from preprocess import preprocess_text


def load_dataset(directory):
    data = []
    for label_type in ['pos', 'neg']:
        dir_name = os.path.join(directory, label_type)
        for filename in os.listdir(dir_name):
            if filename.endswith('.txt'):
                with open(os.path.join(dir_name, filename), 'r', encoding='utf-8') as f:
                    try:
                        text = f.read()
                    except UnicodeDecodeError:
                        continue
                preprocessed_text = preprocess_text(text)
                data.append((preprocessed_text, label_type))
    df = pd.DataFrame(data, columns=['text', 'label'])
    return df


train_dir = './dataset/train'
test_dir = './dataset/test'

train_data = load_dataset(train_dir)
test_data = load_dataset(test_dir)


train_data.to_csv('csv/train_data.csv', index=False)
test_data.to_csv('csv/test_data.csv', index=False)
