import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import re

# Установка устройства (GPU, если доступно)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загрузка токенизатора BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Функция для преобразования меток
def encode_labels(labels):
    label_map = {'B': 0, 'C': 1, 'D': 2, 'E': 3, 'O': 4}  # O для токенов, не входящих в метки
    encoded = []

    for label_set in labels:
        encoded_set = []
        for i, label in enumerate(label_set):
            if isinstance(label, str) and label:  # Проверка, что это строка и не пустая
                tokens = tokenizer.tokenize(label)
                # Добавляем метки для каждого токена
                encoded_set.extend([i] * len(tokens))
            else:
                encoded_set.append(4)  # O для пустых меток
        encoded.append(encoded_set)

    return encoded

# Создание датасета
class BibliographyDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = 128  # Максимальная длина последовательности

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        # Создаем label_ids той же длины, что и input_ids
        label_ids = torch.full((self.max_len,), 4)  # Заполняем 4 (O) по умолчанию

        # Обрезаем или дополняем label до нужной длины
        label = label[:self.max_len - 2]  # -2 для учета [CLS] и [SEP]
        label_ids[1:len(label)+1] = torch.tensor(label)
        label_ids[0] = 4  # [CLS] токен
        label_ids[len(label)+1] = 4  # [SEP] токен

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label_ids
        }

# Загрузка данных
df = pd.read_excel(r"C:/Users/Lev/Desktop/data.xlsx")
sources = df['A'].tolist()
labels = df[['B', 'C', 'D', 'E']].values.tolist()
encoded_labels = encode_labels(labels)

# Разделение данных
train_texts, val_texts, train_labels, val_labels = train_test_split(sources, encoded_labels, test_size=0.2, random_state=42)

# Создание датасетов
train_dataset = BibliographyDataset(train_texts, train_labels)
val_dataset = BibliographyDataset(val_texts, val_labels)

# Создание загрузчиков данных
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Инициализация модели
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=5)
model.to(device)

# Оптимизатор
optimizer = AdamW(model.parameters(), lr=2e-5)

# Функция обучения
def train(model, dataloader, optimizer):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)

# Функция оценки
def evaluate(model, dataloader):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            pred = torch.argmax(logits, dim=2)
            predictions.extend(pred.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return predictions, true_labels

# Обучение модели
num_epochs = 8
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}")

# Оценка модели
predictions, true_labels = evaluate(model, val_loader)

flat_predictions = [p for pred in predictions for p in pred]
flat_true_labels = [l for label in true_labels for l in label]

print(classification_report(flat_true_labels, flat_predictions))

def predict_elements(source):
    model.eval()
    encoding = tokenizer.encode_plus(
        source,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=2)
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    label_map = {0: 'B', 1: 'C', 2: 'D', 3: 'E', 4: 'O'}
    results = {label: '' for label in label_map.values()}
    
    previous_label = None
    
    for token, prediction in zip(tokens, predictions[0]):
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            continue
        
        label = label_map[prediction.item()]
        
        if token.startswith('##'):
            results[previous_label] += token[2:]
        else:
            if previous_label == label:
                results[label] += ' ' + token
            else:
                if results[label]:
                    results[label] += ' '
                results[label] += token
        
        previous_label = label
    
    # Удаление пустых токенов и пробелов в начале и конце
    for key in results:
        results[key] = results[key].strip()
    
    # Попробуем собрать журнал из меток D и O
    journal_parts = []
    journal_parts.append(results['D'])
    journal_parts.append(results['O'])
    journal = ' '.join(journal_parts).replace('.', '').replace('//', '').replace('/ /', '').strip()
    journal = re.sub(r'\s+', ' ', journal)
    
    # Попытка извлечь название журнала из метки O более аккуратно
    if len(journal) < 5:
        journal_match = re.search(r'//\s*(.*?)[,.]', source)
        if journal_match:
            journal = journal_match.group(1).strip()
        else:
            parts = source.split('//')
            if len(parts) > 1:
                journal_part = parts[1].split(',')[0].strip()
                journal = journal_part

    # Удаляем части, которые не относятся к журналу
    journal_cleaned = []
    for part in journal.split():
        if not re.match(r'^\d+$', part) and not re.match(r'^\w{2,3}$', part):
            journal_cleaned.append(part)
    journal = ' '.join(journal_cleaned).strip()

    # Обработка результата для извлечения нужной информации
    author = results['B'].replace('.', '').replace(',', '').strip()  # Убираем лишние точки и запятые
    author = re.sub(r'^\d+\s+', '', author)  # Убираем номер перед именем
    author = re.sub(r'\s+', ' ', author)  # Убираем лишние пробелы
    title = results['C'].replace('.', '').strip()  # Убираем лишние точки
    title = title.replace('Своиства', 'Свойства')  # Исправляем опечатку
    year = re.search(r'\d{4}', source)  # Извлечение года
    year = year.group(0) if year else ''
    
    return {
        'author': author.title(),  # Форматируем имя с заглавными буквами
        'title': title.title(),  # Форматируем название с заглавными буквами
        'journal': journal.title(),  # Форматируем название журнала с заглавными буквами
        'year': year
    }

# Пример использования
test_source = "Овчинникова С. В. Дополнение к видовому составу бурачниковых (Boraginaceae) Внешней Монголии // Turczaninowia, 2019. Т. 22, № 3. С. 97–110. DOI: 10.14258/turczaninowia.22.3.5"
print(predict_elements(test_source))