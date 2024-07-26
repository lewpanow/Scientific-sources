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
