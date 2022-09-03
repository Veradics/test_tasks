# программа осуществляет парсинг диалогов из файла test_data.csv
import pandas as pd
import nltk
import pymorphy2
import json

morph = pymorphy2.MorphAnalyzer()

# данные
data = pd.read_csv('test_data.csv')


# проверка реплики, является ли она самопрезентацией менеджера
def is_self_presentation(message, name):    
    if (
        f'меня {name.lower()} зовут' in message.lower() or
        f'меня зовут {name.lower()}' in message.lower() or
        f'это {name.lower()}' in message.lower()
    ):
        return True
    return False


# нахождение в сообщении имени менеджера
def name_recognize(message):
    global morph
    prob_thresh = 0.2  # порог для определения слова-имени
    name_list = []     # на случай, если в сообщении не одно имя
    
    for word in nltk.word_tokenize(message.lower()):
        is_name = any(('Name' in p.tag and p.score >= prob_thresh) for p in morph.parse(word))
        if is_name:
            name_list.append(word)
            
    for name in name_list:
        if is_self_presentation(message.lower(), name):
            return name


# нахождение в сообщении названия компании
def company_recognize(message):
    global morph
    if 'компания' not in message.lower():
        return
    result = ''
    for word in nltk.word_tokenize(message.lower()):
        # является ли слово началом названия компании
        is_company = any(f'компания {word}' in message.lower() and 'NOUN' in p.tag for p in morph.parse(word))
        # является ли слово частью названия компании
        is_part_name = any(
            ((len(result) > 0) and f'{result}{word}' in message.lower()) and 
            'NOUN' in p.tag for p in morph.parse(word)
        )
        
        if is_company or is_part_name:
            result += word + ' '
            
    return result.strip()


# проверка реплики, содержит ли она приветствие
def is_greeting(message):
    if (
        'добры' in message.lower() or         # добрый день, добрый вечер...
        'здравствуй' in message.lower() or    # здравствуй, здравствуйте
        'привет'in message.lower()            # привет, приветствую
    ):          
        return True

    return False


# проверка реплики, содержит ли она прощание
def is_goodbye_message(message):
    if (
        'свидани' in message.lower() or   # до свидания (и если вдруг ошибка в написании слова)
        'встречи' in message.lower() or   # до встречи 
        'хороше' in message.lower() or    # хорошего дня, хорошего понедельника... НО: не 'хорошо' - это не приветствие
        'добро' in message.lower()        # всего доброго, доброго дня...
    ): 
        return True
    
    return False

# проверка на выполнение менеджером поставленного условия
def check_manager(greeting, goodbye):
    if greeting and goodbye:
        return True
    return False


# сам парсер
def parser_dialog(dialog, dlg_id):
    dialog_info = {}         # для записи итоговых значений
    dtable = dialog.copy()   # копия датасета
    
    dtable['name'] = dtable['text'].apply(name_recognize)        # имя менеджера, если оно появилось в сообщении
    dtable['company'] = dtable['text'].apply(company_recognize)  # название компании, если оно появилось
    dtable['greeting'] = dtable['text'].apply(is_greeting)       # является ли текст приветствием (true/false)
    dtable['goodbye'] = dtable['text'].apply(is_goodbye_message) # является ли текст прощанием (true/false)
    
    # записываем в словарь номер диалога
    dialog_info['dlg_id'] = dlg_id

    # извлекаем имя менеджера и компании
    dialog_info['manager_name'] = (''.join(list(dtable['name'].fillna('').unique()))).capitalize()
    dialog_info['company_name'] = (''.join(list(dtable['company'].fillna('').unique()))).capitalize()
    
    # если имена в диалоге не представлены
    if dialog_info['manager_name'] == '':
        dialog_info['manager_name'] = 'unknown'
    if dialog_info['company_name'] == '':
        dialog_info['company_name'] = 'unknown'
    
    
    # является ли реплика само-представлением менеджера
    dtable['self_present'] = dtable.apply(lambda x: is_self_presentation(x['text'], dialog_info['manager_name']), axis=1)
    
                                          
    # извлекаем приветственное сообщение
    dgreeting = dtable.query('greeting')
    dialog_info['greeting_message'] = {row['line_n']: row['text'] for index, row in dgreeting.iterrows()}
    
    # извлекаем прощальное сообщение
    dgoodbye = dtable.query('goodbye')
    dialog_info['goodbye_message'] = {row['line_n']: row['text'] for index, row in dgoodbye.iterrows()}
    
    # извлекаем реплику, где менеджер представляется
    dself = dtable.query('self_present')
    dialog_info['manager_self_presentation'] = {row['line_n']: row['text'] for index, row in dself.iterrows()}
                                          
    
    # выполнил ли менеджер поставленное условие (поздороваться и попрощаться)
    dialog_info['requirement_done'] = (dgreeting.shape[0] != 0 and dgoodbye.shape[0] != 0)
    
    return dialog_info


# парсинг диалогов из файла
dialogs = []
for i in data['dlg_id'].unique():
    dialogs.append(parser_dialog(data.query('dlg_id == @i and role == "manager"'), str(i)))

# запись результата в файл формата json 
with open("parsed_dialogs.json", "w", encoding="utf-8") as file:
    json.dump(dialogs, file,  ensure_ascii=False)
