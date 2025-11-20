#!/usr/bin/env python3
"""
Prepare conversation data from JSON files for Gemma fine-tuning
Converts User-Claude conversations into instruction-response pairs
"""

import os
import json
import glob
import logging
from pathlib import Path
from typing import List, Dict, Any
import argparse
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_json_file(file_path: str) -> Any:
    """Загружает один JSON-файл."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Не удалось прочитать JSON в файле {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Не удалось загрузить файл {file_path}: {e}")
        return None

def extract_instruction_response_pairs(conversations: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Извлекает пары "инструкция-ответ" из данных диалога."""
    pairs = []

    # Группируем сообщения, находя пары "сообщение Пользователя - ответ Claude"
    user_message = None

    for entry in conversations:
        sender = entry.get('sender')
        user_input = entry.get('userInput')
        claude_output = entry.get('claudeOutputText')

        # Если сообщение от пользователя и оно не пустое
        if sender == 'User' and user_input and user_input.strip():
            # Сохраняем сообщение пользователя, чтобы связать его со следующим ответом Claude
            user_message = user_input.strip()

        # Если сообщение от Claude, оно не пустое, и у нас есть предыдущее сообщение от пользователя
        elif sender == 'Claude' and claude_output and claude_output.strip() and user_message:
            # Мы нашли полную пару "инструкция-ответ"
            pairs.append({
                'instruction': user_message,
                'response': claude_output.strip(),
                'context': None  # Поле для совместимости формата, здесь не используется
            })
            user_message = None  # Сбрасываем, чтобы начать поиск новой пары

    return pairs

def extract_from_instruct_json(data: Any) -> List[Dict[str, str]]:
    """Извлекает пары "инструкция-ответ" из данных с ключами 'instruction' и 'output'."""
    pairs = []
    
    # Если данные - это один объект, превращаем его в список для единообразной обработки
    if isinstance(data, dict):
        data = [data]
        
    if not isinstance(data, list):
        logger.warning("Данные для инструкций не являются словарем или списком. Пропускаем.")
        return []
        
    for item in data:
        # Проверяем, что у объекта есть нужные поля 'instruction' и 'output'
        if isinstance(item, dict) and 'instruction' in item and 'output' in item:
            instruction = item.get('instruction')
            response = item.get('output')
            
            # Убеждаемся, что инструкция и ответ - непустые строки
            if isinstance(instruction, str) and isinstance(response, str) and instruction.strip() and response.strip():
                pairs.append({
                    'instruction': instruction.strip(),
                    'response': response.strip(),
                    'context': None
                })
        else:
            logger.warning(f"Пропускаем неправильно отформатированный элемент в файле.")
            
    return pairs

def chunk_dialogue(messages: List[Dict[str, str]], chunk_sizes: List[int], chunk_overlap: int) -> List[Dict[str, Any]]:
    """
    Создает обучающие примеры из одного диалога, используя расширяющийся контекст и "скользящие окна".
    (Эта функция оказалась слишком сложной и была заменена на 'extract_sliding_window_conversational')
    """
    examples = []
    
    # Убеждаемся, что сообщения идут парами (пользователь, модель)
    if len(messages) % 2 != 0:
        messages = messages[:-1]

    # 1. Расширяющаяся история (от начала диалога до текущего момента)
    for i in range(2, len(messages) + 1, 2):
        # Каждый пример включает всю историю до ответа модели
        examples.append({"messages": messages[:i]})

    # 2. "Скользящие окна" - фрагменты диалога фиксированного размера
    for size in chunk_sizes:
        if size <= 0:
            continue
        
        # Шаг, с которым окно будет двигаться по диалогу
        step = max(2, size - chunk_overlap)
        
        for i in range(0, len(messages) - size + 1, step):
            chunk = messages[i:i + size]
            # Убеждаемся, что фрагмент заканчивается ответом модели
            if len(chunk) > 0 and chunk[-1]['role'] == 'model':
                examples.append({"messages": chunk})
    
    # Удаление дубликатов, чтобы модель не обучалась на одном и том же несколько раз
    unique_examples = [dict(t) for t in {tuple(sorted(d.items())) for d in [item for sublist in [e['messages'] for e in examples] for item in sublist]}]
    
    final_examples = []
    seen_hashes = set()

    for ex in examples:
        # Создаем уникальный "отпечаток" для каждого диалога, чтобы найти дубликаты
        msg_tuple = tuple(tuple(sorted(msg.items())) for msg in ex['messages'])
        if msg_tuple not in seen_hashes:
            final_examples.append(ex)
            seen_hashes.add(msg_tuple)
            
    return final_examples

def extract_sliding_window_conversational(conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Создает обучающие примеры из одного диалога, используя три метода:
    1. Пары "Пользователь-Модель": Создает пример для каждого сообщения пользователя в паре со следующим ответом модели.
    2. Расширяющаяся история: Создает пример для каждого ответа модели, включая всю предыдущую историю.
    3. "Скользящие окна": Создает примеры из фрагментов диалога фиксированного размера.
    """
    # Сначала собираем все сообщения в один плоский список
    raw_messages = []
    for entry in conversations:
        sender = entry.get('sender')
        user_input = (entry.get('userInput') or '').strip()
        claude_output = (entry.get('claudeOutputText') or '').strip()
        
        # Заменяем пустые сообщения на заглушки
        if user_input == "" or user_input is None: user_input = "continue"
        if claude_output == "" or claude_output is None: claude_output = "huh?..."

        if sender == 'User' and user_input:
            raw_messages.append({"role": "user", "content": user_input})
        elif sender == 'Claude' and claude_output:
            raw_messages.append({"role": "model", "content": claude_output})

    if not raw_messages:
        return []

    generated_samples = []

    # --- Метод 1: Пары "Пользователь-Модель" ---
    # Этот метод обрабатывает случаи, когда пользователь пишет несколько сообщений подряд.
    # Каждое из этих сообщений будет сопоставлено со следующим ответом модели.
    i = 0
    while i < len(raw_messages):
        if raw_messages[i]['role'] == 'user':
            user_message_block = []
            # Собираем все подряд идущие сообщения от пользователя
            while i < len(raw_messages) and raw_messages[i]['role'] == 'user':
                user_message_block.append(raw_messages[i])
                i += 1
            
            # Если после них есть ответ модели
            if i < len(raw_messages) and raw_messages[i]['role'] == 'model':
                model_message = raw_messages[i]
                # Создаем отдельный обучающий пример для каждого сообщения пользователя
                for user_msg in user_message_block:
                    generated_samples.append({"messages": [user_msg, model_message]})
                i += 1
        else:
            i += 1

    # Для следующих методов нам нужен диалог, где роли строго чередуются.
    # Объединяем подряд идущие сообщения от одного и того же автора.
    merged_messages = []
    if raw_messages:
        merged_messages.append(raw_messages[0])
        for msg in raw_messages[1:]:
            if msg['role'] == merged_messages[-1]['role']:
                # Добавляем текст к предыдущему сообщению
                merged_messages[-1]['content'] += "\n" + msg['content']
            else:
                # Добавляем новое сообщение
                merged_messages.append(msg)
    
    # Диалог должен начинаться с сообщения пользователя
    if merged_messages and merged_messages[0]['role'] == 'model':
        merged_messages = merged_messages[1:]

    # --- Метод 2: Расширяющаяся история ---
    # Создает пример, который включает все сообщения от начала диалога до текущего ответа модели.
    if merged_messages:
        for i in range(len(merged_messages)):
            if merged_messages[i]['role'] == 'model':
                context = merged_messages[:i+1]
                # Убеждаемся, что в контексте есть хотя бы одно сообщение от пользователя
                if any(msg['role'] == 'user' for msg in context):
                    generated_samples.append({"messages": context})

    # --- Метод 3: "Скользящие окна" ---
    # Создает примеры из фрагментов ("окон") диалога фиксированного размера.
    chunk_sizes_in_pairs = [8, 16] # Размеры окон в парах "вопрос-ответ" (8 пар = 16 сообщений)
    for num_pairs in chunk_sizes_in_pairs:
        chunk_size = num_pairs * 2
        for i in range(0, len(merged_messages) - chunk_size + 1, 2):
            chunk = merged_messages[i : i + chunk_size]
            # Убеждаемся, что фрагмент начинается с пользователя и заканчивается моделью
            if chunk and chunk[0]['role'] == 'user' and chunk[-1]['role'] == 'model':
                generated_samples.append({"messages": chunk})

    # --- Удаление дубликатов ---
    # Так как разные методы могли создать одинаковые примеры, мы их удаляем.
    unique_samples = []
    seen_conversations = set()

    for sample in generated_samples:
        # Превращаем диалог в уникальный "отпечаток", чтобы проверить на дубликаты
        conversation_tuple = tuple(tuple(sorted(msg.items())) for msg in sample['messages'])
        
        if conversation_tuple not in seen_conversations:
            unique_samples.append(sample)
            seen_conversations.add(conversation_tuple)

    return unique_samples


def process_conversation_folder(
    folder_path: str, 
    output_file_base: str, 
    data_format: str, 
    validation_split: int,
    max_samples: int = None,
    chunk_sizes: List[int] = [],
    chunk_overlap: int = 2
) -> int:
    """Обрабатывает все JSON-файлы в папке и ее подпапках для создания обучающих данных."""
    if not os.path.exists(folder_path):
        logger.error(f"Папка {folder_path} не существует")
        return 0

    # Рекурсивно находим все JSON-файлы в указанной папке
    json_files = [str(p) for p in Path(folder_path).rglob("*.json")]

    if not json_files:
        logger.error(f"В папке {folder_path} не найдено JSON-файлов")
        return 0

    logger.info(f"Найдено {len(json_files)} JSON-файлов для обработки")

    all_entries = []

    for file_path in json_files:
        logger.info(f"Обработка {os.path.basename(file_path)}...")
        data = load_json_file(file_path)
        
        if not data:
            continue
        
        # Если выбран формат "инструкция-ответ"
        if data_format == "instruct":
            entries = []
            # Простая проверка, чтобы определить формат JSON-файла
            if isinstance(data, list) and data and isinstance(data[0], dict) and 'sender' in data[0]:
                entries = extract_instruction_response_pairs(data) # Формат диалога User/Claude
            else:
                entries = extract_from_instruct_json(data) # Формат с ключами 'instruction'/'output'

            if entries:
                all_entries.extend(entries)
                logger.info(f"Извлечено {len(entries)} пар инструкций.")
        
        # Если выбран формат "диалог"
        elif data_format == "conversational":
            if not isinstance(data, list):
                logger.warning(f"Файл {file_path} не является списком, пропускаем.")
                continue

            entries = extract_sliding_window_conversational(data)
            all_entries.extend(entries)
            logger.info(f"Создано {len(entries)} примеров диалогов.")

    # Перемешиваем и ограничиваем количество примеров
    logger.info(f"Всего создано {len(all_entries)} примеров до перемешивания и ограничения.")
    import random
    random.shuffle(all_entries)

    # Если задан лимит, обрезаем список
    if max_samples and len(all_entries) > max_samples:
        all_entries = all_entries[:max_samples]
        logger.info(f"Количество примеров ограничено до {max_samples}.")

    # --- Разделение на обучающий и валидационный наборы ---
    if validation_split > 0 and validation_split < 100:
        split_index = math.ceil(len(all_entries) * (validation_split / 100.0))
        
        eval_entries = all_entries[:split_index]
        train_entries = all_entries[split_index:]
        
        logger.info(f"Разделение данных: {len(train_entries)} для обучения ({100-validation_split}%) и {len(eval_entries)} для валидации ({validation_split}%).")
    else:
        train_entries = all_entries
        eval_entries = []
        logger.info("Валидационный набор не создается (split = 0).")

    # --- Сохранение файлов ---
    output_dir = os.path.dirname(output_file_base)
    base_name = os.path.basename(output_file_base)
    
    # Определяем имена файлов для обучающего и валидационного наборов
    train_output_file = os.path.join(output_dir, f"train_{base_name}")
    eval_output_file = os.path.join(output_dir, f"eval_{base_name}")

    # Сохраняем обучающий набор
    logger.info(f"Сохранение {len(train_entries)} записей в файл {train_output_file}")
    with open(train_output_file, 'w', encoding='utf-8') as f:
        for entry in train_entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

    # Сохраняем валидационный набор, если он есть
    if eval_entries:
        logger.info(f"Сохранение {len(eval_entries)} записей в файл {eval_output_file}")
        with open(eval_output_file, 'w', encoding='utf-8') as f:
            for entry in eval_entries:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')

    logger.info("Подготовка данных завершена!")
    return len(train_entries), len(eval_entries)

def validate_output_file(output_file: str, data_format: str) -> bool:
    """Проверяет созданный файл с обучающими данными на корректность формата."""
    if not os.path.exists(output_file):
        logger.error(f"Выходной файл {output_file} не был создан")
        return False

    try:
        count = 0
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                # Проверяем наличие нужных ключей для каждого формата
                if data_format == "instruct" and ('instruction' not in data or 'response' not in data):
                    logger.error("Неверный формат 'instruct' в выходном файле")
                    return False
                if data_format == "conversational" and 'messages' not in data:
                    logger.error("Неверный формат 'conversational' в выходном файле")
                    return False
                count += 1

        logger.info(f"Проверка успешна: найдено {count} корректных записей")
        return True

    except Exception as e:
        logger.error(f"Ошибка при проверке файла: {e}")
        return False

def main():
    import argparse

    # Настройка аргументов командной строки для запуска скрипта
    parser = argparse.ArgumentParser(description="Подготовка данных из диалогов для дообучения Gemma")
    parser.add_argument(
        "--input-folder",
        default="dataset",
        help="Папка с исходными JSON-файлами диалогов"
    )
    parser.add_argument(
        "--output-file",
        default="data/dataset.jsonl",
        help="Базовое имя выходного JSONL-файла (будет добавлено 'train_' и 'eval_')."
    )
    parser.add_argument(
        "--format",
        type=str,
        default="instruct",
        choices=["instruct", "conversational"],
        help="Формат выходных данных: 'instruct' (инструкция-ответ) или 'conversational' (диалог)."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=4000,
        help="Максимальное количество обучающих примеров."
    )
    parser.add_argument(
        "--validation-split",
        type=int,
        default=5,
        help="Процент данных для валидационного набора (например, 5 для 5%%)."
    )
    parser.add_argument(
        '--chunk-sizes', 
        nargs='+', 
        type=int, 
        default=[8, 16], 
        help='Список размеров "окон" для диалогового формата (например, 4 8 20).'
    )
    parser.add_argument(
        '--chunk-overlap', 
        type=int, 
        default=2, 
        help='Количество сообщений, на которое будут пересекаться "окна".'
    )

    args = parser.parse_args()

    # Убеждаемся, что папка для выходного файла существует
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Запускаем обработку файлов
    total_train, total_eval = process_conversation_folder(
        args.input_folder,
        args.output_file,
        args.format,
        args.validation_split,
        args.max_samples,
        args.chunk_sizes,
        args.chunk_overlap
    )

    if total_train == 0 and total_eval == 0:
        logger.error("Обучающие данные не были созданы")
        return 1

    # Проверяем результат
    output_dir = os.path.dirname(args.output_file)
    base_name = os.path.basename(args.output_file)
    train_output_file = os.path.join(output_dir, f"train_{base_name}")
    eval_output_file = os.path.join(output_dir, f"eval_{base_name}")
    
    logger.info("--- Проверка созданных файлов ---")
    valid_train = validate_output_file(train_output_file, args.format)
    
    valid_eval = True
    if total_eval > 0:
        valid_eval = validate_output_file(eval_output_file, args.format)

    if not valid_train or not valid_eval:
        logger.error("Один или несколько файлов не прошли проверку.")
        return 1

    logger.info(f"Успешно подготовлено {total_train} обучающих и {total_eval} валидационных примеров в формате '{args.format}'")
    return 0

if __name__ == "__main__":
    exit(main())
