# Как дообучить Gemma — Простое руководство

Этот проект поможет вам научить языковую модель  говорить в вашем стиле. Всё настроено так, чтобы быть максимально простым. Для опытных пользователей это может показаться слишком простым, но главная цель — лёгкость в использовании. Весь процесс сводится к запуску нескольких `.bat` файлов.

---

### Что такое "дообучение" (Fine-Tuning)? (Для новичков)

Представьте, что у вас есть супер-умный робот, который знает почти всё на свете (это Gemma!). Но он говорит очень обобщенно. "Дообучение" — это как дать этому роботу сборник ваших собственных разговоров и сказать: "Эй, учись на этом, чтобы общаться в похожем стиле!"

Вы не учите его новым фактам. Вы учите его новому *стилю* общения. После этого процесса робот останется таким же умным, но будет звучать немного больше как в примерах, которые вы ему дали.

---

### Пошаговая инструкция

#### Короткая версия
1. prepare_venv.bat     - готовит окружение
2. download_model.bat   - качает базовую модель
3. prepare_dataset.bat  - готовит датасет
4. finetune_gemma.bat   - тренирует лору
4.1 check_tensorboard_runs.bat   - проверяет прогресс
4.2 Если прервано обучение и надо продолжить - finetune_gemma.bat
5. merge_lora.bat       - сшивает лору с моделью
6. convert_to_gguf.bat  - конвертирует и сжимает

#### Внимание: Требования к оборудованию!
Чтобы обучить модель Gemma-3n, вам понадобится мощная видеокарта (GPU) с **минимум 24 ГБ видеопамяти (VRAM)**, например, NVIDIA RTX 3090, 4090 или профессиональная карта A-серии. Использование карты с меньшим объемом памяти будет *очень* медленным или может вообще не сработать.

#### Шаг 1: Подготовьте ваши данные для разговора

1.  Найдите файлы с историей ваших чатов. Этот скрипт предназначен для работы с файлами в формате JSON, экспортированными с различных платформ.
2.  Создайте папку с именем `dataset` в той же директории, где находятся `.bat` файлы.
3.  Поместите все ваши `.json` файлы с разговорами прямо в папку `dataset`.

**Как должны выглядеть данные?**
Скрипты ожидают, что ваши JSON-файлы содержат список сообщений, где каждое сообщение выглядит примерно так:
```json
{
    "sender": "User",
    "userInput": "Привет, можешь мне помочь?"
},
{
    "sender": "Claude",
    "claudeOutputText": "Конечно! Чем могу помочь?"
}
```
Так же будут взяты пары "инструкция-ответ" из данных с ключами 'instruction' и 'output'. Либо `"User"` (это будет инструкция) и следующий за ним ответ от `"Claude"`.


#### Шаг 2: Запустите установку

Этот шаг сделает всё за вас! Он загрузит все необходимые инструменты и базовую модель Gemma. Это нужно сделать только один раз.

> **Дважды щелкните по файлу `prepare_venv.bat`.**

Подождите, пока он завершится. Он загрузит много файлов, так что это может занять некоторое время в зависимости от вашего интернет-соединения.

#### Шаг 3: Начните обучение!

Это запустит основной процесс обучения модели вашему стилю общения.

> **Дважды щелкните по файлу `finetune_gemma.bat`.**

Этот шаг займет очень много времени — скорее всего, несколько часов, даже на мощном компьютере. Просто дайте ему поработать.

#### Шаг 4: Конвертируйте финальную модель (При необходимости)

После завершения обучения скрипт автоматически попытается преобразовать модель в её конечный, готовый к использованию формат GGUF.

Если по какой-то причине вам нужно повторно запустить только этот последний шаг, вы можете:
> **Дважды щелкните по файлу `convert_only.bat`.**

---

### Что вы получите в конце

Когда всё будет готово, вы найдете новый файл в папке `output` с именем `finetuned_gemma.gguf`. Это ваша персонализированная модель! Теперь вы можете загрузить этот файл в программы, поддерживающие формат GGUF (например, LM Studio, KoboldCpp и т.д.), и общаться с версией Gemma, которая научилась на ваших разговорах.

---
<br>

# How to Fine-Tune Gemma - The Simple Guide

This project helps you teach a smart computer brain (a Large Language Model named Gemma) to talk more like you. It's all set up to be as easy as possible. This is designed for simple use, not for power users. It's just a matter of clicking a couple of `.bat` files to get started!

---

### What is "Fine-Tuning"? (For Beginners)

Imagine you have a super-smart robot that knows almost everything in the world (that's Gemma!). But, it talks in a very general way. "Fine-tuning" is like giving this robot a collection of your own conversations and telling it, "Hey, learn from these so you can chat more like this!"

You are not teaching it new facts. You are teaching it a new *style* of conversation. After this process, the robot will still be the same smart robot, but it will sound a bit more like the examples you gave it.

---

### Step-by-Step Instructions

#### Hardware Warning!
To teach (fine-tune) the Gemma-3n model, you need a powerful computer graphics card (a GPU) with **at least 24GB of memory (VRAM)**, like an NVIDIA RTX 3090, 4090, or a professional A-series card. Using a card with less memory will be *extremely* slow or may not work at all.

#### Step 1: Prepare Your Conversation Data

1.  Find your conversation history files. This process is designed to work with JSON files exported from various chat platforms.
2.  Create a folder named `dataset` in the same directory as the `.bat` files.
3.  Place all your conversation `.json` files directly inside the `dataset` folder.

**How should the data look?**
The scripts expect your JSON files to contain a list of messages, where each message looks something like this:
```json
{
    "sender": "User",
    "userInput": "Hey, can you help me with something?"
},
{
    "sender": "Claude",
    "claudeOutputText": "Of course! What do you need help with?"
}
```
The script looks for messages from a `"User"` (this will be the instruction) and the `"Claude"` response that follows it.

#### Step 2: Run the Setup

This step does everything for you! It will download all the necessary tools and the base Gemma model. You only need to do this once.

> **Double-click the `prepare_venv.bat` file.**

Wait for it to finish. It will download a lot of files, so it might take a while depending on your internet connection.

#### Step 3: Start the Training!

This will start the main process of teaching the model your conversation style.

> **Double-click the `finetune_gemma.bat` file.**

This step will take a very long time—likely several hours, even on a powerful computer. Just let it run.

#### Step 4: Convert the Final Model (If Needed)

After training finishes, it will automatically try to convert the model into its final, ready-to-use GGUF format.

If for some reason you need to re-run only this final step, you can:
> **Double-click the `convert_only.bat` file.**

---

### What You Get at the End

After everything is done, you will find a new file in the `output` folder named `finetuned_gemma.gguf`. This is your personalized model! You can now load this file into programs that support the GGUF format (like LM Studio, KoboldCpp, etc.) and chat with a version of Gemma that has learned from your conversations.

---

# Технические детали проекта

Этот раздел объясняет, как скрипты обрабатывают ваши данные и какие важные изменения были внесены для обеспечения стабильной и эффективной работы.

## Как подготавливаются данные (`prepare_conversation_data.py`)

Скрипт гибко и надёжно обрабатывает вашу историю чатов, превращая её в формат, на котором модель может учиться.

**Формат входных данных:**
- Скрипт рекурсивно сканирует папку `dataset` на наличие `.json` файлов.
- Ожидается, что каждый файл содержит список сообщений, где у каждого есть поля `sender` (`"User"` или `"Claude"`), `userInput` и `claudeOutputText`.

**Режимы обработки:**
Вы можете выбрать один из двух форматов данных для обучения:

1.  **Формат `instruct` (По умолчанию)**
    -   **Что делает:** Создаёт простые пары "инструкция-ответ". Он находит сообщение пользователя и сопоставляет его со следующим ответом модели.
    -   **Зачем использовать:** Это прямой и эффективный способ научить модель стилю "вопрос-ответ". Рекомендуется как вариант по умолчанию.

2.  **Формат `conversational`**
    -   **Что делает:** Более продвинутый метод, который создаёт множество обучающих примеров из одного чата, чтобы научить модель естественному течению и контексту разговора. Данные генерируются тремя способами:
        -   **Пары "один на один":** Каждое сообщение пользователя сопоставляется с последующим ответом модели.
        -   **Расширяющаяся история:** Модели показывают разговор по мере его роста (сначала 2 реплики, потом 4, 6 и т.д.).
        -   **Скользящие окна:** Модели показывают небольшие фрагменты диалога фиксированного размера (например, последние 8 реплик).
    -   **Зачем использовать:** Этот метод создаёт более разнообразный набор данных, что может помочь модели лучше поддерживать длинный, последовательный разговор.

**Результат:**
Скрипт автоматически разделяет данные на обучающий (95%) и проверочный (5%) наборы, создавая файлы `train_dataset.jsonl` и `eval_dataset.jsonl` в папке `data`.

---

## Ключевые изменения и исправления (`finetune_gemma.py`)

Основной скрипт дообучения содержит несколько критически важных исправлений, которые обеспечивают стабильный процесс обучения, особенно для чувствительной архитектуры Gemma-3n. Эти проблемы были выявлены и решены, как задокументировано в `dev_log.txt`.

### Основные функции
-   **Дообучение QLoRA**: Используется 4-битная квантизация для значительного сокращения использования памяти, что позволяет дообучать модель на потребительских видеокартах с 24 ГБ VRAM.
-   **Умное управление чекпоинтами**: Специальный скрипт `ManageBestCheckpointsCallback` сохраняет только 2 лучших чекпоинта на основе потерь при валидации, экономя место на диске.
-   **Корректный расчет потерь**: В конвейере обработки данных маскируются токены заполнения (padding). Это критически важное исправление, которое не позволяет модели наказываться за предсказание пустого пространства, делая обучение эффективным.

### Патчи стабильности для Gemma-3n
Модель Gemma-3n оказалась нестабильной при стандартном дообучении. Были реализованы следующие патчи для устранения критических ошибок и "взрывных градиентов":
-   **Ручная деквантизация**: Слои `altup` и `lm_head`, которые вызывали сбои, вручную приводятся к полной точности `float32`.
-   **Заморозка неиспользуемых энкодеров**: Визуальные и аудио компоненты модели генерировали бесконечные значения, вызывая ошибки. Эти модули теперь заморожены для обеспечения численной стабильности.
-   **Оптимизированные аргументы обучения**: Конфигурация обучения была тщательно настроена: используется оптимизатор `paged_adamw_32bit`, более низкая скорость обучения (`5e-6`) и плавная схема прогрева (warmup), чтобы избежать проблемы "взрывного градиента".

---

# Project Technical Details

This section explains how the scripts process your data and what important changes were made for a stable and efficient workflow.

## How The Data Is Prepared (`prepare_conversation_data.py`)

The script is designed to be flexible and robust, processing your chat history into a format the model can learn from.

**Input Format:**
- The script recursively scans the `dataset` folder for any `.json` files.
- It expects each file to contain a list of messages, where each message object has a `sender` (`"User"` or `"Claude"`), a `userInput` field, and a `claudeOutputText` field. Alternatively works for pairs of `instruction` и `output`.

**Processing Modes:**
You can choose between two data formats for training:

1.  **`instruct` Format (Default)**
    -   **What it does:** Creates simple "instruction-response" pairs. It finds a user's message and pairs it with the model's next reply.
    -   **Why use it:** This is a straightforward and effective way to teach the model a direct question-and-answer style. It's the recommended default.

2.  **`conversational` Format**
    -   **What it does:** This is a more advanced method that creates multiple training examples from a single chat to teach the model the natural flow of a conversation. It generates data in three ways:
        -   **One-on-one Pairs:** Every user message is paired with the subsequent model response.
        -   **Expanding History:** The model is shown the conversation as it grows (first 2 turns, then 4, 6, etc.).
        -   **Sliding Windows:** The model is shown small, fixed-size chunks of the conversation (e.g., the last 8 turns).
    -   **Why use it:** This method produces a more diverse dataset, which can lead to a model that is better at holding a coherent, multi-turn conversation.

**Output:**
The script automatically splits your data into a training set (95%) and a validation set (5%), creating `train_dataset.jsonl` and `eval_dataset.jsonl` in the `data` folder.

---

## Technical Details & Key Changes (`finetune_gemma.py`)

The main fine-tuning script contains several critical fixes and features to ensure a stable and efficient training process, particularly for the sensitive Gemma-3n architecture. These were identified and solved as documented in the `dev_log.txt`.

### Core Features
-   **QLoRA Fine-Tuning**: Utilizes 4-bit quantization to drastically reduce memory usage, allowing the Gemma-3n model to be fine-tuned on consumer GPUs with 24GB of VRAM.
-   **Smart Checkpoint Management**: A custom `ManageBestCheckpointsCallback` saves only the top 2 performing checkpoints based on evaluation loss, saving disk space.
-   **Correct Loss Calculation**: The data processing pipeline correctly masks padded tokens. This is a critical fix that prevents the model from being penalized for predicting empty space, ensuring the training is effective.

### Gemma-3n Stability Patches
The Gemma-3n model proved to be unstable under standard QLoRA fine-tuning. The following patches were implemented to solve critical runtime errors and exploding gradients:
-   **Manual De-quantization**: The `altup` and `lm_head` layers, which were found to cause crashes, are manually cast to full `float32` precision.
-   **Freezing Unused Encoders**: The model's vision and audio components were found to generate infinite values, causing errors. These modules are now frozen to ensure numerical stability.
-   **Optimized Training Arguments**: The training configuration was carefully tuned for stability. It uses the `paged_adamw_32bit` optimizer, a lower learning rate (`5e-6`), and a gentle warmup schedule to avoid the "exploding gradient" problem.
