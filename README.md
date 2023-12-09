# Деплой ML модели в продакшен

Этот репозиторий содержит примеры кода к серии видеолекций "[Деплой ML модели](https://www.youtube.com/playlist?list=PL-aDfJxlxktwqi4AfMXNlXncFsKEZdL-L)".
Обсудим, что делать после того, как вы обучили модель, и как развернуть модель в проде.

## Примерный план:
О чём поговорим:
- Создание простой модели мультиклассовой классификации в Jupyter Notebook;
- Перенос в python скрипт;
- Пакетная обработка данных;
- Онлайн обработка данных, REST API сервис; (_in progress_)
- Docker контейнеризация; (_in progress_)

## Структура репозитория

- data - обучающие данные;
- models - обученные сериализованные модели;
- notebooks - папка для тетрадок Jupyter Notebooks;
- src - код приложения.

## Как воспользоваться репозиторием:

0. Убедитесь, что на вашем компьютере установлен Python и все необходимые пакеты (см. файл _requirements.txt_).
1. Склонировать репозиторий на ваш компьютер
    ```bash
    git clone https://github.com/PhySci/ML_deploy.git
    ```
2. Скачать файл с тренировочными данными _train.csv_ с [официальной страницы датасета](https://www.kaggle.com/datasets/purumalgi/music-genre-classification/data) и положить его в папку data
3. Выполнить все ячейки в тетрадке _notebooks/1.ipynb_; разобраться, что делается в каждой из них. Как результат, в папке _models/v1_ должны появиться сериализованные модели (файлы с расширением pcl).
4. Запустить файл _src/batch.py_

## Обратная связь

Если у вас остались вопросы или вы хотите предложить тему для следующего видео, то напишите об этом в комментариях к опубликованным видео (например, [здесь](https://www.youtube.com/watch?v=RrNeE9dc_70)), либо отправьте мне сообщение в [Телеграм](https://t.me/x00dr).
