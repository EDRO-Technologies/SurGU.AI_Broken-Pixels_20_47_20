import telebot
from otherconvert import *

# Инициализация бота с токеном
botTimeWeb = telebot.TeleBot('id:token')

user_ready_for_questions = {}


# Обработчик команды /start при запуске бота
@botTimeWeb.message_handler(commands=['start'])
def start_bot(message):
    first_mess = f"<b>{message.from_user.first_name}</b>, привет!\nЯ умею находить ответы на вопросы, опираясь на отправленный вами документ.\n\nПожалуйста, загрузите файл (.pdf)"
    botTimeWeb.send_message(message.chat.id, first_mess, parse_mode='html')
    user_ready_for_questions[message.chat.id] = False


# Обработчик для сообщений с документами
@botTimeWeb.message_handler(content_types=['document'])
def handle_document(message):
    if message.document.mime_type == 'application/pdf':
        try:
            file_info = botTimeWeb.get_file(message.document.file_id)
            downloaded_file = botTimeWeb.download_file(file_info.file_path)

            src = 'src' + message.document.file_name

            with open(src, 'wb') as new_file:
                new_file.write(downloaded_file)

        except Exception as e:
            print(message, e)

        botTimeWeb.send_message(message.chat.id, "Файл принят. Начинаю обработку...")
        print(src)
        documents = load_documents(src)

        chunks = split_documents(documents, message.document.file_name)
        # for i in chunks:
        #     print(i)
        add_to_chroma(chunks)

        botTimeWeb.send_message(message.chat.id,"Обработка завершена.\nГотов ответить на ваши вопросы")

        user_ready_for_questions[message.chat.id] = True
    else:
        botTimeWeb.send_message(message.chat.id, "Файл должен быть формата PDF")


# Обработчик для текстовых сообщений от пользователя
@botTimeWeb.message_handler(content_types=['text'])
def handle_text_message(message):
    print(f"{message.from_user.first_name} написал: {message.text}")

    if user_ready_for_questions.get(message.chat.id, False):

        botTimeWeb.send_message(message.chat.id, query_rag(message.text))

    else:
        botTimeWeb.send_message(message.chat.id, "Для начала работы необходимо загрузить файл")


# Запуск бота в режиме бесконечного опроса
botTimeWeb.infinity_polling()