from langchain_community.document_loaders import PyPDFLoader
from langchain.schema.document import Document
import re
from langchain_community.embeddings.yandex import YandexGPTEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_chroma import Chroma
import requests, json
from langchain.prompts import ChatPromptTemplate


# Функция для загрузки PDF-документов с помощью PyPDFLoader
def load_documents(source_file):
    document_loader = PyPDFLoader(source_file)
    return document_loader.load()

# Функция для разделения документов на чанки
def split_documents(documents: list[Document], source: str):
    min_chunk_size = 300
    max_chunk_size = 600
    max_sentence_size = 700
    overlap_size = 2

    all_text = []
    for doc in documents:
        page_text = doc.page_content
        page_number = doc.metadata.get('page', 1)
        cleaned_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', page_text)
        sentences = re.split(r'(?<=[.!?;])\s+', cleaned_text)

        for sentence in sentences:
            all_text.append((sentence, page_number))

    chunks = []
    current_chunk = []
    current_length = 0
    current_page = None

    for sentence, page in all_text:
        sentence_length = len(sentence)

        if sentence_length > max_sentence_size:
            if current_chunk:
                chunks.append((current_chunk, current_page))
                current_chunk = []
            chunks.append(([sentence], page))
            current_length = 0
            current_page = page
            continue

        if current_length + sentence_length > max_chunk_size and len(
                current_chunk) >= 2 and current_length >= min_chunk_size:
            chunks.append((current_chunk, current_page))
            current_chunk = []
            current_length = 0
            current_page = page

        if not current_chunk:
            current_page = page

        current_chunk.append(sentence)
        current_length += sentence_length

        if current_length >= max_chunk_size:
            if len(current_chunk) < overlap_size:
                continue

            if len(current_chunk[-1]) < 500:
                overlap = current_chunk[-overlap_size:]
                chunks.append((current_chunk, current_page))
                current_chunk = overlap
                current_page = page
                current_length = sum(len(s) for s in overlap)
            else:
                chunks.append((current_chunk, current_page))
                current_chunk = []
                current_length = 0

    if current_chunk:
        chunks.append((current_chunk, current_page))

    formatted_chunks = [{'content': ' '.join(chunk), 'source': source, 'page': page}
                        for chunk, page in chunks if len(' '.join(chunk)) >= min_chunk_size]

    return formatted_chunks



CHROMA_PATH = "chroma"

# Функция для создания уникальных идентификаторов чанков
def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        if isinstance(chunk, dict):
            source = chunk.get("source")
            page = chunk.get("page")
            if "metadata" not in chunk:
                chunk["metadata"] = {}
        else:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")

        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        if isinstance(chunk, dict):
            chunk["metadata"]["id"] = chunk_id
        else:
            chunk.metadata["id"] = chunk_id

    return chunks

# Путь для хранения базы данных Chroma
CHROMA_PATH = "chroma"

# Функция для добавления чанков в базу данных Chroma
def add_to_chroma(chunks: list):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")
    across = 0
    new_chunks = []
    for chunk in chunks_with_ids:
        chunk_id = chunk.get("metadata", {}).get("id") if isinstance(chunk, dict) else chunk.metadata.get("id")
        if chunk_id not in existing_ids:
            if isinstance(chunk, dict):
                chunk = Document(
                    page_content=chunk.get("content", ""),
                    metadata=chunk.get("metadata", {})
                )
            new_chunks.append(chunk)
        else:
            across += 1

    print(across)

    if new_chunks:
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("No new documents to add")



# Функция для создания объекта Yandex Embeddings для векторизации текста
def get_embedding_function():
    embeddings = YandexGPTEmbeddings(
        folder_id="folder_id",
        api_key='api_key')

    return embeddings


# Функция для создания запросов к Chroma и обработки ответов от Яндекс LLM
def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    text_prompt = """ 
    Ответьте на вопрос, основываясь только на следующем контексте:
    {context}

    Ответьте на вопрос, основываясь на приведенном выше контексте: 
    {question}
    """

    results = db.similarity_search_with_score(query_text, k=5)
    #
    for i in results:
        print(i)
    print()
    print()
    if len(results) == 0 or results[0][1] > 1.4:
        return "Информация по данному вопросу отсутствует во входных данных."

    context_text = "\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(text_prompt)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)


    # Параметры для отправки запроса в Yandex LLM API
    promt = {
    "modelUri": "gpt://modelUri/yandexgpt-lite",
    "completionOptions": {
        "stream": False,
        "temperature": 0.2,
        "maxTokens": 6000
    },
    "messages": [
        {
            "role": "user",
            "text": prompt
        }
        ]
    }
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Api-Key Authorization",
        "x-folder-id": "folder"
    }

    response = requests.post(url, headers=headers, json=promt)
    data = json.loads(response.text)

    return (data['result']['alternatives'][0]['message']['text'])




