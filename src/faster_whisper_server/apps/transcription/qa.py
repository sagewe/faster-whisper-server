import os
import queue
import string
import threading
import zhon
from abc import ABC
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain_chroma import Chroma
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_community.llms.tongyi import Tongyi
from typing import Any


PROMPT_TEMPLATE = """
妳是一個誠實的{lang}語音助手，可以通過查閱給出的參考文檔內容來進行歸納總結，現在給出的參考文檔內容如下:
{context}
之前的對話歷史:
{chat_history}
請基於[參考文檔內容]和對話歷史進行總結和回答，不要聯想或編造內容，答案需要體現你的專業性，同埋需要非常簡潔、一定要來自參考內容文檔。如果用戶的問題答案無法從[參考文檔內容]中提取總結得到，請回答現有的知識庫檢索不到答案。
回答需要使用香港粵語，請記住香港粵語用繁體字噶。用戶提問的問題是: {question}
註意: 所有內容均不要聯想，只可以從[參考文檔內容]中進行提取!!另外，需要過濾廣告用語，比如電話鏈接、產品廣告、或者引導去某個網站等。
"""

mapping = {"银行与保险知识库": "rag-chroma"}

punctuation = set(string.punctuation + zhon.hanzi.punctuation + "\n")


class StreamingCallbackHandler(BaseCallbackHandler, ABC):
    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.q.put(token)


class RAGLLM:
    def __init__(
        self,
        vectorstore_path=None,
        collection_name="银行与保险知识库",
        chat_model="qwen-max",
        embedding_model="text-embedding-v1",
        api_key=None,
        streaming=False,
    ):
        if vectorstore_path is None or vectorstore_path == "" or not os.path.exists(vectorstore_path):
            raise ValueError(f"Vectorstore path {vectorstore_path} not found")
        if collection_name not in mapping:
            raise ValueError(f"Collection name {collection_name} not found in mapping")
        if api_key is None or api_key == "":
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if api_key is None or api_key == "":
                raise ValueError("DASHSCOPE_API_KEY is not set")
        self.dashscope_api_key = api_key
        self.vectorstore = self.get_vectorstore(
            vectorstore_path, collection_name=mapping[collection_name], embedding_model=embedding_model
        )
        self.retriever = self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 100})
        self.token_queue = queue.Queue()
        self.llm = self.get_chat_llm(chat_model=chat_model, streaming=streaming)
        self.prompt = PromptTemplate(
            template=PROMPT_TEMPLATE, input_variables=["context", "chat_history", "question", "lang"]
        )
        self.memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=True)

    def query(self, q: str, lang: str, verbose=False):
        qa_chain = self.get_chain(verbose)
        result = qa_chain.invoke({"question": q, "lang": lang})
        return result["answer"]

    def stream(self, q: str, lang: str, verbose=False):
        qa_chain = self.get_chain(verbose)
        sem = threading.Semaphore(0)
        thread = threading.Thread(target=execute_query, args=(qa_chain, {"question": q, "lang": lang}, sem))
        thread.start()

        while True:
            if self.token_queue.empty():
                if sem.acquire(blocking=False):
                    break

                continue
            token = self.token_queue.get()

            while True:
                pos = get_punctuation_pos(token)
                if pos == -1:
                    yield token
                    break
                yield token[:pos]
                yield token[pos]
                if pos + 1 == len(token):
                    break
                token = token[pos + 1 :]

    def get_chat_llm(self, chat_model, streaming=False):
        params = dict(model=chat_model, dashscope_api_key=self.dashscope_api_key, temperature=0, streaming=streaming)
        if streaming:
            params.update(dict(callbacks=[StreamingCallbackHandler(self.token_queue)]))

        return Tongyi(**params)

    def get_chain(self, verbose=False):
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            verbose=verbose,
            combine_docs_chain_kwargs={"prompt": self.prompt},
        )

        return qa_chain

    def get_embedding(self, embedding_model):
        return DashScopeEmbeddings(model=embedding_model, dashscope_api_key=self.dashscope_api_key)

    def get_vectorstore(self, db_dir, collection_name, embedding_model):
        return Chroma(
            collection_name=collection_name,
            persist_directory=db_dir,
            embedding_function=self.get_embedding(embedding_model=embedding_model),
        )


def execute_query(qa, query, sem):
    ret = qa.stream(query)
    try:
        ret = list(ret)
    finally:
        sem.release()

    return ret


def get_punctuation_pos(token):
    for idx, c in enumerate(token):
        if c in punctuation:
            return idx

    return -1


if __name__ == "__main__":
    vectorstore_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir, "vectorstore")
    )
    rag_llm = RAGLLM(vectorstore_path=vectorstore_path, collection_name="银行与保险知识库", streaming=True)
    for s in rag_llm.stream("分红型储蓄有咩优势", "zh"):
        print(s)
    for s in rag_llm.stream("第二点展开说下", "zh"):
        print(s)
