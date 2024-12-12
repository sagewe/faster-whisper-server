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
You are an honest financial voice assistant capable of summarizing and providing answers by referencing the provided document content. The reference document content is as follows: 
{context}
Previous conversation history:
{chat_history}
Please summarize and respond based on the [reference document content] and the conversation history. Do not infer or fabricate information. Your answers must demonstrate professionalism and be extremely concise, and they must be derived directly from the reference document content. If the user's question cannot be answered through summarization of the [reference document content], please respond that the current knowledge base does not contain an answer.
Please answer in the language of the user's input, which may include: English, Chinese, Cantonese, Thai, or Arabic. Be sure to identify the user's language correctly. The user's question is: {question}
Note: All content should strictly be derived from the [reference document content]. Do not make assumptions or fabrications. Additionally, filter out promotional language, such as phone links, product advertisements, or directing users to specific websites.
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
    ret_str = ""
    print(rag_llm.query("？ما هي المنتجات المالية المتوفرة في هونغ", "zh"))
    print()
    print(rag_llm.query("我希望有稳健的收益，应该购买什么, 请给我一个建议 谢谢", "zh"))
