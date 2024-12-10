import os
import pickle
import argparse
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader


embedding_model = "text-embedding-v1"
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
db_name = "rag-chroma"


def load_urls(path):
    url_list = []
    with open(path, "r") as fin:
        for line in fin:
            line = line.strip()
            if line.startswith("http"):
                url_list.append(line)

    return url_list


def get_doc_splits():
    if args.source == "doc":
        files = os.listdir(args.input_path)
        docs_splits_list = []
        for file in files:
            docs = TextLoader(args.input_path + "/" + file).load()
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
            )
            doc_splits = text_splitter.split_documents(docs)
            docs_splits_list.extend(doc_splits)
    else:
        urls = load_urls(args.input_path)
        docs = []
        for url in urls:
            try:
                docs.append(WebBaseLoader(url).load())
            except BaseException:
                continue
        docs_list = [item for sublist in docs for item in sublist]

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
        )
        docs_splits_list = text_splitter.split_documents(docs_list)

    if args.output_doc_splits_path is not None:
        with open(args.output_doc_splits_path, "wb") as fout:
            fout.write(pickle.dumps(docs_splits_list))

    return docs_splits_list


def build_vectorstore(docs):
    embedding = DashScopeEmbeddings(model=embedding_model, dashscope_api_key=dashscope_api_key)

    if args.db_init:
        vectorstore = Chroma.from_documents(
            documents=docs, collection_name=db_name, embedding=embedding, persist_directory=args.persist_dir
        )
    else:
        vectorstore = Chroma(
            collection_name="rag-chroma", persist_directory=args.persist_dir, embedding_function=embedding
        )
        vectorstore.add_documents(documents=docs)

    vectorstore.persist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--source", type=str, default="doc", choices=["doc", "url"])
    parser.add_argument("--output_doc_splits_path", type=str, default=None)
    parser.add_argument("--persist_dir", type=str)
    parser.add_argument("--db_init", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--chunk_overlap", type=int, default=256)

    args = parser.parse_args()

    docs_splits = get_doc_splits()
    build_vectorstore(docs_splits)
