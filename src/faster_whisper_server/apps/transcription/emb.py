import pickle
import argparse
import json
import sys
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_urls(path):
    url_list = []
    with open(path, "r") as fin:
        for line in fin:
            line = line.strip()
            if line.startswith("http"):
                url_list.append(line)

    return url_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_urls_path", type=str)
    parser.add_argument("--output_doc_splits_path", type=str)
    parser.add_argument("--persist_dir", type=str)

    args = parser.parse_args()

    urls = load_urls(args.input_urls_path)

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1024, chunk_overlap=512)
    doc_splits = text_splitter.split_documents(docs_list)

    with open(args.output_doc_splits_path, "wb") as fout:
        fout.write(pickle.dumps(doc_splits))

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=DashScopeEmbeddings(
            model="text-embedding-v1",
            # dashscope_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            dashscope_api_key="sk-a16d5f40946d4e50867a1635c65a610c",
        ),
        persist_directory=args.persist_dir,
    )
    # retriever = vectorstore.as_retriever()
