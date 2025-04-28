import dotenv
dotenv.load_dotenv()

import os
import glob
import json
import asyncio
import time
import aiohttp
from typing import List, Tuple, Dict, Any
from octoai.text_gen import ChatMessage, ChatCompletionResponseFormat
from octoai.client import AsyncOctoAI, OctoAI
from pydantic import BaseModel
from ast import literal_eval
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

class RepoManager:
    @staticmethod
    def clone_repo(repo_url: str) -> None:
        os.system(f'git clone {repo_url}')

    @staticmethod
    def flatten_repo(folder_path: str) -> None:
        files = glob.glob(os.path.join(folder_path, '**'), recursive=True)
        files = [file for file in files if os.path.isfile(file)]
        flat_folder_path = 'flat'
        for file in files:
            file_name = os.path.basename(file)
            new_file_path = os.path.join(flat_folder_path, file_name)
            os.rename(file, new_file_path)

class FileProcessor:
    @staticmethod
    def read_files(folder_path: str) -> List[Tuple[str, str, str]]:
        file_tuples = []
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                with open(file_path, 'r') as file:
                    file_text = file.read()
                    file_type = os.path.splitext(file_name)[1]
                    file_tuples.append((file_text, file_name, file_type))
        return file_tuples

class TextSplitter:
    @staticmethod
    def split_files(file_tuples: List[Tuple[str, str, str]], chunk_size: int = 10000, chunk_overlap: int = 1000) -> List:
        splitter = RecursiveCharacterTextSplitter.from_language("python", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        file_text_list = [file_tuple[0] for file_tuple in file_tuples]
        file_info_list = [{'file_name': file_tuple[1], 'file_type': file_tuple[2]} for file_tuple in file_tuples]
        python_output = splitter.create_documents(file_text_list, file_info_list)
        return python_output

class Summary(BaseModel):
    summaries: List[str]

class Summarizer:
    def __init__(self):
        self.client = OctoAI(api_key=os.getenv("OCTOAI_API_KEY"))
        self.async_client = AsyncOctoAI(api_key=os.getenv("OCTOAI_API_KEY"))
        self.sem = asyncio.Semaphore(4)

    async def generate_summaries(self, document) -> List[str]:
        await self.sem.acquire()
        response = await self.client.text_gen.create_chat_completion(
            max_tokens=512,
            messages=[
                ChatMessage(
                    content="You are an expert coder that creates summaries of code files.",
                    role="system"
                ),
                ChatMessage(
                    content=f"Take this code file and create a list of summaries for each class and function in the block of code. Here is the code: {document}",
                    role="user"
                )
            ],
            model="mistral-7b-instruct-v0.3",
            presence_penalty=0,
            temperature=0,
            top_p=1,
            response_format=ChatCompletionResponseFormat(type='json_object', schema=Summary.model_json_schema())
        )
        time.sleep(1)
        self.sem.release()
        return json.loads(response.choices[0].message.content)['summaries']

    def generate_sync_summary(self, document) -> str:
        response = self.client.text_gen.create_chat_completion(
            max_tokens=4000,
            messages=[
                ChatMessage(
                    content="You are an expert coder that creates summaries of code files. Respond only following the provided json schema.",
                    role="system"
                ),
                ChatMessage(
                    content=f"Take this code file and create a list of summaries for what this file does. Here is the code: {document.page_content}",
                    role="user"
                )
            ],
            model="meta-llama-3-8b-instruct",
            presence_penalty=0,
            temperature=0,
            top_p=1,
            response_format=ChatCompletionResponseFormat(type='json_object', schema=Summary.model_json_schema())
        )
        return response.choices[0].message.content

class PineconeManager:
    def __init__(self):
        self.index_name = 'hackathon'
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def add_docs_to_pinecone(self, docs, start: int, python_output: List) -> None:
        summarizer = Summarizer()
        all_summaries = [summarizer.generate_sync_summary(doc) for doc in docs]
        all_j_summ = []

        for i, sum in enumerate(all_summaries):
            try:
                all_j_summ.append(json.loads(sum)['summaries'])
            except:
                try:
                    all_j_summ.append(literal_eval('[' + sum.split('[')[1].split(']')[0] + ']'))
                except:
                    all_j_summ.append([])
                    print(f'error for element {start + i}')

        all_docs = []
        for i, summ in enumerate(all_j_summ):
            for s in summ:
                doc = python_output[i].copy()
                doc.metadata['code'] = doc.page_content
                doc.page_content = s
                all_docs.append(doc)
      
        PineconeVectorStore.from_documents(
            all_docs,
            index_name=self.index_name,
            embedding=self.embeddings,
            namespace='codebase'
        )
        print(f"Success upload: {start} to {start + len(docs)}")

class LlamaIndexManager:
    def __init__(self, pinecone_index):
        self.vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

    def create_index(self, documents):
        return VectorStoreIndex.from_documents(documents, storage_context=self.storage_context)

# Example Running Code

repo_url = 'https://github.com/python/mypy.git'

RepoManager.clone_repo(repo_url)
RepoManager.flatten_repo('flat')

file_tuples = FileProcessor.read_files('flat')
python_output = TextSplitter.split_files(file_tuples)

summarizer = Summarizer()
summaries = asyncio.run(summarizer.generate_summaries(pythonOutput))

pinecone_manager = PineconeManager()
pinecone_manager.add_docs_to_pinecone(python_output, 0, python_output)

pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pinecone.Index('hackathon')

llama_manager = LlamaIndexManager(pinecone_index)
index = llama_manager.create_index(python_output)