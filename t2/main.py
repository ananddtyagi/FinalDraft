import os
import glob
import json
import asyncio
import aiohttp
from typing import List, Tuple
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from octoai.text_gen import ChatMessage, ChatCompletionResponseFormat
from octoai.client import AsyncOctoAI, OctoAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore as LlamaPineconeVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

load_dotenv()

class Summary(BaseModel):
    summaries: List[str]

class RepoProcessor:
    def __init__(self, repo_url: str, flat_folder_path: str):
        self.repo_url = repo_url
        self.flat_folder_path = flat_folder_path

    def clone_repo(self) -> None:
        os.system(f"git clone {self.repo_url}")

    def flatten_repo(self) -> None:
        files = glob.glob(os.path.join(self.flat_folder_path, '**'), recursive=True)
        files = [file for file in files if os.path.isfile(file)]
        for file in files:
            file_name = os.path.basename(file)
            new_file_path = os.path.join(self.flat_folder_path, file_name)
            os.rename(file, new_file_path)

    def process_files(self) -> List[Tuple[str, str, str]]:
        file_tuples = []
        for file_name in os.listdir(self.flat_folder_path):
            file_path = os.path.join(self.flat_folder_path, file_name)
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        file_text = file.read()
                        file_type = os.path.splitext(file_name)[1]
                        file_tuples.append((file_text, file_name, file_type))
                except UnicodeDecodeError:
                    print(f"Skipping file {file_name} due to encoding issues.")
        return file_tuples

class CodeSplitter:
    @staticmethod
    def split_code(file_tuples: List[Tuple[str, str, str]]):
        splitter = RecursiveCharacterTextSplitter.from_language(
            "python",
            chunk_size=10000,
            chunk_overlap=1000,
        )
        file_text_list = [file_tuple[0] for file_tuple in file_tuples]
        file_info_list = [{'file_name': file_tuple[1], 'file_type': file_tuple[2]} for file_tuple in file_tuples]
        return splitter.create_documents(file_text_list, metadatas=file_info_list)

class SummaryGenerator:
    def __init__(self, api_key: str):
        self.client = OctoAI(api_key=api_key)
        self.async_client = AsyncOctoAI(api_key=api_key)
        self.sem = asyncio.Semaphore(4)

    async def generate_summaries(self, document):
        async with self.sem:
            response = await self.async_client.text_gen.create_chat_completion(
                max_tokens=512,
                messages=[
                    ChatMessage(content="You are an expert coder that creates summaries of code files.", role="system"),
                    ChatMessage(content=f"Take this code file and create a list of summaries for each class and function in the block of code. Here is the code: {document}", role="user")
                ],
                model="mistral-7b-instruct-v0.3",
                presence_penalty=0,
                temperature=0,
                top_p=1,
                response_format=ChatCompletionResponseFormat(type='json_object', schema=Summary.model_json_schema())
            )
            await asyncio.sleep(1)
            return json.loads(response.choices[0].message.content)['summaries']

    async def get_all_summaries(self, documents):
        async with aiohttp.ClientSession() as session:
            return await asyncio.gather(*[self.generate_summaries(doc) for doc in documents])

    def generate_file_summary(self, document):
        response = self.client.text_gen.create_chat_completion(
            max_tokens=4000,
            messages=[
                ChatMessage(content="You are an expert coder that creates summaries of code files. Respond only following the provided json schema.", role="system"),
                ChatMessage(content=f"Take this code file and create a list of summaries for what this file does. Here is the code: {document.page_content}", role="user")
            ],
            model="meta-llama-3-8b-instruct",
            presence_penalty=0,
            temperature=0,
            top_p=1,
            response_format=ChatCompletionResponseFormat(type='json_object', schema=Summary.model_json_schema())
        )
        return response.choices[0].message.content

class VectorStoreManager:
    def __init__(self, index_name: str, api_key: str):
        self.index_name = index_name
        self.api_key = api_key
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def add_docs_to_pinecone(self, docs, start: int):
        all_summaries = [SummaryGenerator(self.api_key).generate_file_summary(doc) for doc in docs]
        all_j_summ = []
        for i, sum in enumerate(all_summaries):
            try:
                all_j_summ.append(json.loads(sum)['summaries'])
            except json.JSONDecodeError:
                try:
                    all_j_summ.append(eval('[' + sum.split('[')[1].split(']')[0] + ']'))
                except:
                    all_j_summ.append([])
                    print(f'error for element {start + i}')
        
        all_docs = []
        for i, summ in enumerate(all_j_summ):
            for s in summ:
                doc = docs[i].copy()
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

    def create_llama_index(self, docs):
        pc = Pinecone(api_key=self.api_key)
        pci = pc.Index(self.index_name)
        vector_store = LlamaPineconeVectorStore(pinecone_index=pci)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_documents(docs, storage_context=storage_context)

def main():
    repo_url = "https://github.com/python/mypy.git"
    flat_folder_path = "flat"
    index_name = "hackathon"
    
    repo_processor = RepoProcessor(repo_url, flat_folder_path)
    repo_processor.clone_repo()
    repo_processor.flatten_repo()
    file_tuples = repo_processor.process_files()
    
    code_splitter = CodeSplitter()
    pythonOutput = code_splitter.split_code(file_tuples)
    
    summary_generator = SummaryGenerator(os.getenv("OCTOAI_API_KEY"))
    all_summaries = asyncio.run(summary_generator.get_all_summaries(pythonOutput))
    
    vector_store_manager = VectorStoreManager(index_name, os.getenv("PINECONE_API_KEY"))
    
    start = 0
    batch_size = 10
    while start < len(pythonOutput):
        vector_store_manager.add_docs_to_pinecone(pythonOutput[start:start + batch_size], start)
        start += batch_size
    
    llama_index = vector_store_manager.create_llama_index(pythonOutput)

if __name__ == "__main__":
    main()