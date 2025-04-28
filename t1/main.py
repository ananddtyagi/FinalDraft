import os
import glob
import json
import time
import asyncio
import aiohttp
from typing import List, Tuple
from dotenv import load_dotenv
from git import Repo
from langchain.text_splitter import RecursiveCharacterTextSplitter
from octoai.text_gen import ChatMessage, ChatCompletionResponseFormat
from octoai.client import AsyncOctoAI, OctoAI
from pydantic import BaseModel
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore as LlamaPineconeVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from ast import literal_eval

load_dotenv()

class Summary(BaseModel):
    summaries: List[str]

class CodeProcessor:
    def __init__(self, repo_url: str, flat_folder_path: str):
        self.repo_url = repo_url
        self.flat_folder_path = flat_folder_path
        self.client = OctoAI(api_key=os.getenv("OCTOAI_API_KEY"))
        self.async_client = AsyncOctoAI(api_key=os.getenv("OCTOAI_API_KEY"))
        self.sem = asyncio.Semaphore(4)

    def clone_repo(self) -> None:
        """Clone the repository."""
        Repo.clone_from(self.repo_url, self.flat_folder_path)

    def flatten_repo(self) -> None:
        """Flatten the repository structure."""
        files = glob.glob(os.path.join(self.flat_folder_path, '**'), recursive=True)
        files = [file for file in files if os.path.isfile(file)]
        for file in files:
            file_name = os.path.basename(file)
            new_file_path = os.path.join(self.flat_folder_path, file_name)
            os.rename(file, new_file_path)

    def read_files(self) -> List[Tuple[str, str, str]]:
        """Read all files in the flattened repo."""
        file_tuples = []
        for file_name in os.listdir(self.flat_folder_path):
            file_path = os.path.join(self.flat_folder_path, file_name)
            if os.path.isfile(file_path):
                with open(file_path, 'r') as file:
                    file_text = file.read()
                    file_type = os.path.splitext(file_name)[1]
                    file_tuples.append((file_text, file_name, file_type))
        return file_tuples

    def split_files(self, file_tuples: List[Tuple[str, str, str]]) -> List:
        """Split files into chunks."""
        splitter = RecursiveCharacterTextSplitter.from_language(
            "python", chunk_size=10000, chunk_overlap=1000
        )
        file_text_list = [ft[0] for ft in file_tuples]
        file_info_list = [{'file_name': ft[1], 'file_type': ft[2]} for ft in file_tuples]
        return splitter.create_documents(file_text_list, file_info_list)

    async def generate_summaries(self, document: str) -> List[str]:
        """Generate summaries for a document."""
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

    async def get_all_summaries(self, documents: List) -> List[List[str]]:
        """Get summaries for all documents."""
        async with aiohttp.ClientSession():
            return await asyncio.gather(*[self.generate_summaries(doc) for doc in documents])

    def generate_response(self, document) -> str:
        """Generate a response for a document."""
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
    def __init__(self, index_name: str):
        self.index_name = index_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.pci = self.pc.Index(index_name)

    def add_docs_to_pinecone(self, docs: List, start: int) -> None:
        """Add documents to Pinecone."""
        processor = CodeProcessor("", "")  # Temporary instance for using generate_response
        all_summaries = [processor.generate_response(doc) for doc in docs]
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

    def create_llama_index(self, docs: List) -> VectorStoreIndex:
        """Create a Llama index from documents."""
        vector_store = LlamaPineconeVectorStore(pinecone_index=self.pci)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_documents(docs, storage_context=storage_context)

def main():
    processor = CodeProcessor("https://github.com/python/mypy.git", "flat")
    processor.clone_repo()
    processor.flatten_repo()
    file_tuples = processor.read_files()
    pythonOutput = processor.split_files(file_tubes)

    vector_store_manager = VectorStoreManager('hackathon')

    start = 30
    batch_size = 10
    while start < len(pythonOutput):
        vector_store_manager.add_docs_to_pinecone(pythonOutput[start:start + batch_size], start)
        start += batch_size

    llama_index = vector_store_manager.create_llama_index(pythonOutput)

if __name__ == "__main__":
    main()