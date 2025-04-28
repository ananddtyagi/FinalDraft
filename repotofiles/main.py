import os
import glob
import json
import time
import asyncio
import aiohttp
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter
from octoai.text_gen import ChatMessage, ChatCompletionResponseFormat
from octoai.client import AsyncOctoAI, OctoAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import (
    PineconeVectorStore as LlamaPineconeVectorStore,
)
from llama_index.core import VectorStoreIndex, StorageContext

load_dotenv()


class Summary(BaseModel):
    summaries: List[str]


class RepoProcessor:
    def __init__(self, repo_url: str, flat_folder_path: str):
        self.repo_url = repo_url
        self.flat_folder_path = flat_folder_path

    def clone_repo(self) -> None:
        """Clone the repository."""
        os.system(f"git clone {self.repo_url}")

    def flatten_repo(self) -> None:
        """Flatten the repository structure."""
        files = glob.glob(os.path.join(self.flat_folder_path, "**"), recursive=True)
        files = [file for file in files if os.path.isfile(file)]

        for file in files:
            file_name = os.path.basename(file)
            new_file_path = os.path.join(self.flat_folder_path, file_name)
            os.rename(file, new_file_path)

    def read_files(self) -> List[Tuple[str, str, str]]:
        """Read files and return a list of tuples containing file content, name, and type."""
        file_tuples = []
        for file_name in os.listdir(self.flat_folder_path):
            file_path = os.path.join(self.flat_folder_path, file_name)
            if os.path.isfile(file_path):
                with open(file_path, "r") as file:
                    file_text = file.read()
                    file_type = os.path.splitext(file_name)[1]
                    file_tuples.append((file_text, file_name, file_type))
        return file_tuples


class TextSplitter:
    @staticmethod
    def split_text(file_tuples: List[Tuple[str, str, str]]) -> List[Dict[str, Any]]:
        """Split text into chunks using RecursiveCharacterTextSplitter."""
        splitter = RecursiveCharacterTextSplitter.from_language(
            "python", chunk_size=10000, chunk_overlap=1000
        )
        file_text_list = [file_tuple[0] for file_tuple in file_tuples]
        file_info_list = [
            {"file_name": file_tuple[1], "file_type": file_tuple[2]}
            for file_tuple in file_tuples
        ]
        return splitter.create_documents(file_text_list, file_info_list)


class SummaryGenerator:
    def __init__(self):
        self.client = OctoAI(api_key=os.getenv("OCTOAI_API_KEY"))
        self.async_client = AsyncOctoAI(api_key=os.getenv("OCTOAI_API_KEY"))
        self.semaphore = asyncio.Semaphore(4)

    async def generate_summary(self, document: str) -> List[str]:
        """Generate summary for a given document."""
        async with self.semaphore:
            response = await self.async_client.text_gen.create_chat_completion(
                max_tokens=512,
                messages=[
                    ChatMessage(
                        content="You are an expert coder that creates summaries of code files.",
                        role="system",
                    ),
                    ChatMessage(
                        content=f"Take this code file and create a list of summaries for each class and function in the block of code. Here is the code: {document}",
                        role="user",
                    ),
                ],
                model="mistral-7b-instruct-v0.3",
                presence_penalty=0,
                temperature=0,
                top_p=1,
                response_format=ChatCompletionResponseFormat(
                    type="json_object", schema=Summary.model_json_schema()
                ),
            )
            time.sleep(1)
            return json.loads(response.choices[0].message.content)["summaries"]

    async def generate_all_summaries(self, documents: List[str]) -> List[List[str]]:
        """Generate summaries for all documents."""
        async with aiohttp.ClientSession():
            return await asyncio.gather(
                *[self.generate_summary(doc) for doc in documents]
            )


class VectorStoreManager:
    def __init__(self, index_name: str):
        self.index_name = index_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.pinecone_index = self.pinecone_client.Index(index_name)

    def add_documents_to_pinecone(
        self, documents: List[Dict[str, Any]], summaries: List[List[str]]
    ) -> None:
        """Add documents and their summaries to Pinecone."""
        all_docs = []
        for doc, summary in zip(documents, summaries):
            for s in summary:
                new_doc = doc.copy()
                new_doc.metadata["code"] = new_doc.page_content
                new_doc.page_content = s
                all_docs.append(new_doc)

        PineconeVectorStore.from_documents(
            all_docs,
            index_name=self.index_name,
            embedding=self.embeddings,
            namespace="codebase",
        )

    def create_llama_index(self, documents: List[Dict[str, Any]]) -> VectorStoreIndex:
        """Create a LlamaIndex from documents."""
        vector_store = LlamaPineconeVectorStore(pinecone_index=self.pinecone_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )


def main():
    repo_processor = RepoProcessor("https://github.com/python/mypy.git", "flat")
    repo_processor.clone_repo()
    repo_processor.flatten_repo()
    file_tuples = repo_processor.read_files()

    text_splitter = TextSplitter()
    split_documents = text_splitter.split_text(file_tuples)

    summary_generator = SummaryGenerator()
    summaries = asyncio.run(
        summary_generator.generate_all_summaries(
            [doc.page_content for doc in split_documents]
        )
    )

    vector_store_manager = VectorStoreManager("hackathon")
    vector_store_manager.add_documents_to_pinecone(split_documents, summaries)
    llama_index = vector_store_manager.create_llama_index(split_documents)

    print("Processing complete. Documents indexed and ready for querying.")


if __name__ == "__main__":
    main()
