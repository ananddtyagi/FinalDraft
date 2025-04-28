import os
import glob
import dotenv
import asyncio
import time
import aiohttp
import json
from typing import List, Tuple
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from octoai.text_gen import ChatMessage, ChatCompletionResponseFormat
from octoai.client import AsyncOctoAI, OctoAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from ast import literal_eval

dotenv.load_dotenv()


class FileProcessor:
    def __init__(self, repo_url: str, local_folder: str):
        self.repo_url = repo_url
        self.local_folder = local_folder
        self.flat_folder_path = "flat"

    def clone_repo(self):
        os.system(f"git clone {self.repo_url}")

    def flatten_repo(self):
        files = glob.glob(os.path.join(self.local_folder, "**"), recursive=True)
        files = [file for file in files if os.path.isfile(file)]

        for file in files:
            file_name = os.path.basename(file)
            new_file_path = os.path.join(self.flat_folder_path, file_name)
            os.rename(file, new_file_path)

    def load_files(self) -> List[Tuple[str, str, str]]:
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
    def __init__(self, language: str, chunk_size: int, chunk_overlap: int):
        self.splitter = RecursiveCharacterTextSplitter.from_language(
            language, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def split_texts(self, file_texts: List[str], file_info: List[dict]) -> List:
        return self.splitter.create_documents(file_texts, file_info)


class Summary(BaseModel):
    summaries: List[str]


class SummaryGenerator:
    def __init__(self, api_key: str):
        self.client = OctoAI(api_key=api_key)
        self.async_client = AsyncOctoAI(api_key=api_key)
        self.sem = asyncio.Semaphore(4)

    async def generate_summaries(self, document):
        async with self.sem:
            response = await self.client.text_gen.create_chat_completion(
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
            await asyncio.sleep(1)
            return json.loads(response.choices[0].message.content)["summaries"]

    async def get_all_resps(self, python_output: List):
        async with aiohttp.ClientSession() as session:
            resps = await asyncio.gather(
                *[self.generate_summaries(doc.page_content) for doc in python_output]
            )
            return resps

    def generate_response(self, document):
        response = self.client.text_gen.create_chat_completion(
            max_tokens=4000,
            messages=[
                ChatMessage(
                    content="You are an expert coder that creates summaries of code files. Respond only following the provided json schema.",
                    role="system",
                ),
                ChatMessage(
                    content=f"Take this code file and create a list of summaries for what this file does. Here is the code: {document.page_content}",
                    role="user",
                ),
            ],
            model="meta-llama-3-8b-instruct",
            presence_penalty=0,
            temperature=0,
            top_p=1,
            response_format=ChatCompletionResponseFormat(
                type="json_object", schema=Summary.model_json_schema()
            ),
        )
        return response.choices[0].message.content

    def process_summaries(self, summaries):
        j_summ = []
        for sum in summaries:
            try:
                j_summ.append(json.loads(sum)["summaries"])
            except:
                try:
                    j_summ.append(
                        literal_eval("[" + sum.split("[")[1].split("]")[0] + "]")
                    )
                except:
                    j_summ.append([])
        return j_summ


class PineconeProcessor:
    def __init__(self, api_key: str, index_name: str):
        from pinecone import Pinecone

        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.pci = self.pc.Index(index_name)

        self.model_name = "BAAI/bge-base-en-v1.5"
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def add_docs_to_pinecone(self, docs, summaries, index):
        all_docs = []
        for i, summ in enumerate(summaries):
            for s in summ:
                doc = docs[i].copy()
                doc.metadata["code"] = doc.page_content
                doc.page_content = s
                all_docs.append(doc)

        PineconeVectorStore.from_documents(
            all_docs,
            index_name=self.index_name,
            embedding=self.embeddings,
            namespace="codebase",
        )
        print(f"Success upload: {index} to {index + len(docs)}")

    def upload_batches(self, docs, batch_size=10):
        start = 0
        while start < len(docs):
            self.add_docs_to_pinecone(docs[start : start + batch_size], start)
            start += batch_size


if __name__ == "__main__":
    # Example running code
    repo_url = "https://github.com/python/mypy.git"
    local_folder = "mypy"
    api_key = os.getenv("OCTOAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    # Step 1: Clone repository and flatten files
    processor = FileProcessor(repo_url, local_folder)
    processor.clone_repo()
    processor.flatten_repo()

    # Step 2: Load and split files
    file_tuples = processor.load_files()
    file_text_list = [file_tuple[0] for file_tuple in file_tuples]
    file_info_list = [
        {"file_name": file_tuple[1], "file_type": file_tuple[2]}
        for file_tuple in file_tuples
    ]

    splitter = TextSplitter(language="python", chunk_size=10000, chunk_overlap=1000)
    pythonOutput = splitter.split_texts(file_text_list, file_info_list)

    # Step 3: Generate and process summaries
    summary_gen = SummaryGenerator(api_key=api_key)
    asyncio.run(summary_gen.get_all_resps(pythonOutput))

    all_summaries = summary_gen.process_summaries(all_summaries)

    # Step 4: Upload to Pinecone
    pinecone_processor = PineconeProcessor(
        api_key=pinecone_api_key, index_name="hackathon"
    )
    pinecone_processor.upload_batches(pythonOutput, batch_size=10)
