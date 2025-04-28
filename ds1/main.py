import os
import glob
import json
import asyncio
import aiohttp
from typing import List, Tuple
from ast import literal_eval
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter
from octoai.text_gen import ChatMessage, ChatCompletionResponseFormat
from octoai.client import AsyncOctoAI, OctoAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

class Summary(BaseModel):
    summaries: List[str] = Field(default_factory=list)

class GitRepoProcessor:
    def __init__(self, repo_url: str, flat_folder_path: str):
        self.repo_url = repo_url
        self.flat_folder_path = flat_folder_path

    def clone_repo(self) -> None:
        os.system(f"git clone {self.repo_url} {self.flat_folder_path}")

    def flatten_repo(self) -> None:
        files = glob.glob(os.path.join(self.flat_folder_path, '**'), recursive=True)
        files = [file for file in files if os.path.isfile(file)]
        for file in files:
            file_name = os.path.basename(file)
            new_file_path = os.path.join(self.flat_folder_path, file_name)
            if file != new_file_path:
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
                    print(f"Skipping file {file_name} due to encoding issues")
        return file_tuples

class CodeSplitter:
    @staticmethod
    def split_code(file_tuples: List[Tuple[str, str, str]]) -> List:
        splitter = RecursiveCharacterTextSplitter.from_language(
            "python", chunk_size=10000, chunk_overlap=1000
        )
        file_text_list = [file_tuple[0] for file_tuple in file_tuples]
        file_info_list = [{'file_name': file_tuple[1], 'file_type': file_tuple[2]} for file_tuple in file_tuples]
        return splitter.create_documents(file_text_list, metadatas=file_info_list)

class SummaryGenerator:
    def __init__(self, api_key: str):
        self.client = OctoAI(api_key=api_key)
        self.async_client = AsyncOctoAI(api_key=api_key)
        self.sem = asyncio.Semaphore(4)

    async def generate_summary(self, document: str) -> List[str]:
        async with self.sem:
            try:
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
            except Exception as e:
                print(f"Error generating summary: {str(e)}")
                return []

    async def generate_all_summaries(self, documents: List[str]) -> List[List[str]]:
        async with aiohttp.ClientSession():
            return await asyncio.gather(*[self.generate_summary(doc) for doc in documents])

    def generate_file_summary(self, document) -> str:
        try:
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
        except Exception as e:
            print(f"Error generating file summary: {str(e)}")
            return json.dumps({"summaries": []})

class PineconeManager:
    def __init__(self, api_key: str, index_name: str):
        self.api_key = api_key
        self.index_name = index_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def add_docs_to_pinecone(self, docs: List, start: int) -> None:
        summary_generator = SummaryGenerator(os.getenv("OCTOAI_API_KEY"))
        all_summaries = [summary_generator.generate_file_summary(doc) for doc in docs]
        all_j_summ = []
        for i, sum_str in enumerate(all_summaries):
            try:
                all_j_summ.append(json.loads(sum_str)['summaries'])
            except json.JSONDecodeError:
                try:
                    all_j_summ.append(literal_eval('[' + sum_str.split('[')[1].split(']')[0] + ']'))
                except:
                    all_j_summ.append([])
                    print(f'Error parsing summary for element {start + i}')
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
        print(f"Successfully uploaded: {start} to {start + len(docs)}")

    def batch_upload(self, docs: List, batch_size: int = 10) -> None:
        start = 0
        while start < len(docs):
            self.add_docs_to_pinecone(docs[start:start + batch_size], start)
            start += batch_size

def main():
    repo_processor = GitRepoProcessor("https://github.com/python/mypy.git", "flat")
    repo_processor.clone_repo()
    repo_processor.flatten_repo()
    file_tuples = repo_processor.process_files()

    code_splitter = CodeSplitter()
    pythonOutput = code_splitter.split_code(file_tuples)

    summary_generator = SummaryGenerator(os.getenv("OCTOAI_API_KEY"))
    all_summaries = asyncio.run(summary_generator.generate_all_summaries([doc.page_content for doc in pythonOutput]))

    pinecone_manager = PineconeManager(os.getenv("PINECONE_API_KEY"), "hackathon")
    pinecone_manager.batch_upload(pythonOutput)

if __name__ == "__main__":
    main()