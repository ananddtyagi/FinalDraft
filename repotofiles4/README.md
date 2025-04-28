# README.md

## Code Repository Manager

This project provides a Python toolkit to facilitate the process of cloning, processing, summarizing, and indexing code repositories. The toolkit leverages several libraries for advanced functionalities, including text splitting, asynchronous API calls, and working with Pinecone and Llama Index for document storage and retrieval.

### Features

- **RepoManager**: Clone and flatten a Git repository.
- **FileProcessor**: Read files from a directory into text tuples.
- **TextSplitter**: Split file content into smaller chunks for easier processing.
- **Summarizer**: Generate summaries of code files using OctoAI.
- **PineconeManager**: Store summarized documents into Pinecone for vector searches.
- **LlamaIndexManager**: Create and manage a Llama vector index from Pinecone-stored documents.

### Installation

1. Clone the repository:
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2. Create and activate a virtual environment (recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up environment variables by creating a `.env` file:
    ```bash
    touch .env
    ```
    Add the following variables to the `.env` file:
    ```env
    OCTOAI_API_KEY=your-octoai-api-key
    PINECONE_API_KEY=your-pinecone-api-key
    ```

### Usage

1. Clone and flatten a repository:
    ```python
    from your_script import RepoManager
    
    repo_url = 'https://github.com/python/mypy.git'
    RepoManager.clone_repo(repo_url)
    RepoManager.flatten_repo('flat')
    ```

2. Process files and split content:
    ```python
    from your_script import FileProcessor, TextSplitter

    file_tuples = FileProcessor.read_files('flat')
    python_output = TextSplitter.split_files(file_tuples)
    ```

3. Generate summaries:
    ```python
    import asyncio
    from your_script import Summarizer

    summarizer = Summarizer()
    summaries = asyncio.run(summarizer.generate_summaries(python_output))
    ```

4. Store documents to Pinecone:
    ```python
    from your_script import PineconeManager

    pinecone_manager = PineconeManager()
    pinecone_manager.add_docs_to_pinecone(python_output, 0, python_output)
    ```

5. Create an index using Llama Index Manager:
    ```python
    from your_script import Pinecone, LlamaIndexManager

    pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pinecone_index = pinecone.Index('hackathon')
    
    llama_manager = LlamaIndexManager(pinecone_index)
    index = llama_manager.create_index(python_output)
    ```

### Dependencies

The project relies on the following Python packages:

- `dotenv`
- `os`
- `glob`
- `json`
- `asyncio`
- `time`
- `aiohttp`
- `pydantic`
- `ast`
- `langchain`
- `langchain_pinecone`
- `langchain_community`
- `pinecone-client`
- `llama_index`
- `octoai`

Make sure these packages are installed, ideally using the provided `requirements.txt`.

### Contributions and License

Contributions are welcome! Please create a pull request or raise an issue for any enhancements or bug fixes. 

This project is licensed under the MIT License.

