```markdown
# Repo Summarizer and Pinecone Uploader

This repository contains a toolset to clone a Git repository, process the files, generate summaries of the code, and upload the summarized documents to a Pinecone vector store. It leverages several libraries including `langchain`, `octoai`, `pydantic`, and `pinecone`.

## Features

- **Clone and Flatten Repos:** Clone a Git repository and flatten the directory structure.
- **Text Splitting:** Split long text/code documents into manageable chunks.
- **Automated Summarization:** Generate summaries of code files using an LLM via the OctoAI service.
- **Pinecone Integration:** Upload the summarized documents to a Pinecone vector store for efficient retrieval and similarity search.

## Installation

1. Clone the repository:

   ```sh
   git clone <your-repo-url>
   cd <your-repo-directory>
   ```

2. Create a virtual environment and activate it:

   ```sh
   python -m venv venv
   source venv/bin/activate  # on macOS/Linux
   venv\Scripts\activate  # on Windows
   ```

3. Install the required dependencies:

   ```sh
   pip install -r requirements.txt
   ```

4. Setup environment variables by creating a `.env` file:

   ```
   OCTOAI_API_KEY=your-octoai-api-key
   PINECONE_API_KEY=your-pinecone-api-key
   ```

## Usage

### Example Running Code

1. Clone the repository and flatten the files:

    ```python
    repo_url = "https://github.com/python/mypy.git"
    local_folder = "mypy"

    processor = FileProcessor(repo_url, local_folder)
    processor.clone_repo()
    processor.flatten_repo()
    ```

2. Load and split files:

    ```python
    file_tuples = processor.load_files()
    file_text_list = [file_tuple[0] for file_tuple in file_tuples]
    file_info_list = [{'file_name': file_tuple[1], 'file_type': file_tuple[2]} for file_tuple in file_tuples]

    splitter = TextSplitter(language="python", chunk_size=10000, chunk_overlap=1000)
    pythonOutput = splitter.split_texts(file_text_list, file_info_list)
    ```

3. Generate and process summaries:

    ```python
    summary_gen = SummaryGenerator(api_key=api_key)
    asyncio.run(summary_gen.get_all_resps(pythonOutput))
    all_summaries = summary_gen.process_summaries(all_summaries)
    ```

4. Upload to Pinecone:

    ```python
    pinecone_processor = PineconeProcessor(api_key=pinecone_api_key, index_name='hackathon')
    pinecone_processor.upload_batches(pythonOutput, batch_size=10)
    ```

## Dependencies

- python-dotenv
- aiohttp
- asyncio
- pydantic
- langchain
- pinecone-client
- huggingface-hub
- octoai

## Contributing

Feel free to open issues or submit pull requests for any bugs or feature requests. Contributions are more than welcome!

## License

This repository is licensed under the MIT License. See the `LICENSE` file for more details.
```

