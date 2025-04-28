# Code Repository Analyzer and Indexer

This project is a Python-based tool that automates the process of analyzing and indexing code repositories. It clones a given repository, flattens its structure, generates summaries for each code file, and indexes the content using vector stores for efficient querying.

## Features

- Repository cloning and flattening
- Text splitting for large files
- Asynchronous summary generation using OctoAI
- Vector indexing using Pinecone and LlamaIndex
- Integration with Langchain for embeddings

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/code-repo-analyzer.git
   cd code-repo-analyzer
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the project root and add the following:
   ```
   OCTOAI_API_KEY=your_octoai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   ```

## Usage

1. Modify the `main()` function in the script to specify the repository URL and output folder:
   ```python
   repo_processor = RepoProcessor("https://github.com/python/mypy.git", "flat")
   ```

2. Run the script:
   ```
   python main.py
   ```

The script will clone the repository, process its files, generate summaries, and index the content. Once complete, you can use the created index for querying and analysis.

## Dependencies

This project relies on several Python libraries. See the `requirements.txt` file for a complete list.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open-source and available under the MIT License.

