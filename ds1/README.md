# Code Repository Analyzer

This project is a Python-based tool that analyzes and summarizes code repositories. It clones a GitHub repository, processes its files, generates summaries for each code block, and stores the information in a Pinecone vector database for efficient retrieval and analysis.

## Features

- Clones and flattens a Git repository
- Splits code files into manageable chunks
- Generates summaries for code blocks using OctoAI's language models
- Stores summaries and code information in a Pinecone vector database

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/code-repo-analyzer.git
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

1. Update the `main()` function in the script with the desired GitHub repository URL and Pinecone index name:

   ```python
   repo_processor = GitRepoProcessor("https://github.com/username/repo.git", "flat")
   pinecone_manager = PineconeManager(os.getenv("PINECONE_API_KEY"), "your_index_name")
   ```

2. Run the script:
   ```
   python main.py
   ```

The script will clone the repository, process its files, generate summaries, and store the information in the specified Pinecone index.

## Dependencies

- aiohttp
- asyncio
- dotenv
- langchain
- octoai
- pinecone
- pydantic
- typing

For a complete list of dependencies, see the `requirements.txt` file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open-source and available under the MIT License.

