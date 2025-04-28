# Code Repository Analyzer and Summarizer

This project is a Python-based tool that clones a GitHub repository, processes its code files, generates summaries for each file and function, and stores the results in a vector database for efficient querying and analysis.

## Features

- Clone and flatten a GitHub repository
- Split code files into manageable chunks
- Generate summaries for code files and functions using AI
- Store summaries and code in a Pinecone vector database
- Create a searchable index using LlamaIndex

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

1. Modify the `main()` function in the script to specify the GitHub repository you want to analyze:
   ```python
   repo_url = "https://github.com/username/repo.git"
   ```

2. Run the script:
   ```
   python main.py
   ```

The script will clone the repository, process its files, generate summaries, and store the results in the specified Pinecone index.

## Dependencies

This project relies on several external libraries and APIs:

- OctoAI for text generation
- Pinecone for vector storage
- LlamaIndex for creating searchable indices
- Langchain for text processing
- Pydantic for data validation
- aiohttp for asynchronous HTTP requests

Please refer to the `requirements.txt` file for a complete list of dependencies.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

