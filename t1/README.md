# Code Repository Analyzer

This project is a Python-based tool that analyzes code repositories, generates summaries for code files, and stores the information in a vector database for efficient retrieval and analysis.

## Description

The Code Repository Analyzer performs the following main functions:
1. Clones a given GitHub repository
2. Flattens the repository structure
3. Reads and splits the code files into manageable chunks
4. Generates summaries for each code chunk using AI models
5. Stores the summaries and code information in a Pinecone vector database
6. Creates a searchable index using LlamaIndex

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/code-repository-analyzer.git
   cd code-repository-analyzer
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

1. Update the `main()` function in the script with the desired GitHub repository URL and output folder:
   ```python
   processor = CodeProcessor("https://github.com/username/repo.git", "output_folder")
   ```

2. Run the script:
   ```
   python main.py
   ```

The script will clone the repository, process the files, generate summaries, and store the information in the Pinecone vector database.

## Dependencies

- Python 3.7+
- See `requirements.txt` for a full list of Python package dependencies

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open-source and available under the MIT License.

