from pyclbr import readmodule
import re
import nbformat
from typing import List
import os
from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI


def read_notebook(notebook_path: str) -> List[str]:
    with open(notebook_path, "r") as file:
        notebook = nbformat.read(file, nbformat.NO_CONVERT)
        cells = []
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cells.append(cell.source)
        return cells


def call_claude(prompt: str) -> str:
    anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    responses = anthropic.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    return responses.content[0].text


def call_openai(prompt: str) -> str:
    openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    responses = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )

    return responses.choices[0].message.content


def call_llm(prompt: str) -> str:
    # return call_claude(prompt)
    return call_openai(prompt)


def extract_code(raw_response):
    # Extract code from the response
    code = re.search(r"```python\n(.*?)\n```", raw_response, re.DOTALL)
    if code:
        return code.group(1)
    else:
        raise ValueError("No code found in response")


def process_notebook(notebook_path: str, output_dir: str):
    # Read and combine code cells
    cells = read_notebook(notebook_path)
    all_code = "\n\n".join(cells)

    # Restructure code using Claude
    restructure_prompt = f"""
    Restructure the given Python code into a set of reusable classes or utility functions. Do the following:
    
    1. Given this code create a set of coherent, reusable classes or a set of related utility functions.
    2. Ensure all code follows best practices and is well-documented.
    3. Make the code as modular and flexible as possible. 
    4. Add type hints and docstrings.
    5. Add example running code
    
    VERY IMPORTANT: DO NOT ACTUALLY CHANGE ANY OF THE CODE. ASSUME IT WORKS AS IS AND DON'T ADD OR CHANGE ANY NEW TYPE DEFINITIONS.
    
    Return only the restructured Python code without any explanations.

    Here's the code:

    {all_code}
    """
    restructured_code = call_llm(restructure_prompt)
    print(restructured_code)
    print("__________________________________________________________")
    restructured_code = extract_code(restructured_code)

    # Generate README
    followup_documents_prompt = f"""
    Given the following Python code, write an appropriate README.md file that I can publish alongside the repo:

    {restructured_code}

    The README should include:
    1. A brief description of what the code does
    2. Installation instructions
    3. Usage examples
    4. Any dependencies or requirements
    5. Information about contributions or licensing, if applicable
    
    Then, create a requirements.txt file listing all necessary dependencies. 
    
    YOU MUST Separate the two files by the following separator "----requirements.txt----".
    
    Only return the readme content and requirements.txt content and no extra information.
    """
    followup_documents = call_llm(followup_documents_prompt)
    print(followup_documents)
    readme, requirements = followup_documents.split("----requirements.txt----")

    # # Generate requirements.txt
    # requirements_prompt = f"""
    # Based on the following Python code, create a requirements.txt file listing all necessary dependencies:

    # {restructured_code}

    # Include only the package names and versions, one per line.
    # """
    # requirements_content = call_llm(requirements_prompt)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Write files
    with open(os.path.join(output_dir, "main.py"), "w") as f:
        f.write(restructured_code)

    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(readme)

    with open(os.path.join(output_dir, "requirements.txt"), "w") as f:
        f.write(requirements)

    print(f"Files generated in {output_dir}")


if __name__ == "__main__":
    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

    notebook_path = input("Enter the path to your Jupyter notebook: ")
    output_dir = input("Enter the output directory for the generated files: ")
    process_notebook(notebook_path, output_dir)
