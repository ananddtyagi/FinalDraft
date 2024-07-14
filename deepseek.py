import re
import nbformat
from typing import List, Tuple
import os
from anthropic import Anthropic
from openai import OpenAI
from dotenv import load_dotenv
import requests
import ollama


def read_notebook(notebook_path: str) -> List[str]:
    """
    Read a Jupyter notebook and extract code cells.

    Args:
        notebook_path (str): Path to the Jupyter notebook file.

    Returns:
        List[str]: List of code cell contents.
    """
    with open(notebook_path, "r", encoding="utf-8") as file:
        notebook = nbformat.read(file, nbformat.NO_CONVERT)
        return [cell.source for cell in notebook.cells if cell.cell_type == "code"]


def call_deepseek_coder(prompt: str) -> str:
    """
    Call DeepSeek Coder to review and validate the code.

    Args:
        code (str): The code to be reviewed and validated.

    Returns:
        str: The reviewed and validated code.
    """

    response = ollama.chat(
        model="deepseek-coder-v2",
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return response["message"]["content"]

    # api_key = os.getenv("DEEPSEEK_CODER_API_KEY")
    # if not api_key:
    #     raise ValueError("DEEPSEEK_CODER_API_KEY not found in environment variables")

    # headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    # data = {"code": code}
    # response = requests.post(
    #     "https://api.deepseek.com/v2/coder/review", headers=headers, json=data
    # )
    # if response.status_code == 200:
    #     return response.json().get("reviewed_code")
    # else:
    #     raise ValueError(
    #         f"Failed to call DeepSeek Coder: {response.status_code} - {response.text}"
    #     )


def call_llm(prompt: str, use_claude: bool = True) -> str:
    """
    Call an LLM (Claude or GPT-4) with the given prompt.

    Args:
        prompt (str): The prompt to send to the LLM.
        use_claude (bool): Whether to use Claude (True) or GPT-4 (False).

    Returns:
        str: The LLM's response.
    """
    if use_claude:
        anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = anthropic.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    else:
        openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


def extract_code(raw_response: str) -> str:
    """
    Extract Python code from a Markdown-formatted string.

    Args:
        raw_response (str): The raw response containing Markdown-formatted code.

    Returns:
        str: Extracted Python code.

    Raises:
        ValueError: If no code is found in the response.
    """
    code = re.search(r"```python\n(.*?)\n```", raw_response, re.DOTALL)
    if code:
        return code.group(1)
    raise ValueError("No code found in response")


def restructure_code(all_code: str) -> str:
    """
    Restructure the given code using an LLM.

    Args:
        all_code (str): The original code to be restructured.

    Returns:
        str: The restructured code.
    """
    restructure_prompt = f"""
    Restructure the given Python code into a set of reusable classes or utility functions. Do the following:
    
    1. Create a set of coherent, reusable classes or a set of related utility functions.
    2. Ensure all code follows best practices and is well-documented.
    3. Make the code as modular and flexible as possible. 
    4. Add type hints and docstrings.
    5. Add example running code.
    6. Ensure the restructured code maintains the original functionality.
    7. If any imports are missing, add them at the top of the file.
    8. If you encounter any potential issues or edge cases, add appropriate error handling.
    
    IMPORTANT: Maintain the core functionality of the original code. Do not introduce new features or significantly alter the existing logic.
    
    Return only the restructured Python code without any explanations.

    Here's the code:

    {all_code}
    """

    restructured_code = call_llm(restructure_prompt)
    return extract_code(restructured_code)


def review_and_fix_code(restructured_code: str) -> str:
    """
    Review and fix any issues in the restructured code using an LLM.

    Args:
        restructured_code (str): The restructured code to be reviewed and fixed.

    Returns:
        str: The reviewed and fixed code.
    """
    review_prompt = f"""
    Review the following Python code and fix any issues you find. Focus on:

    1. Ensuring all functions and classes work as intended.
    2. Fixing any syntax errors or logical issues.
    3. Improving error handling and edge case management.
    4. Ensuring all necessary imports are present.
    5. Verifying that the code follows PEP 8 style guidelines.
    6. Checking that all type hints and docstrings are correct and complete.

    If you find any issues, fix them directly in the code. If you're unsure about a particular change, add a comment explaining the potential issue and your suggested fix.

    Return the reviewed and fixed Python code without any additional explanations.

    Here's the code to review and fix:

    {restructured_code}
    """

    reviewed_code = call_deepseek_coder(review_prompt)
    return extract_code(reviewed_code)


def process_notebook(notebook_path: str, output_dir: str):
    """
    Process a Jupyter notebook, restructure its code, review and fix issues, and generate supporting files.

    Args:
        notebook_path (str): Path to the input Jupyter notebook.
        output_dir (str): Directory to save the output files.
    """
    os.makedirs(output_dir, exist_ok=True)

    cells = read_notebook(notebook_path)
    all_code = "\n\n".join(cells)

    print("Restructuring code...")
    restructured_code = restructure_code(all_code)
    print(restructured_code)
    print("Code restructured. Now reviewing and fixing...")
    print("Calling DeepSeek Coder for final review...")
    final_code = review_and_fix_code(restructured_code)
    print("Code review and fix completed.")

    # final_code = call_deepseek_coder(final_code)
    print("DeepSeek Coder review completed.")

    with open(os.path.join(output_dir, "main.py"), "w", encoding="utf-8") as f:
        f.write(final_code)

    followup_documents_prompt = f"""
    Given the following Python code, write an appropriate README.md file that I can publish alongside the repo:

    {final_code}

    The README should include:
    1. A brief description of what the code does
    2. Installation instructions
    3. Usage examples
    4. Any dependencies or requirements
    5. Information about contributions or licensing, if applicable
    
    Then, create a requirements.txt file listing all necessary dependencies. 
    
    Separate the two files by the following separator "----requirements.txt----".
    
    Only return the readme content and requirements.txt content and no extra information.
    """

    followup_documents = call_llm(followup_documents_prompt)

    if "----requirements.txt----" in followup_documents:
        readme, requirements = followup_documents.split("----requirements.txt----")

        with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(readme)

        with open(
            os.path.join(output_dir, "requirements.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(requirements)
    else:
        with open(os.path.join(output_dir, "both"), "w", encoding="utf-8") as f:
            f.write(followup_documents)

    print(f"Files generated in {output_dir}")


if __name__ == "__main__":
    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY found in environment variables"
        )

    notebook_path = input("Enter the path to your Jupyter notebook: ")
    output_dir = input("Enter the output directory for the generated files: ")
    process_notebook(notebook_path, output_dir)
