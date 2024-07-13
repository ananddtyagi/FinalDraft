import ast
import nbformat
from typing import List, Dict, Union


# Define our block types
class CodeBlock:
    def __init__(self, type: str, name: str, content: str):
        self.type = type
        self.name = name
        self.content = content


def read_notebook(notebook_path: str) -> List[str]:
    with open(notebook_path, "r") as file:
        notebook = nbformat.read(file, nbformat.NO_CONVERT)
        cells = []
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cells.append(cell.source)
        return cells


def parse_code_cells(cells: List[str]) -> List[CodeBlock]:
    blocks = []
    current_class = None
    current_function = None

    for cell in cells:
        try:
            tree = ast.parse(cell)
        except SyntaxError:
            # If we can't parse the cell, treat it as a raw code block
            blocks.append(CodeBlock("raw", "", cell))
            continue

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                if current_class:
                    # Finish the previous class block
                    blocks.append(current_class)
                current_class = CodeBlock("class", node.name, ast.unparse(node))
            elif isinstance(node, ast.FunctionDef):
                if current_class:
                    # If we're in a class, add this function to the class content
                    current_class.content += "\n\n" + ast.unparse(node)
                else:
                    # If not in a class, create a new function block
                    blocks.append(CodeBlock("function", node.name, ast.unparse(node)))
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Str):
                # This is a comment (docstring)
                comment = ast.unparse(node).strip()
                if current_class:
                    current_class.content += "\n" + comment
                elif blocks and blocks[-1].type == "comment":
                    # Combine with previous comment block
                    blocks[-1].content += "\n" + comment
                else:
                    blocks.append(CodeBlock("comment", "", comment))
            else:
                # This is a standalone line of code
                code = ast.unparse(node)
                blocks.append(CodeBlock("raw", "", code))

    # Add the last class if there is one
    if current_class:
        blocks.append(current_class)

    return blocks


def parse_notebook(notebook_path: str) -> List[CodeBlock]:
    cells = read_notebook(notebook_path)
    return parse_code_cells(cells)


# Example usage
# blocks = parse_notebook("repotofiles.ipynb")
# for block in blocks:
#     # print(f"Type: {block.type}, Name: {block.name}")
#     print(block.content)
#     print("---")
