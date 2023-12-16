import re
from typing import List
import sys
import ast

def exec_with_return(code: str, globals:dict, locals:dict):
    a = ast.parse(code)
    last_expression = None
    if a.body:
        if isinstance(a_last := a.body[-1], ast.Expr):
            last_expression = ast.unparse(a.body.pop())
        elif isinstance(a_last, ast.Assign):
            last_expression = ast.unparse(a_last.targets[0])
        elif isinstance(a_last, (ast.AnnAssign, ast.AugAssign)):
            last_expression = ast.unparse(a_last.target)
    exec(ast.unparse(a), globals, locals)
    if last_expression:
        return eval(last_expression, globals, locals)

def extract_numbers(target_string: str, filename: str) -> List[str]:
    try:
        with open(filename, 'r') as f:
            text = f.read()
    except OSError as e:
        print(f"Error reading file {e.filename}\n{e.strerror}")
        sys.exit(1)

    pattern = re.compile(f"{target_string} (\d+\.\d+)")
    matches = re.findall(pattern, text)
    matches = [float(m) for m in matches]
    return matches


def extract_namespace(filename: str):
    loc = {}
    try:
        with open(filename, 'r') as file:
            for line in file:
                match = re.search(r'args\s*=\s*(.*)', line)
                if match:
                    s = match.group(1)
                    s = s.replace("Namespace", "", 1)
                    s = s.replace("(","")
                    s = s.replace(")", "")
                    exps = s.split(",")
                    for i in exps:
                        i = i.strip()
                        exec(i, {}, loc)
                    return loc
    except OSError as e:
        print(f"Error reading file {e.filename}\n{e.strerror}")
        sys.exit(1)

    return None

def select_keys(input_dict, keys_to_select):
    return {key: input_dict[key] for key in keys_to_select if key in input_dict}


