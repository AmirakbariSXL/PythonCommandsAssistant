import tkinter as tk
from tkinter import ttk

#follow us in telegram channel :@pythonism_xl

commands = {
    "print": {
        "syntax": "print(value, ..., sep=' ', end='\\n', file=sys.stdout, flush=False)",
        "example": "print('Hello, World!')\n# Prints Hello, World!"
    },
    "if": {
        "syntax": "if condition:\n    # code",
        "example": "if 5 > 2:\n    print('5 is greater than 2')"
    },
    "for": {
        "syntax": "for variable in sequence:\n    # code",
        "example": "for i in range(5):\n    print(i)"
    },
    "while": {
        "syntax": "while condition:\n    # code",
        "example": "count = 0\nwhile count < 5:\n    print(count)\n    count += 1"
    },
    "def": {
        "syntax": "def function_name(parameters):\n    # code",
        "example": "def greet(name):\n    print(f'Hello, {name}')\ngreet('Alice')"
    },
    "class": {
        "syntax": "class ClassName:\n    def __init__(self, parameters):\n        # code",
        "example": "class Dog:\n    def __init__(self, name):\n        self.name = name\n    def bark(self):\n        print(f'{self.name} says woof!')\nd = Dog('Buddy')\nd.bark()"
    },
    "try": {
        "syntax": "try:\n    # code\nexcept Exception:\n    # code",
        "example": "try:\n    x = int('foo')\nexcept ValueError:\n    print('Invalid number')"
    },
    "lambda": {
        "syntax": "lambda arguments: expression",
        "example": "square = lambda x: x ** 2\nprint(square(5))"
    },
    "list comprehension": {
        "syntax": "[expression for item in iterable]",
        "example": "[x**2 for x in range(10)]"
    },
    "import": {
        "syntax": "import module",
        "example": "import math\nprint(math.sqrt(16))"
    },
    "input": {
        "syntax": "input(prompt)",
        "example": "name = input(Enter your name: )\nprint(name)"
    },
    "len": {
        "syntax": "len(object)",
        "example": "my_list = [1, 2, 3, 4]\nlen(my_list)"
    },
    "map": {
        "syntax": "map(function, iterable)",
        "example": "print(list(map(lambda x: x ** 2, [1, 2, 3])))"
    },
    "filter": {
        "syntax": "filter(function, iterable)",
        "example": "print(list(filter(lambda x: x < 5, [1, 2, 3, 4, 5])))"
    },
    "decorators": {
        "syntax": "@decorator(function)",
        "example": "def my_decorator(func):\n\tdef caller():\n\t\tprint(before called.)"
                   "\n\t\tfunc()\n\t\tprint(after called.)\n\treturn caller\n"
                   "\n@my_decorator\ndef say_hi():\n\tprint(Hi)\nsay_hi()",
    },
}


def show_command_example(event=None):
    command = command_combo.get()
    if command in commands:
        syntax = commands[command]['syntax']
        example = commands[command]['example']
    else:
        syntax = "No syntax available"
        example = "No example available"
    
    syntax_text.delete(1.0, tk.END)
    syntax_text.insert(tk.END, syntax)
    
    example_text.delete(1.0, tk.END)
    example_text.insert(tk.END, example)


def search_commands(*args):
    query = search_var.get().lower()
    filtered_commands = [cmd for cmd in commands if query in cmd.lower()]
    command_combo['values'] = filtered_commands


root = tk.Tk()
root.title("Python Commands with Examples BY Pythonism")


search_var = tk.StringVar()
search_var.trace("w", search_commands)

search_label = ttk.Label(root, text="Search Python Command:")
search_label.pack(pady=5)

search_entry = ttk.Entry(root, textvariable=search_var)
search_entry.pack(pady=5)


command_label = ttk.Label(root, text="Select Python Command:")
command_label.pack(pady=5)

command_combo = ttk.Combobox(root, values=list(commands.keys()))
command_combo.pack(pady=5)
command_combo.bind("<<ComboboxSelected>>", show_command_example)


syntax_label = ttk.Label(root, text="Syntax:")
syntax_label.pack(pady=5)

syntax_text = tk.Text(root, height=5, width=60)
syntax_text.pack(pady=5)


example_label = ttk.Label(root, text="Example:")
example_label.pack(pady=5)

example_text = tk.Text(root, height=10, width=60)
example_text.pack(pady=5)


root.mainloop()
