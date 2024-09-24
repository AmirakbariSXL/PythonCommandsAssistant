import tkinter as tk
from tkinter import ttk
from commands import commands  # Import commands from the separate file
# follow us in telegram channel :@pythonism_xl


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

syntax_text = tk.Text(root, height=10, width=90)
syntax_text.pack(pady=5)


example_label = ttk.Label(root, text="Example:")
example_label.pack(pady=5)

example_text = tk.Text(root, height=30, width=150)
example_text.pack(pady=5)


root.mainloop()
