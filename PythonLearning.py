import tkinter as tk
from tkinter import ttk

# follow us in telegram channel :@pythonism_xl


commands = {
    "print": {
        "syntax": "print(value, ..., sep=' ', end='\\n', file=sys.stdout, flush=False)",
        "example": "print('Hello', 'World', sep=', ', end='!')\n# Prints: Hello, World!"
    },
    "if": {
        "syntax": "if condition:\n    # code\nelif condition:\n    # code\nelse:\n    # code",
        "example": "x = 10\nif x > 10:\n    print('x is greater than 10')\nelif x < 10:\n    print('x is less than 10')\nelse:\n    print('x is equal to 10')"
    },
    "for": {
        "syntax": "for variable in iterable:\n    # code",
        "example": "for i in range(5):\n    print(i)\n\n# Using enumerate\nfruits = ['apple', 'banana', 'cherry']\nfor index, fruit in enumerate(fruits):\n    print(f'{index}: {fruit}')"
    },
    "while": {
        "syntax": "while condition:\n    # code",
        "example": "import random\nnumber = random.randint(1, 10)\nguess = 0\nwhile guess != number:\n    guess = int(input('Guess the number: '))\nprint('You guessed it!')"
    },
    "def": {
        "syntax": "def function_name(parameters):\n    # code\n    return value",
        "example": "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n-1)\n\nprint(factorial(5))  # Prints: 120"
    },
    "class": {
        "syntax": "class ClassName:\n    def __init__(self, parameters):\n        # constructor code\n    def method(self, parameters):\n        # method code",
        "example": "class BankAccount:\n    def __init__(self, balance=0):\n        self.balance = balance\n    def deposit(self, amount):\n        self.balance += amount\n    def withdraw(self, amount):\n        if self.balance >= amount:\n            self.balance -= amount\n        else:\n            print('Insufficient funds')\n\naccount = BankAccount(100)\naccount.deposit(50)\naccount.withdraw(75)\nprint(account.balance)  # Prints: 75"
    },
    "try": {
        "syntax": "try:\n    # code that might raise an exception\nexcept ExceptionType as e:\n    # code to handle the exception\nelse:\n    # code to run if no exception occurred\nfinally:\n    # code that always runs",
        "example": "def divide(x, y):\n    try:\n        result = x / y\n    except ZeroDivisionError as e:\n        print(f'Error: {e}')\n    else:\n        print(f'Result is {result}')\n    finally:\n        print('Execution completed')\n\ndivide(10, 2)\ndivide(10, 0)"
    },
    "lambda": {
        "syntax": "lambda arguments: expression",
        "example": "# Sort a list of tuples by the second element\npairs = [(1, 'one'), (3, 'three'), (2, 'two')]\nsorted_pairs = sorted(pairs, key=lambda pair: pair[1])\nprint(sorted_pairs)  # Prints: [(1, 'one'), (3, 'three'), (2, 'two')]"
    },
    "list comprehension": {
        "syntax": "[expression for item in iterable if condition]",
        "example": "# Create a list of squares of even numbers from 0 to 9\nsquares = [x**2 for x in range(10) if x % 2 == 0]\nprint(squares)  # Prints: [0, 4, 16, 36, 64]"
    },
    "import": {
        "syntax": "import module\nfrom module import function\nfrom module import function as alias\nfrom module import *",
        "example": "import math\nprint(math.pi)\n\nfrom random import randint\nprint(randint(1, 6))\n\nfrom datetime import datetime as dt\nprint(dt.now())\n\nfrom string import *\nprint(ascii_lowercase)"
    },
    "with": {
        "syntax": "with expression as variable:\n    # code block",
        "example": "import tempfile\nimport os\n\nwith tempfile.TemporaryDirectory() as temp_dir:\n    path = os.path.join(temp_dir, 'temp_file.txt')\n    with open(path, 'w') as f:\n        f.write('Hello, World!')\n    # File is automatically closed and directory is cleaned up"
    },
    "dict comprehension": {
        "syntax": "{key_expr: value_expr for item in iterable if condition}",
        "example": "# Create a dictionary of character frequencies in a string\ntext = 'hello world'\nchar_freq = {char: text.count(char) for char in set(text)}\nprint(char_freq)"
    },
    "generators": {
        "syntax": "def generator_function():\n    yield value",
        "example": "def fibonacci():\n    a, b = 0, 1\n    while True:\n        yield a\n        a, b = b, a + b\n\nfib = fibonacci()\nfor _ in range(10):\n    print(next(fib), end=' ')\n# Prints: 0 1 1 2 3 5 8 13 21 34"
    },
    "decorators": {
        "syntax": "@decorator\ndef function():\n    # code",
        "example": "import time\n\ndef timing_decorator(func):\n    def wrapper(*args, **kwargs):\n        start = time.time()\n        result = func(*args, **kwargs)\n        end = time.time()\n        print(f'{func.__name__} took {end - start:.2f} seconds')\n        return result\n    return wrapper\n\n@timing_decorator\ndef slow_function():\n    time.sleep(1)\n\nslow_function()"
    },
    "async/await": {
        "syntax": "async def function_name():\n    await asyncio.sleep(1)",
        "example": "import asyncio\n\nasync def fetch_data(url):\n    print(f'Fetching data from {url}')\n    await asyncio.sleep(1)  # Simulate network delay\n    print(f'Data fetched from {url}')\n    return f'Data from {url}'\n\nasync def main():\n    urls = ['http://example.com', 'http://example.org', 'http://example.net']\n    tasks = [fetch_data(url) for url in urls]\n    results = await asyncio.gather(*tasks)\n    print(results)\n\nasyncio.run(main())"
    },
    "f-strings": {
        "syntax": "f'string {expression}'",
        "example": "name = 'Alice'\nage = 30\npi = 3.14159\nprint(f'{name} is {age} years old.')\nprint(f'Pi is approximately {pi:.2f}')"
    },
    "set": {
        "syntax": "{value1, value2, ...}",
        "example": "# Set operations\nset1 = {1, 2, 3, 4, 5}\nset2 = {4, 5, 6, 7, 8}\nprint(f'Union: {set1 | set2}')\nprint(f'Intersection: {set1 & set2}')\nprint(f'Difference: {set1 - set2}')\nprint(f'Symmetric Difference: {set1 ^ set2}')"
    },
    "tuple": {
        "syntax": "(value1, value2, ...)",
        "example": "# Tuple unpacking\npoint = (3, 4)\nx, y = point\nprint(f'x: {x}, y: {y}')\n\n# Tuple as a key in dictionary\nlocations = {(40.7128, -74.0060): 'New York City',\n             (51.5074, -0.1278): 'London'}\nprint(locations[(40.7128, -74.0060)])"
    },
    "list methods": {
        "syntax": "list_object.method(arguments)",
        "example": "fruits = ['apple', 'banana', 'cherry']\nfruits.append('date')\nfruits.insert(1, 'blueberry')\nfruits.remove('cherry')\npopped = fruits.pop()\nfruits.sort()\nfruits.reverse()\nprint(fruits)\nprint(f'Popped item: {popped}')"
    },
    "dict methods": {
        "syntax": "dict_object.method(arguments)",
        "example": "person = {'name': 'John', 'age': 30}\nperson.update({'city': 'New York', 'age': 31})\nkeys = person.keys()\nvalues = person.values()\nitems = person.items()\nage = person.get('age', 'Unknown')\nperson.pop('city')\nprint(person)\nprint(f'Keys: {keys}, Values: {values}, Items: {items}')"
    },
    "string methods": {
        "syntax": "string_object.method(arguments)",
        "example": "text = '  Hello, World!  '\nprint(text.strip())\nprint(text.lower())\nprint(text.upper())\nprint(text.replace('World', 'Python'))\nprint(text.split(','))\nprint('world' in text.lower())"
    },
    "file operations": {
        "syntax": "with open(filename, mode) as file:\n    # file operations",
        "example": "# Writing to a file\nwith open('example.txt', 'w') as f:\n    f.write('Hello, World!')\n\n# Reading from a file\nwith open('example.txt', 'r') as f:\n    content = f.read()\n    print(content)\n\n# Appending to a file\nwith open('example.txt', 'a') as f:\n    f.write('\\nPython is awesome!')"
    },
    "exceptions": {
        "syntax": "raise ExceptionType('message')",
        "example": "def divide(x, y):\n    if y == 0:\n        raise ValueError('Cannot divide by zero')\n    return x / y\n\ntry:\n    result = divide(10, 0)\nexcept ValueError as e:\n    print(f'Error: {e}')\nelse:\n    print(f'Result: {result}')"
    },
    "context managers": {
        "syntax": "class ContextManager:\n    def __enter__(self):\n        # setup code\n        return self\n    def __exit__(self, exc_type, exc_value, traceback):\n        # cleanup code",
        "example": "class Timer:\n    def __enter__(self):\n        self.start = time.time()\n        return self\n    def __exit__(self, *args):\n        self.end = time.time()\n        print(f'Elapsed time: {self.end - self.start:.2f} seconds')\n\nwith Timer():\n    time.sleep(1)"
    },
    "argparse": {
        "syntax": "import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('argument', help='description')\nargs = parser.parse_args()",
        "example": "import argparse\n\nparser = argparse.ArgumentParser(description='A simple greeting program')\nparser.add_argument('name', help='Name of the person to greet')\nparser.add_argument('--uppercase', '-u', action='store_true', help='Print the greeting in uppercase')\nargs = parser.parse_args()\n\ngreeting = f'Hello, {args.name}!'\nif args.uppercase:\n    greeting = greeting.upper()\nprint(greeting)\n\n# Run with: python script.py Alice --uppercase"
    },
    "regular expressions": {
        "syntax": "import re\npattern = re.compile(r'regex_pattern')\nmatch = pattern.search(string)",
        "example": "import re\n\ntext = 'The quick brown fox jumps over the lazy dog.'\npattern = re.compile(r'\\b\\w{5}\\b')  # Find all 5-letter words\nmatches = pattern.findall(text)\nprint(matches)  # Prints: ['quick', 'brown', 'jumps']"
    },
    "threading": {
        "syntax": "import threading\nthread = threading.Thread(target=function, args=(arg1, arg2))\nthread.start()\nthread.join()",
        "example": "import threading\nimport time\n\ndef worker(name):\n    print(f'Worker {name} starting')\n    time.sleep(2)\n    print(f'Worker {name} finished')\n\nthreads = []\nfor i in range(5):\n    t = threading.Thread(target=worker, args=(i,))\n    threads.append(t)\n    t.start()\n\nfor t in threads:\n    t.join()\n\nprint('All workers finished')"
    },
    "multiprocessing": {
        "syntax": "import multiprocessing\nprocess = multiprocessing.Process(target=function, args=(arg1, arg2))\nprocess.start()\nprocess.join()",
        "example": "import multiprocessing\nimport time\n\ndef worker(name):\n    print(f'Worker {name} starting')\n    time.sleep(2)\n    print(f'Worker {name} finished')\n\nif __name__ == '__main__':\n    processes = []\n    for i in range(5):\n        p = multiprocessing.Process(target=worker, args=(i,))\n        processes.append(p)\n        p.start()\n\n    for p in processes:\n        p.join()\n\n    print('All workers finished')"
    },
    "decorators with arguments": {
        "syntax": "def decorator_with_args(decorator_arg1, decorator_arg2):\n    def decorator(func):\n        def wrapper(*args, **kwargs):\n            # use decorator_arg1 and decorator_arg2\n            return func(*args, **kwargs)\n        return wrapper\n    return decorator",
        "example": "def repeat(times):\n    def decorator(func):\n        def wrapper(*args, **kwargs):\n            for _ in range(times):\n                result = func(*args, **kwargs)\n            return result\n        return wrapper\n    return decorator\n\n@repeat(3)\ndef greet(name):\n    print(f'Hello, {name}!')\n\ngreet('Alice')  # Prints 'Hello, Alice!' three times"
    },
    "property decorator": {
        "syntax": "@property\ndef method_name(self):\n    # getter method\n\n@method_name.setter\ndef method_name(self, value):\n    # setter method",
        "example": "class Circle:\n    def __init__(self, radius):\n        self._radius = radius\n\n    @property\n    def radius(self):\n        return self._radius\n\n    @radius.setter\n    def radius(self, value):\n        if value < 0:\n            raise ValueError('Radius cannot be negative')\n        self._radius = value\n\n    @property\n    def area(self):\n        return 3.14 * self._radius ** 2\n\ncircle = Circle(5)\nprint(circle.radius)  # 5\ncircle.radius = 10\nprint(circle.area)  # 314.0"
    },
    "dataclasses": {
        "syntax": "from dataclasses import dataclass\n\n@dataclass\nclass ClassName:\n    attribute1: type\n    attribute2: type = default_value",
        "example": "from dataclasses import dataclass\n\n@dataclass\nclass Point:\n    x: float\n    y: float\n    z: float = 0.0\n\np1 = Point(1.0, 2.0)\np2 = Point(3.0, 4.0, 5.0)\nprint(p1)  # Point(x=1.0, y=2.0, z=0.0)\nprint(p2)  # Point(x=3.0, y=4.0, z=5.0)"
    },
    "type hinting": {
        "syntax": "def function_name(param1: type1, param2: type2) -> return_type:\n    # function body",
        "example": "from typing import List, Dict, Optional\n\ndef process_data(items: List[int], options: Optional[Dict[str, str]] = None) -> List[str]:\n    result = []\n    for item in items:\n        if options and 'prefix' in options:\n            result.append(f'{options['prefix']}{item}')\n        else:\n            result.append(str(item))\n    return result\n\ndata = [1, 2, 3]\nopts = {'prefix': 'Item: '}\nprint(process_data(data, opts))  # ['Item: 1', 'Item: 2', 'Item: 3']"
    },
    "abstract base classes": {
        "syntax": "from abc import ABC, abstractmethod\n\nclass AbstractClass(ABC):\n    @abstractmethod\n    def abstract_method(self):\n        pass",
        "example": "from abc import ABC, abstractmethod\n\nclass Shape(ABC):\n    @abstractmethod\n    def area(self):\n        pass\n\n    @abstractmethod\n    def perimeter(self):\n        pass\n\nclass Rectangle(Shape):\n    def __init__(self, width, height):\n        self.width = width\n        self.height = height\n\n    def area(self):\n        return self.width * self.height\n\n    def perimeter(self):\n        return 2 * (self.width + self.height)\n\nrect = Rectangle(5, 3)\nprint(f'Area: {rect.area()}, Perimeter: {rect.perimeter()}')"
    },
    "context managers with contextlib": {
        "syntax": "from contextlib import contextmanager\n\n@contextmanager\ndef context_manager_name():\n    # setup\n    try:\n        yield\n    finally:\n        # cleanup",
        "example": "from contextlib import contextmanager\n\n@contextmanager\ndef temp_file(filename):\n    try:\n        f = open(filename, 'w')\n        yield f\n    finally:\n        f.close()\n        import os\n        os.remove(filename)\n\nwith temp_file('test.txt') as f:\n    f.write('Hello, World!')\n    # File is automatically closed and deleted after the with block"
    },
    "asyncio event loop": {
        "syntax": "import asyncio\n\nasync def main():\n    # async code\n\nasyncio.run(main())",
        "example": "import asyncio\n\nasync def say_after(delay, what):\n    await asyncio.sleep(delay)\n    print(what)\n\nasync def main():\n    print('started at', asyncio.get_event_loop().time())\n    await say_after(1, 'hello')\n    await say_after(2, 'world')\n    print('finished at', asyncio.get_event_loop().time())\n\nasyncio.run(main())"
    },
    "metaclasses": {
        "syntax": "class MetaClassName(type):\n    def __new__(cls, name, bases, attrs):\n        # customize class creation\n        return super().__new__(cls, name, bases, attrs)\n\nclass ClassName(metaclass=MetaClassName):\n    pass",
        "example": "class LoggingMeta(type):\n    def __new__(cls, name, bases, attrs):\n        for attr_name, attr_value in attrs.items():\n            if callable(attr_value):\n                attrs[attr_name] = cls.log_call(attr_value)\n        return super().__new__(cls, name, bases, attrs)\n\n    @staticmethod\n    def log_call(func):\n        def wrapper(*args, **kwargs):\n            print(f'Calling {func.__name__}')\n            return func(*args, **kwargs)\n        return wrapper\n\nclass MyClass(metaclass=LoggingMeta):\n    def method1(self):\n        print('Method 1')\n\n    def method2(self):\n        print('Method 2')\n\nobj = MyClass()\nobj.method1()  # Prints: Calling method1\\nMethod 1\nobj.method2()  # Prints: Calling method2\\nMethod 2"
    },
    "descriptors": {
        "syntax": "class DescriptorName:\n    def __get__(self, obj, type=None) -> object:\n        pass\n    def __set__(self, obj, value) -> None:\n        pass\n    def __delete__(self, obj) -> None:\n        pass",
        "example": "class Verbose_attribute():\n    def __get__(self, obj, type=None):\n        print('Accessing the attribute...')\n        return obj._x\n\n    def __set__(self, obj, value):\n        print('Setting the attribute...')\n        obj._x = value\n\nclass MyClass():\n    x = Verbose_attribute()\n    def __init__(self):\n        self._x = 0\n\nobj = MyClass()\nobj.x = 10  # Prints: Setting the attribute...\nprint(obj.x)  # Prints: Accessing the attribute...\\n10"
    },
    "functools": {
        "syntax": "from functools import lru_cache, partial, wraps",
        "example": "from functools import lru_cache\n\n@lru_cache(maxsize=None)\ndef fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nprint([fibonacci(n) for n in range(10)])  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]"
    },
    "itertools": {
        "syntax": "from itertools import count, cycle, repeat, chain, combinations, permutations",
        "example": "from itertools import cycle, islice\n\ncyclical = cycle('ABCD')\nprint(list(islice(cyclical, 10)))  # ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D', 'A', 'B']"
    },
    "numpy basics": {
        "syntax": "import numpy as np",
        "example": "import numpy as np\n\narr = np.array([1, 2, 3, 4, 5])\nprint(arr * 2)  # [2 4 6 8 10]\nprint(np.mean(arr))  # 3.0\nprint(np.std(arr))  # 1.4142135623730951"
    },
    "pandas basics": {
        "syntax": "import pandas as pd",
        "example": "import pandas as pd\n\ndf = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})\nprint(df)\nprint(df['A'].mean())\nprint(df.describe())"
    },
    "matplotlib basics": {
        "syntax": "import matplotlib.pyplot as plt",
        "example": "import matplotlib.pyplot as plt\nimport numpy as np\n\nx = np.linspace(0, 10, 100)\ny = np.sin(x)\nplt.plot(x, y)\nplt.title('Sine Wave')\nplt.xlabel('x')\nplt.ylabel('sin(x)')\nplt.show()"
    },
    "requests": {
        "syntax": "import requests",
        "example": "import requests\n\nresponse = requests.get('https://api.github.com')\nprint(response.status_code)\nprint(response.json())"
    },
    "pytest": {
        "syntax": "import pytest",
        "example": "# test_example.py\ndef func(x):\n    return x + 1\n\ndef test_answer():\n    assert func(3) == 4\n\n# Run with: pytest test_example.py"
    },
    "sqlalchemy": {
        "syntax": "from sqlalchemy import create_engine, Column, Integer, String\nfrom sqlalchemy.ext.declarative import declarative_base\nfrom sqlalchemy.orm import sessionmaker",
        "example": "from sqlalchemy import create_engine, Column, Integer, String\nfrom sqlalchemy.ext.declarative import declarative_base\nfrom sqlalchemy.orm import sessionmaker\n\nBase = declarative_base()\n\nclass User(Base):\n    __tablename__ = 'users'\n    id = Column(Integer, primary_key=True)\n    name = Column(String)\n    age = Column(Integer)\n\nengine = create_engine('sqlite:///example.db')\nBase.metadata.create_all(engine)\n\nSession = sessionmaker(bind=engine)\nsession = Session()\n\nnew_user = User(name='Alice', age=30)\nsession.add(new_user)\nsession.commit()\n\nusers = session.query(User).all()\nfor user in users:\n    print(f'{user.name}, {user.age} years old')"
    },
    "asyncio with aiohttp": {
        "syntax": "import asyncio\nimport aiohttp",
        "example": "import asyncio\nimport aiohttp\n\nasync def fetch(session, url):\n    async with session.get(url) as response:\n        return await response.text()\n\nasync def main():\n    urls = ['http://python.org', 'http://example.com', 'http://github.com']\n    async with aiohttp.ClientSession() as session:\n        tasks = [fetch(session, url) for url in urls]\n        responses = await asyncio.gather(*tasks)\n        for url, response in zip(urls, responses):\n            print(f'{url}: {len(response)} bytes')\n\nasyncio.run(main())"
    },
    "fastapi basics": {
        "syntax": "from fastapi import FastAPI",
        "example": "from fastapi import FastAPI\nfrom pydantic import BaseModel\n\napp = FastAPI()\n\nclass Item(BaseModel):\n    name: str\n    price: float\n\n@app.post('/items/')\nasync def create_item(item: Item):\n    return {'item_name': item.name, 'item_price': item.price}\n\n@app.get('/items/{item_id}')\nasync def read_item(item_id: int):\n    return {'item_id': item_id}\n\n# Run with: uvicorn main:app --reload"
    },
    "django basics": {
        "syntax": "from django.db import models\nfrom django.shortcuts import render",
        "example": "# models.py\nfrom django.db import models\n\nclass Post(models.Model):\n    title = models.CharField(max_length=200)\n    content = models.TextField()\n    pub_date = models.DateTimeField('date published')\n\n# views.py\nfrom django.shortcuts import render\nfrom .models import Post\n\ndef post_list(request):\n    posts = Post.objects.order_by('-pub_date')[:5]\n    return render(request, 'blog/post_list.html', {'posts': posts})"
    },
    "flask basics": {
        "syntax": "from flask import Flask, request, jsonify",
        "example": "from flask import Flask, request, jsonify\n\napp = Flask(__name__)\n\n@app.route('/')\ndef hello_world():\n    return 'Hello, World!'\n\n@app.route('/api/data', methods=['POST'])\ndef receive_data():\n    data = request.json\n    return jsonify({'received': data}), 201\n\nif __name__ == '__main__':\n    app.run(debug=True)"
    },
    "pytorch basics": {
        "syntax": "import torch\nimport torch.nn as nn\nimport torch.optim as optim",
        "example": "import torch\nimport torch.nn as nn\nimport torch.optim as optim\n\nclass SimpleNet(nn.Module):\n    def __init__(self):\n        super(SimpleNet, self).__init__()\n        self.fc = nn.Linear(10, 5)\n\n    def forward(self, x):\n        return self.fc(x)\n\nmodel = SimpleNet()\ninput_data = torch.randn(3, 10)\noutput = model(input_data)\nprint(output)"
    },
    "tensorflow basics": {
        "syntax": "import tensorflow as tf",
        "example": "import tensorflow as tf\n\nmn = tf.keras.models.Sequential([\n    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),\n    tf.keras.layers.Dense(32, activation='relu'),\n    tf.keras.layers.Dense(1, activation='sigmoid')\n])\n\nmodel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])\n\n# Assuming you have x_train and y_train data\n# model.fit(x_train, y_train, epochs=10, batch_size=32)"
    },
    "pyspark basics": {
        "syntax": "from pyspark.sql import SparkSession",
        "example": "from pyspark.sql import SparkSession\nfrom pyspark.sql.functions import col\n\nspark = SparkSession.builder.appName('PySparkExample').getOrCreate()\n\ndf = spark.createDataFrame([(1, 'John'), (2, 'Jane'), (3, 'Doe')], ['id', 'name'])\nresult = df.filter(col('id') > 1).select('name')\nresult.show()"
    },
    "scikit-learn basics": {
        "syntax": "from sklearn import datasets, model_selection, svm",
        "example": "from sklearn import datasets, model_selection, svm\n\niris = datasets.load_iris()\nX_train, X_test, y_train, y_test = model_selection.train_test_split(iris.data, iris.target, test_size=0.3)\n\nclf = svm.SVC()\nclf.fit(X_train, y_train)\n\naccuracy = clf.score(X_test, y_test)\nprint(f'Accuracy: {accuracy}')"
    },
    "type checking with mypy": {
        "syntax": "# Run: mypy script.py",
        "example": "from typing import List, Dict\n\ndef process_items(items: List[int]) -> Dict[str, int]:\n    result: Dict[str, int] = {}\n    for item in items:\n        result[str(item)] = item * 2\n    return result\n\ndef main() -> None:\n    data = [1, 2, 3]\n    processed = process_items(data)\n    print(processed)\n\nif __name__ == '__main__':\n    main()\n\n# Run: mypy script.py"
    },
    "concurrent.futures": {
        "syntax": "from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor",
        "example": "import concurrent.futures\nimport time\n\ndef task(n):\n    time.sleep(n)\n    return f'Slept for {n} seconds'\n\nwith concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:\n    futures = [executor.submit(task, i) for i in range(3)]\n    for future in concurrent.futures.as_completed(futures):\n        print(future.result())"
    },
    "websockets": {
        "syntax": "import websockets",
        "example": "import asyncio\nimport websockets\n\nasync def echo(websocket, path):\n    async for message in websocket:\n        await websocket.send(f'Echo: {message}')\n\nasync def main():\n    server = await websockets.serve(echo, 'localhost', 8765)\n    await server.wait_closed()\n\nasyncio.run(main())"
    },
    "cryptography": {
        "syntax": "from cryptography.fernet import Fernet",
        "example": "from cryptography.fernet import Fernet\n\nkey = Fernet.generate_key()\nf = Fernet(key)\ntoken = f.encrypt(b'Secret message')\nprint(token)\ndecrypted = f.decrypt(token)\nprint(decrypted)  # b'Secret message'"
    },
    "graphene (GraphQL)": {
        "syntax": "import graphene",
        "example": "import graphene\n\nclass Query(graphene.ObjectType):\n    hello = graphene.String(name=graphene.String(default_value='World'))\n\n    def resolve_hello(self, info, name):\n        return f'Hello {name}'\n\nschema = graphene.Schema(query=Query)\nresult = schema.execute('{ hello }')\nprint(result.data['hello'])  # 'Hello World'"
    },
    "structlog": {
        "syntax": "import structlog",
        "example": "import structlog\n\nlogger = structlog.get_logger()\nlogger.info('Hello, World!', key1='value1', key2='value2')\n\n# Configure structlog to output JSON\nstructlog.configure(\n    processors=[structlog.processors.JSONRenderer()]\n)\nlogger.info('Structured logging', event='example', status='success')"
    },
    "hypothesis": {
        "syntax": "from hypothesis import given, strategies as st",
        "example": "from hypothesis import given, strategies as st\n\ndef encode_decode(x):\n    return x.encode('utf-8').decode('utf-8')\n\n@given(st.text())\ndef test_encode_decode(s):\n    assert encode_decode(s) == s\n\n# Run with pytest: pytest test_hypothesis.py"
    },
    "pattern matching (Python 3.10+)": {
        "syntax": "match subject:\n    case pattern1:\n        # code\n    case pattern2:\n        # code",
        "example": "def http_error(status):\n    match status:\n        case 400:\n            return 'Bad request'\n        case 404:\n            return 'Not found'\n        case 418:\n            return 'I'm a teapot'\n        case _:\n            return 'Something's wrong with the internet'\n\nprint(http_error(404))  # 'Not found'"
    },
    "typing with Protocols": {
        "syntax": "from typing import Protocol",
        "example": "from typing import Protocol, List\n\nclass Drawable(Protocol):\n    def draw(self) -> None:\n        ...\n\nclass Circle:\n    def draw(self) -> None:\n        print('Drawing a circle')\n\nclass Square:\n    def draw(self) -> None:\n        print('Drawing a square')\n\ndef draw_all(shapes: List[Drawable]) -> None:\n    for shape in shapes:\n        shape.draw()\n\ndraw_all([Circle(), Square()])  # Valid, both Circle and Square implement Drawable"
    },
    "functools.cache (Python 3.9+)": {
        "syntax": "from functools import cache",
        "example": "from functools import cache\n\n@cache\ndef fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nprint([fibonacci(n) for n in range(10)])  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]"
    }
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
