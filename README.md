Real-time talk alignment tool using VAD and Whisper STT (Faster Whisper). Matches live speech with a prepared script and displays the current line. By default, Korean speech is recognized (Other languages could be supported upon requests).

## Prerequisites

1. Python (3.12.4 recommended)
2. Terminal (Git Bash recommended for Windows)
3. Git
4. Microphone devices

## Installation for MacOS

1. git clone this package.
```
$ git clone https://github.com/Jin-Myung/talk-align.git
```

2. Change current directory to the cloned package.
```
$ cd talk-align
```

3. Create virtual python environment:
```
$ python -m venv venv
$ . venv/bin/activate
```

4. Install required packages:
```
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## Test for MacOS

1. Change current directory to the cloned package.
```
$ cd talk-align
```

2. Activate the virtual environment:
```
$ . venv/bin/activate
```

3. Start webserver
```
$ uvicorn server:app --host 0.0.0.0 --port 8000
```

4. Open another terminal and run Step 1 and 2.
5. Run the main program:
```
$ python main.py
```

6. Open the operator webpage (http://localhost:8000/public/operator.html).
7. Open the audience webpage (http://localhost:8000/public/audience.html).
8. Load script (KO) and prompt (EN) files from the operator webpage.
9. Speak the lines in the script, and the corresponding part will be highlighted in the audience webpage.
10. Ctrl+C in the terminal to end the main program (main.py).
11. You may leave the webserver running and webpages open for future runs. Close them if needed.

## Installation for Windows

1. git clone this package.
```
$ git clone https://github.com/Jin-Myung/talk-align.git
```

2. Change current directory to the cloned package.
```
$ cd talk-align
```

3. Create virtual python environment:
```
$ python -m venv venv
$ . venv/Scripts/activate
```

4. Install required packages:
```
$ python -m pip install --upgrade pip
$ pip install -r requirements.txt
```

## Test for Windows

1. Change current directory to the cloned package.
```
$ cd talk-align
```

2. Activate the virtual environment:
```
$ . venv/Scripts/activate
```

3. Start webserver
```
$ uvicorn server:app --host 0.0.0.0 --port 8000
```

4. Open another terminal and run Step 1 and 2.
5. Run the main program:
```
$ python main.py
```

6. Open the operator webpage (http://localhost:8000/public/operator.html).
7. Open the audience webpage (http://localhost:8000/public/audience.html).
8. Load script (KO) and prompt (EN) files from the operator webpage.
9. Speak the lines in the script, and the corresponding part will be highlighted in the audience webpage.
10. Ctrl+C in the terminal to end the main program (main.py).
11. You may leave the webserver running and webpages open for future runs. Close them if needed.
