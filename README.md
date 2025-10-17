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
5. Run the program with a script file and a prompt file:
```
$ python main.py input_kor.txt input_eng.txt
```

6. Speak the lines from the script, and the corresponding part will be displayed.
7. Ctrl+C to end the program safely.

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
5. Run the program with a script file and a prompt file:
```
$ python main.py input_kor.txt input_eng.txt
```

6. Speak the lines from the script, and the corresponding part will be displayed.
7. Ctrl+C to end the program safely.
