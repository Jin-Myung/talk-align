Real-time talk alignment tool using VAD and Whisper STT (Faster Whisper). Matches live speech with a prepared script and displays the current line. By default, Korean speech is recognized (Other languages could be supported upon requests).

## Installation (for MacOS)

1. git clone this package.
2. Change current directory to the cloned package.
```
$ cd talk-align
```

3. Create virtual python environment:
```
$ python3 -m venv venv
$ . venv/bin/activate
```

4. Install required packages:
```
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## Test

1. Activate the virtual environment:
```
$ . venv/bin/activate
```
2. Run the program with a script file:
```
$ python main.py input.txt
```

3. Speak the lines from the script, and the corresponding part will be displayed.
4. Ctrl+C to end the program.
