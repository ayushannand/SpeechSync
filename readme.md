# SpeechSync

SpeechSync is a Python application that converts speech into text and transcribes it into different languages. Follo the steps belo to set up and run the app.

## Prerequisites

- Python 3.11
- ffmpeg (command-line tool)

### Install ffmpeg

#### Ubuntu or Debian

```bash
sudo apt update && sudo apt install ffmpeg
```

#### Arch Linux

```bash
sudo pacman -S ffmpeg
```

#### MacOS (using Homebrew)

```bash
brew install ffmpeg
```

#### Windows (using Chocolatey)

```bash
choco install ffmpeg
```

#### Windows (using Scoop)

```bash
scoop install ffmpeg
```

### Rust (if needed)

If tiktoken does not provide a pre-built wheel for your platform, you may need to install Rust. Follo the Getting Started page to install the Rust development environment. Additionally, configure the PATH environment variable:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

If you encounter the error "No module named 'setuptools_rust'", install setuptools_rust:

```bash
pip install setuptools-rust
```

## Installation

### Step 1 - Clone the repository

```bash
git clone git@github.com:ayushannand/SpeechSync.git
```

### Step 2 - Create a virtual environment

```bash
python3 -m venv env
```

### Step 3 - Activate the virtual environment

```bash
source env/bin/activate
```

### Step 4 - Install whisper and translator in the virtual environment

```bash
pip install --upgrade pip
pip install --upgrade git+https://github.com/huggingface/transformers.git accelerate
pip install torch
pip install sacremoses
```

## Running the App

### Step 5 - Run the app

```bash
python app.py
```

Now, the SpeechSync app is ready to transcribe speech into text in different languages. Enjoy using the application!