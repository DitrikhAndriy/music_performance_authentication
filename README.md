# Music Performance Authentication

A web application for authenticating authorship of musical performances using audio signal processing and machine learning methods.

## Requirements

* Python 3.10+
* FFmpeg

### FFmpeg

Download FFmpeg (build 2026-06-29) from:

https://www.gyan.dev/ffmpeg/builds/

Extract it into:

```text
tools/ffmpeg/
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running

Run the web application:

```bash
python web.py
```

Then open:

```text
http://127.0.0.1:5000
```
