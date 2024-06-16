---
title: Lyricbox
emoji: 💻
colorFrom: pink
colorTo: purple
sdk: streamlit
sdk_version: 1.31.0
app_file: app.py
pinned: false
license: apache-2.0
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference\

# LyricsBox

LyricsBox is a Streamlit application that allows users to extract the lyrics from a YouTube song URL. The app downloads the audio from the provided URL, transcribes it using OpenAI's Whisper model, and displays the lyrics to the user.

## Demo
A demo of the app can be found here: [Lyricbox Demo](https://shubhendu-ghosh-lyricbox.hf.space)

## Features

- Extracts lyrics from any YouTube video link.
- Utilizes OpenAI's Whisper model for accurate transcription.
- User-friendly and easy to use.
- Ensures privacy by not storing any user data or video.

## Installation

To run this application locally, follow these steps:

1. **Clone the repository:**

   ```sh
   git clone https://github.com/shubhendu-ghosh-DS/Lyricbox.git
   cd LyricsBox
   ```
2. **Create and activate a virtual environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. **nstall the dependencies:**
   ```sh
   pip install -r requirements.txt
   ````

## Usage

1.  **Run the Streamlit app:**

    ```sh
    streamlit run app.py
    ````

2. **Open your browser and navigate to:**
   ```arduino
   http://localhost:8501
   ```

3. **Enter the YouTube video URL:**
   Paste the URL of the YouTube song you want the lyrics for and click "Get Lyrics".


## Dependencies

The project requires the following Python libraries:

- streamlit
- requests
- pytube
- torch
- transformers
- accelerate

These can be found in the `requirements.txt` file.

## File Structure

- `app.py`: The main Streamlit application file.
- `requirements.txt`: A list of Python dependencies required for the application.
- `.gitattributes`: Configuration for Git Large File Storage (LFS).
- `.streamlit/config.toml`: Configuration file for Streamlit theming.

## How It Works

1. **Input YouTube URL**: The user inputs a YouTube URL.
2. **Download Audio**: The app uses `pytube` to download the audio from the provided URL.
3. **Transcribe Audio**: The audio file is transcribed using OpenAI's Whisper model.
4. **Display Lyrics**: The transcribed lyrics are displayed to the user.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.
