import streamlit as st
from pytube import YouTube
import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def get_mp3(url):
    yt = YouTube(str(url))
    audio = yt.streams.filter(only_audio = True).first()
    destination = '.'
    out_file = audio.download(output_path=destination)
    base, ext = os.path.splitext(out_file) 
    new_file = base + '.mp3'
    os.rename(out_file, new_file)
    return new_file
def get_transcript(audio_file):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-tiny"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,)
    
    result = pipe(audio_file, return_timestamps=True)
    result = result["chunks"]
    return result

def format_lyrics(lyrics):
    formatted_lyrics = ""
    for line in lyrics:
        text = line["text"]
        formatted_lyrics += f"{text}\n\n"
    return formatted_lyrics.strip()

def fetch_lyrics(url):
    mp3 = get_mp3(url)
    status_placeholder = st.empty()
    status_placeholder.subheader("Please wait for few seconds. We are preparing the lyrics for you")
    lyrics = get_transcript(mp3)
    status_placeholder.empty()
    lyrics = format_lyrics(lyrics)
    return lyrics


def main():
    text_color = "#d15000"
    #st.markdown('<p style="color: #47f3ff; font-family: sans-serif; text-align:center; font-size: 70px;"><b>LYRICBOX</b></p>', unsafe_allow_html=True)
    st.markdown("""<p style="color: #2BAE66FF;font-size: 70px;font-family: sans-serif; text-align:center;margin-bottom:0px;"><b>Lyrics</b><span style="color: #E94B3CFF;font-size: 70px;font-family: sans-serif;"><b>Box</b></span></p>""", unsafe_allow_html=True)
    st.markdown('<p style="font-family: sans-serif; text-align:center; font-size: 20px; margin-bottom:60px;">Get the Lyrics of your Favorite Song for Free</p>', unsafe_allow_html=True)
    # Input field for the user to enter the URL of the song
    st.markdown('<p style="font-family: sans-serif; text-align:left; font-size: 20px; margin-bottom:0px;">Enter the video link below</p>', unsafe_allow_html=True)
    url = st.text_input("", "")

    if url:
        # Button to trigger fetching and displaying the lyrics
        if st.button("Get Lyrics"):
            lyrics = fetch_lyrics(url)
            st.subheader("Lyrics:")
            st.write(lyrics)

    st.markdown('<p style="font-size: 35px;font-family: sans-serif; text-align:left; margin-top: 100px;"><b>How to Get the Lyrics of your Video?</b></p>', unsafe_allow_html=True)
    st.markdown('<p style="font-family: sans-serif; text-align:left; font-size: 20px">To extract the lyrics of your favorite video using this tool follow the steps. <br /> <br /> &ensp; 1. Copy the link of the video from Youtube \
                <br /> &ensp; 2. Paste the link in the box above <br /> &ensp; 3. Hit the "Get Lyrics" button. </p>', unsafe_allow_html=True)
    
    st.markdown('<p style="font-size: 35px;font-family: sans-serif; text-align:left; margin-top: 40px;"><b>Why Should you Use This Tool</b></p>', unsafe_allow_html=True)
    st.markdown('<p style="font-family: sans-serif; text-align:left; font-size: 20px">Features of this tool are given below <br /> <br /> \
                &ensp; 1. This Tool uses <a href="https://huggingface.co/openai/whisper-tiny/">OpenAI Whisper</a> to extract transcript from audio file\
                <br /> &ensp; 2. We do not save your data or video\
                <br /> &ensp; 3. Easy-to-use and Free-to-use</p>', unsafe_allow_html=True)
    
    st.markdown('<p style="font-size: 35px;font-family: sans-serif; text-align:left; margin-top: 40px;"><b>Developer</b></p>', unsafe_allow_html=True)
    st.markdown('<p style="font-family: sans-serif; text-align:left; font-size: 20px"> &ensp;<a href="https://www.linkedin.com/in/shubhendu-ghosh-423092205/">LinkedIn</a> \
                <br /> &ensp;<a href="https://twitter.com/shubhendubro">Twitter</a></p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()