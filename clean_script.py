import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer
from gtts import gTTS
from googletrans import Translator
import os
import subprocess
def generate_abstractive_summary_bart(input_text, max_length=400):
    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)

    inputs = tokenizer.encode("Summarize: " + input_text, return_tensors="pt", max_length=1024, truncation=True)
    
    # Generate summary
    summary_ids = model.generate(inputs, max_length=max_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

def process_webpage_text(input_text):
    # Clean up extra spaces between paragraphs
    cleaned_webpage_text = ' '.join(input_text.split())
    generated_summary = generate_abstractive_summary_bart(cleaned_webpage_text)
    return generated_summary

def translate_text(text, target_language):
    translator = Translator()
    translated_text = translator.translate(text, dest=target_language).text
    return translated_text

def tts(text, lang='en', output_file='./assets/output.mp3'):
    tts_object = gTTS(text=text, lang=lang, slow=False)
    tts_object.save(output_file)

def save_summary_to_file(summary, output_file='./assets/clean.txt'):
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(summary)

def main():
    st.title("Webpage Text Summarizer")
    st.markdown(
        f"""
            <style>
                [data-testid="stSidebar"] {{
                    background-image: url(data:image/png;base64,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);
                    background-repeat: no-repeat;
                    padding-top: 190px;
                    background-position: 20px 20px;
                    width: 10px;
                }}
            </style>
            """,
        unsafe_allow_html=True,
    )
    # Sidebar navigation
    st.sidebar.header("Press Release Video Generation App!")

    # Read the input from the specified file
    input_file_path = "./assets/webpage_text.txt"
    if not os.path.exists(input_file_path):
        st.error("Input file not found. Please make sure the file exists.")
        return

    with open(input_file_path, 'r', encoding='utf-8') as file:
        webpage_text = file.read()

    st.subheader("Webpage Text:")
    st.write(webpage_text)

    # Sidebar Section
    st.sidebar.header("Text-to-Speech")

    language_names = {
        "en": "English",
        "hi": "Hindi",
        "bn": "Bengali",
        "te": "Telugu",
        "mr": "Marathi",
        "ta": "Tamil",
        "gu": "Gujarati",
        "kn": "Kannada",
        "ml": "Malayalam"
    }

    selected_language = st.sidebar.selectbox("Select Language", list(language_names.values()))

    language_code = {v: k for k, v in language_names.items()}  # Reverse mapping for language codes

    if st.sidebar.button("Text-to-Speech"):
        generated_summary = process_webpage_text(webpage_text)
        
        # Translate the summary to the selected language
        translated_summary = translate_text(generated_summary, language_code[selected_language])

        # Perform Text-to-Speech on the translated summary
        tts(translated_summary, lang=language_code[selected_language])
        st.sidebar.success("Text-to-Speech conversion completed!")

        # Save the translated summary to a file
        save_summary_to_file(translated_summary, output_file='./assets/v.txt')
        st.sidebar.success("Translated summary saved to ./assets/v.txt")
    st.sidebar.header("Video Geneartion")
    if st.sidebar.button("Run Video Generationt"):
            st.info("Running generation... Please wait.")
            
            # Use subprocess to run the clean_script.py
            subprocess.run(["streamlit", "run", "video.py"])

            st.success("video Script running!")
        
    # Summarization Section
    if st.button("Generate Summary"):
        generated_summary = process_webpage_text(webpage_text)
        st.subheader("Generated Summary:")
        st.write(generated_summary)

        # Save the generated summary to a file
        save_summary_to_file(generated_summary)
        st.success("Summary saved to ./assets/clean.txt")

if __name__ == "__main__":
    main()
