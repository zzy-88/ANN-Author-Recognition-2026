
import os
import re
import html
import requests
import numpy as np
import gradio as gr

# تحميل أوزان النموذج والخصائص المحفوظة
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
assets = np.load(os.path.join(BASE_DIR, "inference_assets.npz"), allow_pickle=True)

W1, b1 = assets["W1"], assets["b1"]
W2, b2 = assets["W2"], assets["b2"]
W3, b3 = assets["W3"], assets["b3"]

mean = assets["mean"]
scale = assets["scale"]
feature_columns = assets["feature_columns"].tolist()
classes = assets["classes"].tolist()

# الكتب الأربعة
BOOKS = {
    "Emma - Jane Austen": {
        "author": "Austen",
        "url": "https://www.gutenberg.org/cache/epub/158/pg158.txt"
    },
    "The Call of the Wild - Jack London": {
        "author": "London",
        "url": "https://www.gutenberg.org/cache/epub/215/pg215.txt"
    },
    "Paradise Lost - John Milton": {
        "author": "Milton",
        "url": "https://www.gutenberg.org/cache/epub/20/pg20.txt"
    },
    "Complete Works - William Shakespeare": {
        "author": "Shakespeare",
        "url": "https://www.gutenberg.org/cache/epub/100/pg100.txt"
    }
}

# استخراج الخصائص اللغوية
def extract_features(text):
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

    features = [
        words.count(feature_word.lower())
        for feature_word in feature_columns
    ]

    return np.array(features, dtype=np.float64).reshape(1, -1)


# التنبؤ باستخدام أوزان ANN الأصلية
def predict_author_from_text(text):
    words = re.findall(r"\b[a-zA-Z]+\b", text)
    word_count = len(words)

    features = extract_features(text)
    x = (features - mean) / scale

    z1 = np.maximum(0, x @ W1 + b1)
    z2 = np.maximum(0, z1 @ W2 + b2)
    z3 = z2 @ W3 + b3

    exp_values = np.exp(
        z3 - np.max(z3, axis=1, keepdims=True)
    )
    probabilities = (
        exp_values / exp_values.sum(axis=1, keepdims=True)
    )[0]

    predicted_class = np.argmax(probabilities)
    predicted_author = classes[predicted_class]

    return predicted_author, probabilities, word_count


# تحميل عينة كتاب
def load_sample(book_name):
    if not book_name:
        return "", "Please choose a book first."

    try:
        book_info = BOOKS[book_name]

        response = requests.get(
            book_info["url"],
            timeout=60,
            headers={"User-Agent": "AuthorRecognitionANN/1.0"}
        )
        response.raise_for_status()

        words = re.findall(r"\b[a-zA-Z]+\b", response.text)

        if len(words) < 2700:
            return "", "The selected book text is too short."

        sample_text = " ".join(words[1000:2700])

        info = (
            f"Book: {book_name}\n"
            f"Real Author: {book_info['author']}\n"
            f"Sample Length: 1700 words"
        )

        return sample_text, info

    except Exception as error:
        return "", f"Unable to load the book: {str(error)}"


# تحليل النص وعرض النتائج
def analyze_text_for_interface(text):
    if not text or not text.strip():
        return "", "", "", "Please enter some text first."

    try:
        predicted_author, probabilities, word_count = (
            predict_author_from_text(text)
        )

        probability_html = """
        <div style="
            background:#5C5148;
            padding:20px;
            border-radius:12px;
            color:#FFF4E6;
            font-family:Arial,sans-serif;
        ">
            <h3 style="color:#FFF4E6; margin-top:0;">
                Author Probabilities
            </h3>
        """

        for author, probability in zip(classes, probabilities):
            percentage = float(probability) * 100
            safe_author = html.escape(str(author))

            probability_html += f"""
            <div style="margin-bottom:16px;">
                <div style="
                    display:flex;
                    justify-content:space-between;
                    align-items:center;
                    margin-bottom:7px;
                    color:#FFF4E6;
                    font-size:15px;
                    font-weight:600;
                ">
                    <span>{safe_author}</span>
                    <span style="color:#F3D8B8;">
                        {percentage:.2f}%
                    </span>
                </div>

                <div style="
                    background:#87786D;
                    height:13px;
                    border-radius:10px;
                    overflow:hidden;
                ">
                    <div style="
                        background:#D8B58C;
                        width:{percentage:.2f}%;
                        height:13px;
                        border-radius:10px;
                    "></div>
                </div>
            </div>
            """

        probability_html += "</div>"

        return (
            predicted_author,
            str(word_count),
            probability_html,
            "Analysis completed successfully."
        )

    except Exception as error:
        return "", "", "", f"Analysis failed: {str(error)}"


# تنسيق الواجهة
custom_css = """
.gradio-container {
    background: #F5EBDD !important;
    color: #493326 !important;
}

.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container p,
.gradio-container label {
    color: #493326 !important;
}

.gradio-container input,
.gradio-container textarea {
    background: #62564D !important;
    color: #FFF4E6 !important;
    border: 1px solid #A98A6D !important;
}

.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
    color: #D8C8B8 !important;
    opacity: 1 !important;
}

.gradio-container [data-testid="dropdown"],
.gradio-container [role="combobox"] {
    background: #62564D !important;
    color: #FFF4E6 !important;
    border: 1px solid #A98A6D !important;
}

.gradio-container [data-testid="dropdown"] svg,
.gradio-container [role="combobox"] svg {
    color: #F5EBDD !important;
    fill: #F5EBDD !important;
    stroke: #F5EBDD !important;
}

.gradio-container [role="listbox"],
.gradio-container [role="option"] {
    background: #62564D !important;
    color: #FFF4E6 !important;
}

.gradio-container [role="option"]:hover {
    background: #806D5D !important;
    color: #FFFFFF !important;
}

.gradio-container button {
    background: #D8C1A5 !important;
    border: 1px solid #A98A6D !important;
    color: #493326 !important;
}

.gradio-container .prose {
    color: #493326 !important;
}
"""


# بناء الواجهة
with gr.Blocks(title="Authorship Recognition") as demo:

    gr.Markdown("# Authorship Recognition")

    gr.Markdown(
        "Identify the writing style of one of the four authors."
    )

    gr.Markdown("## 1. Choose a Ready-Made Sample")

    sample_dropdown = gr.Dropdown(
        choices=list(BOOKS.keys()),
        label="Select an Author & Book",
        interactive=True
    )

    load_button = gr.Button("Load Sample")

    sample_info = gr.Textbox(
        label="Sample Information"
    )

    gr.Markdown("---")

    gr.Markdown("## 2. Write or Paste Your Own Text")

    text_input = gr.Textbox(
        label="Your Text",
        placeholder="Paste or type your text here...",
        lines=12
    )

    analyze_button = gr.Button("Analyze Writing Style")

    gr.Markdown("## Analysis Results")

    author_output = gr.Textbox(
        label="Predicted Author"
    )

    word_output = gr.Textbox(
        label="Number of Words"
    )

    probability_output = gr.HTML()

    status_output = gr.Textbox(
        label="Analysis Status"
    )

    load_button.click(
        fn=load_sample,
        inputs=sample_dropdown,
        outputs=[text_input, sample_info]
    )

    analyze_button.click(
        fn=analyze_text_for_interface,
        inputs=text_input,
        outputs=[
            author_output,
            word_output,
            probability_output,
            status_output
        ]
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        theme=gr.themes.Soft(
            primary_hue="amber",
            secondary_hue="orange",
            neutral_hue="stone"
        ),
        css=custom_css
    )
