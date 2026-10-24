"""Minimal text-encoder API used only to launch Kimodo's local NPZ viewer.

It returns a correctly-shaped zero embedding.  Do not use this endpoint for
motion generation; the generated NPZ files were produced by the real encoder.
"""

from pathlib import Path
from uuid import uuid4

import gradio as gr
import numpy as np


OUTPUT_DIR = Path("/tmp/kimodo-viewer-embeddings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def encode_for_viewer(text: str, filename: str):
    del text
    safe_name = Path(filename).name if filename else f"{uuid4()}.npy"
    path = OUTPUT_DIR / safe_name
    np.save(path, np.zeros((1, 4096), dtype=np.float32))
    return gr.DownloadButton(value=str(path), visible=True), "", ""


with gr.Blocks(title="Kimodo viewer text-encoder stub") as demo:
    text = gr.Textbox(label="Text prompt")
    filename = gr.Textbox(value="embedding.npy", visible=False)
    download = gr.DownloadButton(visible=False)
    title = gr.Markdown(visible=False)
    description = gr.Markdown(visible=False)
    button = gr.Button("Encode")
    button.click(
        fn=encode_for_viewer,
        inputs=[text, filename],
        outputs=[download, title, description],
        api_name="DemoWrapper",
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=9550)
