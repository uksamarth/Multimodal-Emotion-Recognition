"""
File: app.py
Author: U K Samarth
Description: Description: Main application file for Facial_Expression_Recognition.
             The file defines the Gradio interface, sets up the main blocks,
             and includes event handlers for various components.
License: MIT License
"""

import gradio as gr

# Importing necessary components for the Gradio app
from app.description import DESCRIPTION_STATIC, DESCRIPTION_DYNAMIC
from app.authors import AUTHORS
from app.app_utils import preprocess_image_and_predict, preprocess_video_and_predict


def clear_static_info():
    return (
        gr.Image(value=None, type="pil"),
        gr.Image(value=None, scale=1, elem_classes="dl5"),
        gr.Image(value=None, scale=1, elem_classes="dl2"),
        gr.Label(value=None, num_top_classes=3, scale=1, elem_classes="dl3"),
    )

def clear_dynamic_info():
    return (
        gr.Video(value=None),
        gr.Video(value=None),
        gr.Video(value=None),
        gr.Video(value=None),
        gr.Plot(value=None),
    )

with gr.Blocks(css="app.css") as demo:
    with gr.Tab("Dynamic App"):
        gr.Markdown(value=DESCRIPTION_DYNAMIC)
        with gr.Row():
            with gr.Column(scale=2):
                input_video = gr.Video(elem_classes="video1")
                with gr.Row():
                    clear_btn_dynamic = gr.Button(
                        value="Clear", interactive=True, scale=1
                    )
                    submit_dynamic = gr.Button(
                        value="Submit", interactive=True, scale=1, elem_classes="submit"
                    )
            with gr.Column(scale=2, elem_classes="dl4"):
                with gr.Row():
                    output_video = gr.Video(label="Original video", scale=1, elem_classes="video2")
                    output_face = gr.Video(label="Pre-processed video", scale=1, elem_classes="video3")
                    output_heatmaps = gr.Video(label="Heatmaps", scale=1, elem_classes="video4")
                output_statistics = gr.Plot(label="Statistics of emotions", elem_classes="stat")
        gr.Examples(
            ["videos/video1.mp4",
            "videos/video2.mp4",
            ],
            [input_video],
        )

    with gr.Tab("Static App"):
        gr.Markdown(value=DESCRIPTION_STATIC)
        with gr.Row():
            with gr.Column(scale=2, elem_classes="dl1"):
                input_image = gr.Image(label="Original image", type="pil")
                with gr.Row():
                    clear_btn = gr.Button(
                        value="Clear", interactive=True, scale=1, elem_classes="clear"
                    )
                    submit = gr.Button(
                        value="Submit", interactive=True, scale=1, elem_classes="submit"
                    )
            with gr.Column(scale=1, elem_classes="dl4"):
                with gr.Row():
                    output_image = gr.Image(label="Face", scale=1, elem_classes="dl5")
                    output_heatmap = gr.Image(label="Heatmap", scale=1, elem_classes="dl2")
                output_label = gr.Label(num_top_classes=3, scale=1, elem_classes="dl3")
        gr.Examples(
            [
                "images/fig7.jpg",
                "images/fig1.jpg",
                "images/fig2.jpg",
                "images/fig3.jpg",
                "images/fig4.jpg",
                "images/fig5.jpg",
                "images/fig6.jpg",
            ],
            [input_image],
        )
    # with gr.Tab("Authors"):
    #     gr.Markdown(value=AUTHORS)

    submit.click(
        fn=preprocess_image_and_predict,
        inputs=[input_image],
        outputs=[output_image, output_heatmap, output_label],
        queue=True,
    )
    clear_btn.click(
        fn=clear_static_info,
        inputs=[],
        outputs=[input_image, output_image, output_heatmap, output_label],
        queue=True,
    )

    submit_dynamic.click(
        fn=preprocess_video_and_predict,
        inputs=input_video,
        outputs=[
            output_video,
            output_face,
            output_heatmaps, 
            output_statistics
        ],
        queue=True,
    )
    clear_btn_dynamic.click(
        fn=clear_dynamic_info,
        inputs=[],
        outputs=[
            input_video,
            output_video,
            output_face,
            output_heatmaps, 
            output_statistics
        ],
        queue=True,
    )

if __name__ == "__main__":
    demo.queue(api_open=False).launch(share=True)
