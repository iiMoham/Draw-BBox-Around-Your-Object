from pathlib import Path

from transformers import (
	AutoImageProcessor,
	AutoModelForObjectDetection,
	DetrImageProcessor,
	pipeline,
)


import gradio as gr
from PIL import Image, ImageFont, ImageDraw



pipe = pipeline("object-detection", model="facebook/detr-resnet-50")

def draw_bboxes(image, detections):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for det in detections:
        box = det["box"]
        label = det["label"]
        score = det["score"]

        x1, y1, x2, y2 = box["xmin"], box["ymin"], box["xmax"], box["ymax"]

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        text = f"{label} {score:.2f}"
        left, top, right, bottom = draw.textbbox((x1, y1), text, font=font)

        draw.rectangle([left, top, right, bottom], fill="red")
        draw.text((x1, y1), text, fill="white", font=font)

    return img

def object_detector(image):
    detections = pipe(image)
    result = draw_bboxes(image, detections=detections)
    return result


demo = gr.Interface(
      fn=object_detector,
      inputs=[gr.Image(label="Select Image", type="pil")],
      outputs=[gr.Image(label="Processed Image", type="pil")],
      title= "Object Detector",

)
demo.launch()
      


