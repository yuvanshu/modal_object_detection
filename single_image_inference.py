import os
import torch
import modal
import json
import time
import argparse
import requests
import matplotlib.pyplot as plt

from transformers import DetaForObjectDetection, DetaImageProcessor
from PIL import Image

gpu_image = (
    modal.Image.debian_slim()
    .apt_install("git", "curl")  # Install any additional system dependencies
    .pip_install(
        'transformers', 
        'pillow', 
        'torch',
        'torchvision',
        'requests',
        'matplotlib'
    )
    .run_commands(
        "pip install torch --extra-index-url https://download.pytorch.org/whl/cu117"  # Ensure GPU support with CUDA 11.7
    )
)

# Create a shared volume for persisting inference results
results_volume = modal.Volume.from_name("single_image_object_detection_results", create_if_missing=True)

app = modal.App("flowstate_single_image_object_detection", image=gpu_image)

@app.function(
    image=gpu_image,
    timeout=10800,
    gpu='T4',
    volumes={"/mnt/data": results_volume})
def run_single_image_detection(model, processor, image_url, threshold, device):

    start_time = time.time()

    image = Image.open(requests.get(image_url, stream=True).raw)
    encoding = processor(image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**encoding.to(device))

    target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
    results = processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]

    end_time = time.time()

    return image, results, (end_time - start_time)

@app.function(
    image=gpu_image,
    timeout=10800,
    gpu='T4',
    volumes={"/mnt/data": results_volume})
def save_object_detection_json(results, inference_time):

    json_results = {}
    json_results['scores'] = results['scores'].tolist()
    json_results['labels'] = results['labels'].tolist()
    json_results['bounding_boxes'] = results['boxes'].tolist()
    json_results['inference_time'] = inference_time

    # Save the formatted results to a JSON file
    with open("/mnt/data/single_image_object_detection.json", 'w') as f:
        json.dump(json_results, f)
    print(f"single_image_object_detection.json has been saved", flush=True)

@app.function(
    image=gpu_image,
    timeout=10800,
    gpu='T4',
    volumes={"/mnt/data": results_volume})
def save_object_detection_img(model, image, results):

    # colors for visualization
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
            [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
    
    scores = results['scores']
    labels = results['labels']
    boxes = results['boxes']

    plt.figure(figsize=(16,10))
    plt.imshow(image)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax), c in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{model.config.id2label[label]}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')

    # Save the plot as a PNG file
    plt.savefig("/mnt/data/single_image_object_detection.png", bbox_inches='tight', pad_inches=0.1)
    print(f"single_image_object_detection.png has been saved", flush=True)

@app.function(
    image=gpu_image,
    timeout=10800,
    gpu='T4',
    volumes={"/mnt/data": results_volume})
def main(get_object_detection_json, get_object_detection_img, threshold, image_url):
    print('Starting Single Image Object Detection App', flush=True)
    # Check if GPU is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # Initialize the processor and model
    model = DetaForObjectDetection.from_pretrained("jozhang97/deta-swin-large").to(device)
    processor = DetaImageProcessor.from_pretrained("jozhang97/deta-resnet-50")

    image, results, inference_time = run_single_image_detection.remote(model, processor, image_url, threshold, device)

    if get_object_detection_json: save_object_detection_json.remote(results, inference_time)

    if get_object_detection_img: save_object_detection_img.remote(model, image, results)

    print('Finished single object detection!', flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--get_object_detection_json", type=bool, default=True)
    parser.add_argument("--get_object_detection_img", type=bool, default=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--image_url", type=str)
    
    args = parser.parse_args()

    with modal.enable_output():
        with app.run():
            main.remote(get_object_detection_json=args.get_object_detection_json, get_object_detection_img=args.get_object_detection_img, threshold=args.threshold, image_url=args.image_url)