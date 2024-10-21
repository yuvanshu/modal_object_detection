import os
import torch
import modal
import json
import time
import argparse

from transformers import DetaForObjectDetection, DetaImageProcessor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image

gpu_image = (
    modal.Image.debian_slim()
    .apt_install("git", "curl")  # Install any additional system dependencies
    .pip_install(
        'transformers', 
        'pillow', 
        'torch',
        'torchvision', 
        'pycocotools'
    )
    .run_commands(
        "pip install torch --extra-index-url https://download.pytorch.org/whl/cu117"  # Ensure GPU support with CUDA 11.7
    )
)
curr_dir = os.getcwd()
coco_data_dir = curr_dir + '/coco_data'

# Mount local directory
data_mount = modal.Mount.from_local_dir(coco_data_dir, remote_path="/mnt/images_data")

# Create a shared volume for persisting inference results
results_volume = modal.Volume.from_name("object_detection_results", create_if_missing=True)

app = modal.App("flowstate_object_detection_inference", image=gpu_image)

@app.function(
    image=gpu_image,
    timeout=10800,
    gpu='T4',
    volumes={"/mnt/data": results_volume},
    mounts=[data_mount])
def run_inference(coco_gt, image_ids, img_dir, model, processor, device):

    formatted_results = []
    counter = 1

    start_time = time.time()
    for idx in range(len(image_ids)):
        img_id = image_ids[idx]
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        
        # Preprocess image using the processor
        inputs = processor(images=image, return_tensors="pt")
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs.to(device))

        target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0).to(device)
        results = processor.post_process_object_detection(outputs, threshold=0, target_sizes=target_sizes)[0]

        # plot_results(image, results['scores'], results['labels'], results['boxes'])

        for i in range(len(results['scores'])):
            box = results['boxes'][i]
            formatted_results.append({
                'image_id': img_id,
                'category_id': results['labels'][i].item(),  # COCO category ID
                'bbox': [box[0].item(), box[1].item(), box[2].item() - box[0].item(), box[3].item() - box[1].item()],  # [x, y, width, height]
                'score': float(results['scores'][i])  # Convert score to float if it's a tensor
            })

        print('processed imgs: ', counter, flush=True)
        counter += 1

    end_time = time.time()
    return formatted_results, (end_time - start_time)

@app.function(
    image=gpu_image,
    timeout=10800,
    gpu='T4',
    volumes={"/mnt/data": results_volume})
def generate_inference_json(results):
    # Save the formatted results to a JSON file
    with open("/mnt/data/validation_results.json", 'w') as f:
        json.dump(results, f)

@app.function(
    image=gpu_image,
    timeout=10800,
    gpu='T4',
    volumes={"/mnt/data": results_volume})
def evaluate_results(coco_gt, get_inference_time, inference_time, image_ids):
    # Load the detection results into COCO format
    coco_dt = coco_gt.loadRes("/mnt/data/validation_results.json")

    # Initialize the COCOeval object with ground-truth and detection results
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')

    # Restrict the evaluation to the single image
    coco_eval.params.imgIds = image_ids

    if get_inference_time: print('Here was your inference time: ', inference_time)

    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    coco_evaluation_results = {}
    coco_evaluation_results['AP'] = coco_eval.stats[0]
    coco_evaluation_results['AP50'] = coco_eval.stats[1]
    coco_evaluation_results['AP75'] = coco_eval.stats[2]
    coco_evaluation_results['AP_small'] = coco_eval.stats[3]
    coco_evaluation_results['AP_medium'] = coco_eval.stats[4]
    coco_evaluation_results['AP_large'] = coco_eval.stats[5]
    coco_evaluation_results['AR1'] = coco_eval.stats[6]
    coco_evaluation_results['AR10'] = coco_eval.stats[7]
    coco_evaluation_results['AR100'] = coco_eval.stats[8]
    coco_evaluation_results['AR_small'] = coco_eval.stats[9]
    coco_evaluation_results['AR_medium'] = coco_eval.stats[10]
    coco_evaluation_results['AR_large'] = coco_eval.stats[11]
    if get_inference_time:
        coco_evaluation_results['inference_time'] = inference_time

    with open("/mnt/data/evaluation_results.json", 'w') as f:
        json.dump(coco_evaluation_results, f)

@app.function(
    image=gpu_image,
    timeout=10800,
    gpu='T4',
    volumes={"/mnt/data": results_volume},
    mounts=[data_mount])
def main(get_inference_time):
    print('Starting Object Detection Inference App', flush=True)
    # Check if GPU is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # Initialize the processor and model
    model = DetaForObjectDetection.from_pretrained("jozhang97/deta-swin-large").to(device)
    processor = DetaImageProcessor.from_pretrained("jozhang97/deta-resnet-50")

    # Path to the COCO annotation file
    annotation_file = '/mnt/images_data/instances_val2017.json' 

    # Load COCO annotations
    coco_gt = COCO(annotation_file)

    image_ids = coco_gt.getImgIds()

    img_dir = '/mnt/images_data/val2017'

    results, inference_time = run_inference.remote(coco_gt, image_ids, img_dir, model, processor, device)
    generate_inference_json.remote(results)
    evaluate_results.remote(coco_gt, get_inference_time, inference_time, image_ids)

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--get_inference_time", type=bool, default=True)
    args = parser.parse_args()

    with modal.enable_output():
        with app.run():
            main.remote(get_inference_time=args.get_inference_time)


