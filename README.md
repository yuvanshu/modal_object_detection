# modal_object_detection
This repo provides script for running modal applications that employ object detection algorithms for inference. This repo investigates the DETA model and evaluates it upon the standard metrics: Average Precision (AP), AP Across Scales, Average Recall (AR), and AR Across Scales. 

## Installation
Before running the inference scripts, make sure you have the required dependencies installed. You can install them using the following command:

```bash
pip install -r requirements.txt
```

## Running Inference on COCO Validation Dataset
To run inference on the COCO validation dataset which consists of 5000 images, run the following command:

```bash
python coco_val_inference.py
```

### Additional Options
* Turn off inference time calculation:

```bash
python coco_val_inference.py --get_inference_time=False
```
* Download evaluation results:

```bash
modal volume get object_detection_results evaluation_results.json ./evaluation_results.json
```

## Running Inference on a Single Image
To run inference on a single image and produce the corresponding scores, labels, and bounding boxes, run the following command:

```bash
python single_image_inference.py --image_url=YOUR_IMAGE_URL
```

### Example Image URLs

http://images.cocodataset.org/val2017/000000000139.jpg
http://images.cocodataset.org/val2017/000000039769.jpg

## Output Options
By default, a JSON file and a PNG image with bounding boxes, scores, and labels will be generated. You can turn off their creation using the following arguments:

```bash
python single_image_inference.py --get_object_detection_json=False --get_object_detection_img=False --image_url=http://images.cocodataset.org/val2017/000000000139.jpg
```

## Setting Probability Score Threshold
The default score threshold for object detection is set to 0.5. You can modify it like this:

```bash
python single_image_inference.py --threshold=0 --image_url=http://images.cocodataset.org/val2017/000000000139.jpg
```

## Downloading Results
To download the JSON and PNG files to your local directory, use the following commands:

* Download JSON:

```bash
modal volume get single_image_object_detection_results single_image_object_detection.json ./single_image_object_detection.json
```

* Download PNG:

```bash
modal volume get single_image_object_detection_results single_image_object_detection.png ./single_image_object_detection.png
```

## Metrics and Evaluation
The model is evaluated using the following metrics:

* Average Precision (AP)
* AP Across Scales (small, medium, large objects)
* Average Recall (AR)
* AR Across Scales

These metrics help to analyze the performance of the object detection model across varying object sizes and conditions.

