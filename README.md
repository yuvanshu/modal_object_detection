# modal_object_detection
This repo provides script for running modal applications that employ object detection algorithms for inference. This repo investigates the DETA model and evaluates it upon the standard metrics: Average Precision (AP), AP Across Scales, Average Recall (AR), and AR Across Scales. 

To run inference on the COCO validation dataset which consists of 5000 images, run the following command:

python coco_val_inference.py

By default, the inference time will be included in the evaluation results. 
To turn off inference time calculation, add the following argument like so:

python coco_val_inference.py --get_inference_time=False

To download the evaluation results file to your local directory, run the following command
modal volume get object_detection_results evaluation_results.json ./evaluation_results.json

To run inference on a single image and produce the corresponding scores, labels, and bounding boxes, run the following command:

python single_image_inference.py --image_url=YOUR_IMAGE_URL

Note the single_image_inference.py file expects an image url. Example image urls to try include:

http://images.cocodataset.org/val2017/000000000139.jpg
http://images.cocodataset.org/val2017/000000039769.jpg

By default, a json to store scores, labels, and bounding boxes, inference time will be generated. A png file of the bounding boxes, scores, and labels overlayed on the image will also be generated.

To turn off json file and png file creation, add the following arguments like so:

python single_image_inference.py --get_object_detection_json=False --get_object_detection_img=False --image_url=http://images.cocodataset.org/val2017/000000000139.jpg

The default probability score threshold for accepting an object detection is set to 0.5. If needed, it can be modified by adding the following arguments like so. Here the threshold is 0:

python single_image_inference.py --threshold=0 --image_url=http://images.cocodataset.org/val2017/000000000139.jpg

To download the json file to your local directory, run the following command:

modal volume get single_image_object_detection_results single_image_object_detection.json ./single_image_object_detection.json

To download the png file to your local directory, run the following command:

modal volume get single_image_object_detection_results single_image_object_detection.png ./single_image_object_detection.png

