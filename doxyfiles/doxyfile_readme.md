## Project Architecture

Training a Yolo Object detection model, with **Lane detection (segmentation)** as well.

\image html ADR/Fluxograma.jpg "Project Structure" width=70%

## Inference Result

\image html scripts/runs/segment/predict/0010.jpg "Results" width=30%

This image is a result of running *testing.py*, so running predict() of our model. The **lane points** (polygons, mask) are in blue.

## Creating Annotations

In the scripts directory file *create_annotations.py* we create the **annotation labels** for lane and object detection. How do we do this? We pass our images through a pre-trained **Yolo11-seg model**, to get the object polygons. Then, we use **supervision** tools to convert the lane **binary masks** to valid Yolo **polygons**. Finaly, we **merge** the object and lane annotations and get the label files.

Be attentive towards the size of the images and masks, we decided to keep the images square, (training and testing), for compatibility. In the scripts directory, file *resize.py* you can resize images with **letterboxing** (keeping **aspect ratio**), or not.

In these scripts, you might need to change some function **parameters**, the original size of the images, and the **paths** to the images, so that it correctly links to your dataset and original size of your images.

For debugging, you can **visualize the annotations** in *scripts/visual_annotations.py*.

## Datasets

The datasets we used to train/validate are from: [Link to dataset](https://onedrive.live.com/?id=4EF9629CA3CB4B5E%213022&cid=4EF9629CA3CB4B5E&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBbDVMeTZPY1l2bE9sMDQxNHNSb3BGVkgyOTVXP2U9Q2pjbDYy).
We used dataset8 and also some images we took of our lab.

Since we noticed a decay in classifying objects outside those of my dataset. We added some COCO segmentation images, you can see this in *scripts/coco.py*.

## Training and Testing

In *training.py* (scripts directory) where we are retraining our model, we set the augmentations to None since it disrupts our images, and add other augmentations that dont disrupt them, such as brightness, saturation and hue.

For testing, (in *scripts/testing.py*), we call our trained model and set it to **predict**, to test the prediction of a given validation image.

## Jetson Nano

In Jetson, we have an ultralytics Yolo **container**, specific for compatibility with Jetson Nano. This container only runs a yolo model above or equal to version 8. In here we will run our Yolo with lane detection.

We have a self-hosted jetson runner, so that everytime I push the code to github, it will deploy my models to jetson, this code is in *.github/deploy_jetson.yml*.