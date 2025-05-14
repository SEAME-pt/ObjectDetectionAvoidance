# Object Detection and Avoidance

Training a Yolo Object detection model, with Lane detection (segmentation) as well.
In the scripts directory file *create_annotations.py* we create the **annotation labels** for lane and object detection. Therefore we pass our images through a trained Yolo11 model, to get the object annotations, and then through our binary lane masks, we add the lane annotations to the label files.

The datasets we used to train/validation are from: [Link to dataset](https://onedrive.live.com/?id=4EF9629CA3CB4B5E%213022&cid=4EF9629CA3CB4B5E&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBbDVMeTZPY1l2bE9sMDQxNHNSb3BGVkgyOTVXP2U9Q2pjbDYy). We used dataset8.