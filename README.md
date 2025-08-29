# Construction Site Safety PPE Detection

This project focuses on real-time detection of Personal Protective Equipment (PPE) at construction sites using state-of-the-art YOLO models. It helps monitor compliance with safety gear requirements such as hardhats, safety vests, and masks, making it useful for automation, tracking, and monitoring applications.

## Dataset

This project uses the [Construction Site Safety Image Dataset](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow) from Kaggle. The dataset is particularly valuable because it distinguishes between wearing and not wearing safety equipment.

### Dataset Details
- **Number of classes**: 10
- **Annotation format**: YOLO (.txt)
- **Class mapping**: 
 - 0: Hardhat
 - 1: Mask  
 - 2: NO-Hardhat
 - 3: NO-Mask
 - 4: NO-Safety Vest
 - 5: Person
 - 6: Safety Cone
 - 7: Safety Vest
 - 8: Machinery
 - 9: Vehicle


## Implementation

Built a real-time PPE detection web application using:
- **Frontend**: HTML webpage
- **Backend**: React
- **Models**: Trained two YOLO models for comparison

## Model Performance

### YOLOv8 Medium
- **Precision**: 0.9087
- **Recall**: 0.7476
- **mAP@50**: 0.8167
- **mAP@50-95**: 0.5399

### YOLOv11 Large
- **Precision**: 0.9363
- **Recall**: 0.8207
- **mAP@50**: 0.8798
- **mAP@50-95**: 0.6131

The YOLOv11 Large model shows superior performance across all metrics, particularly in recall and mean Average Precision.

## Features

- Real-time PPE detection from camera feed
- Detection of safety equipment compliance
- Identification of safety hazards and equipment
- Multi-class object detection and tracking

## Future Plans

- Integration of enhanced features
- Training on larger, more diverse datasets
- Improved model accuracy and performance
- Extended safety equipment detection capabilities
