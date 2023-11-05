# INTR: A Simple Interpretable Transformer for Fine-grained Image Classification and Analysis

INTR is a novel usage of Transformers to make image classification interpretable. In INTR, we investigate a proactive approach to classification, asking each class to look for itself in an image. We learn class-specific queries (one for each class) as input to the decoder, allowing them to look for their presence in an image via cross-attention.  We show that INTR intrinsically encourages each class to attend distinctly; the cross-attention weights thus provide a meaningful interpretation of the model's prediction. Interestingly, via multi-head cross-attention, INTR could learn to localize different attributes of a class, making it particularly suitable for fine-grained classification and analysis.

![Image Description](git_images/architecture.png)

In the INTR model, each query in the decoder is responsible for the prediction of a class. So, a query looks at itself to find class-specific features from the feature map. First, we visualize the feature map i.e., the value matrix of the transformer architecture to see the important parts of the object in the image. To find the specific features, where the model pays attention in the value matrix, we show the heatmap of the attention of the model. To avoid external interference in the classification, we use a shared weight vector for classification so therefore the attention weight explains the model's prediction.

![Image Description](git_images/teaser.png)

## Fine-tune model and results

INTR on [DETR-R50](https://github.com) backbone, classification performance, and fine-tuned models on different datasets.


| Dataset | acc@1 | acc@5 | Model |
|----------|----------|----------|----------|
| CUB | 71.7 | 89.3 |  [checkpoint](https://github.com)|
| Bird | XXX | XXX |  [checkpoint](https://github.com) |





## Installation Instructions

Create python environment (optional)
```sh
conda create -n intr python=3.8 -y
conda activate intr
```

Clone the repository
```sh
git clone https://github.com/dipanjyoti/INTR.git
cd INTR
```

Install python dependencies

```sh
pip install -r requirements.txt
```

## Data Preparation
Follow the below format for data.
```
datasets
├── dataset_name
│   ├── train
│   │   ├── class1
│   │   │   ├── img1.jpeg
│   │   │   ├── img2.jpeg
│   │   │   └── ...
│   │   ├── class2
│   │   │   ├── img3.jpeg
│   │   │   └── ...
│   │   └── ...
│   └── val
│       ├── class1
│       │   ├── img4.jpeg
│       │   ├── img5.jpeg
│       │   └── ...
│       ├── class2
│       │   ├── img6.jpeg
│       │   └── ...
│       └── ...
```

## Evaluation
To evaluate a pre-trained INTR model quantitatively on a multi GPU (e.g., 4 GPUs)

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 12345 --use_env main.py --resume <checkpoint> --dataset_path <path_to_datasets> --dataset_name <dataset_name>
```
To evaluate a pre-trained INTR model quanlitatively for a default batch size of 1.

```sh
python -m tools.visualization --eval --resume <checkpoint> --dataset_name <dataset name> --class_index <class_number>
```
## Training (Interpretable Transformer)
To train INTR model on a 4-GPU server on a pre-trained DETR model

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 12345 --use_env main.py --finetume <path_to_detr_checkpoint> --dataset_path <path_to_datasets> --dataset_name <dataset_name>
```
