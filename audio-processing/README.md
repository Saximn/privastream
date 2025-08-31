# TikTok Techjam 2025 - PII Data Detection Track 7 

This repository contains the code and configurations for our solution in the **PII Data Detection** competition hosted by TikTok. Our team's approach and results are detailed in the **project_report.md** file.


## Table of Contents

  - [Table of Contents](#table-of-contents)
  - [Setup](#setup)
    - [Hardware](#hardware)
    - [Software](#software)
    - [Dependencies](#dependencies)
    - [Datasets](#datasets)
  - [Training](#training)

## Setup

### Hardware
- **GPU:** NVIDIA A100 40GB


### Software

- **Python**: 3.10.13
- **CUDA**: 12.1

### Dependencies

Clone the repository and install the required Python packages:

```shell
git clone https://github.com/Saximn/tiktok-techjam-2025.git
cd tiktok-techjam-2025
pip install -r requirements.txt
```

### Datasets

Ensure the Kaggle API is installed and set up. Use the following script to download the necessary datasets:

```shell
sh ./setup_datasets.sh
```

Note: The script creates a `data` folder in the parent directory and downloads external datasets there.

#### Dataset Attribution

This project uses the following datasets from Kaggle:

1. **PII 1st Solution Datasets** 
   - Dataset: `ympaik/pii-1st-solution-datasets`
   - Author: ympaik
   - URL: https://www.kaggle.com/datasets/ympaik/pii-1st-solution-datasets

2. **PII DD Mistral Generated**
   - Dataset: `nbroad/pii-dd-mistral-generated`  
   - Author: nbroad
   - URL: https://www.kaggle.com/datasets/nbroad/pii-dd-mistral-generated

3. **PII Mixtral8x7B Generated Essays**
   - Dataset: `mpware/pii-mixtral8x7b-generated-essays`
   - Author: mpware  
   - URL: https://www.kaggle.com/datasets/mpware/pii-mixtral8x7b-generated-essays

We gratefully acknowledge the dataset authors for making their data publicly available for research purposes.

## Training
<img width="1327" height="918" alt="image" src="https://github.com/user-attachments/assets/7ee1481f-3967-462d-9c9f-b9b0604f2289" />

Our solution involves five Deberta-v3-large models, incorporating different architectures for diversity and performance. Below are some variants and their training commands:

- Multi-Sample Dropout Custom Model: Improves training stability and performance.

    ```shell
    python train_multi_dropouts.py
    ```

- BiLSTM Layer Custom Model: Adds a BiLSTM layer to enhance feature extraction, includes specific initialization to prevent NaN loss issues.

    ```shell
    python train_bilstm.py
    ```

- Knowledge Distillation: Utilizes well-performing models as teachers to boost a student model's performance, leveraging disparate datasets. It requires a teacher model. We used the best of Multi-Sample dropout models.
Note: it requires a teacher model to be distlled with. We used the best of multi-sample dropout models.

    ```shell
    python train_distil.py
    ```

- Experiment 073: Uses augmented data with name swaps.

    ```shell
    python train_exp073.py
    ```

- Experiment 076: Introduces a random addition of consequential names to training data.

    ```shell
    python train_exp076.py
    ```

## Livestream PII Detection Pipeline

### Pipeline Setup

1. **Install Dependencies**:
   ```shell
   python setup_pipeline.py
   ```

2. **Configure Pipeline**:
   Edit `configs/pipeline_config.yaml` to match your hardware and requirements.


## License 
Under MIT License.
