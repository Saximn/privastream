# PII Data Detection - Tiktok Techjam 2025 Report

## Project Overview

This project implements the solution for Tiktok Techjam 2025. The objective was to develop automated techniques to detect and remove PII from Tiktok videos. 

## Solution Architecture

Our approach employs an ensemble of five diverse DeBERTa-v3-large models, each incorporating different architectural modifications to maximize performance and diversity:

### Model Components

1. **Multi-Sample Dropout Custom Model**
   - Improves training stability and performance
   - Uses dropout variations during training to enhance generalization
   - Training command: `python train_multi_dropouts.py`

2. **BiLSTM Layer Custom Model**
   - Adds a BiLSTM layer to enhance feature extraction
   - Includes specific initialization to prevent NaN loss issues
   - Training command: `python train_bilstm.py`

3. **Knowledge Distillation Model**
   - Utilizes well-performing models as teachers to boost student model performance
   - Leverages disparate datasets for improved generalization
   - Requires a teacher model (uses best multi-sample dropout model)
   - Training command: `python train_distil.py`

4. **Experiment 073 Model**
   - Uses augmented data with name swaps
   - Enhances model's ability to generalize across different name variations
   - Training command: `python train_exp073.py`

5. **Experiment 076 Model**
   - Introduces random addition of consequential names to training data
   - Improves detection of various name patterns
   - Training command: `python train_exp076.py`


## Data Processing Strategy

### Preprocessing Steps
1. **Text Segmentation:** Student texts split to ensure each segment doesn't exceed 400 tokens
2. **Context Preservation:** Maintains pre-trained context length of 512 tokens
3. **Tokenization:** Uses DeBERTa tokenizer for consistent text processing
4. **Label Encoding:** PII labels encoded to match input tokens

### Data Augmentation
- **Name Swaps:** Systematic replacement of names in training data
- **Consequential Names:** Random addition of contextually relevant names
- **Synonym Replacement:** Enhanced vocabulary diversity
- **Token-level Augmentation:** Maintains NER structure while increasing variety

## Results and Performance
<img width="1327" height="918" alt="image" src="https://github.com/user-attachments/assets/7ee1481f-3967-462d-9c9f-b9b0604f2289" />


### Final Competition Results
- **Accuracy Score:** 0.96988
- **Ensemble Approach:** Optuna-tuned voting ensemble with computed thresholds and weights
- **Postprocessing:** Diverse set of cross-validation-based postprocessing methods

### Dataset Composition
The solution utilized multiple data sources for comprehensive training:

#### Primary Datasets
- **MPWARE Public Data:** External augmentation dataset
- **Nicholas Broad Public Data:** Community-contributed dataset

#### Enhanced Datasets (Key Contributors)
- **TonyaRobertson Public Data:** High-quality external dataset
- **Custom 2k Generated Dataset:** Synthetically generated data for improved diversity (specific to real-life context Tiktok videos)

### Individual Model Performance

#### Top Performing Models
1. **Custom Model 23 (Multi-Dropout)**
   - **Score:** 0.96659
   - **Architecture:** 2-stage multi-dropout implementation
   - **Training:** `train_multidropout.py`

2. **Exp076 DeBERTa**
   - **Score:** 0.96266
   - **Innovation:** Random addition of consequential names
   - **Training:** `train_exp076.py`

3. **Basic DeBERTa**
   - **Score:** 0.96521
   - **Role:** Baseline performance anchor

4. **Custom Model 2**
   - **Score:** 0.96273
   - **Purpose:** Additional architectural variant

#### Specialized Models
5. **Exp073 DeBERTa**
   - **Score:** 0.95992
   - **Feature:** Name swap augmentation
   - **Training:** `train_exp073.py`

6. **Custom Model Distilled (4-Fold)**
   - **Score:** 0.95881
   - **Method:** Knowledge distillation using custom model as teacher
   - **Configuration:** 4-fold cross-validation setup
   - **Training:** `train_distil.py`

7. **Custom BiLSTM DeBERTa**
   - **Score:** 0.95382
   - **Architecture:** Enhanced with BiLSTM layer for feature extraction
   - **Training:** `train_bilstm.py`

### Ensemble Strategy

#### Voting Ensemble Configuration
- **Optimization:** Optuna-based hyperparameter tuning
- **Method:** Weighted voting with computed thresholds
- **Models Combined:** All 7 individual models
- **Weight Assignment:** Performance-based weighting system

#### Ensemble Benefits
- **Diversity:** Different architectures capture varied patterns
- **Robustness:** Reduced overfitting through model averaging
- **Performance Gain:** 0.96988 vs highest individual score of 0.96659
- **Stability:** Consistent performance across different data distributions

### Postprocessing Pipeline

#### Cross-Validation Based Methods
- **Threshold Optimization:** CV-based optimal decision boundaries
- **Pattern Recognition:** Rule-based corrections for common errors
- **Confidence Scoring:** Model certainty-based filtering
- **Entity Boundary Refinement:** Improved start/end token detection

#### Performance Impact
- **Final Score Improvement:** Postprocessing contributed to achieving 0.96988
- **Error Reduction:** Systematic correction of ensemble prediction errors
- **Consistency:** Standardized output format across all predictions

### PII Categories Detected
The solution successfully identifies 15 categories of personally identifiable information:
- Names (first, last, full)
- Email addresses
- Phone numbers
- URLs
- Social security numbers
- ID numbers
- Street addresses
- Phone numbers
- Student IDs
- And other sensitive identifiers

### Performance Analysis

#### Strengths
- **High Accuracy:** 0.96988 
- **Model Diversity:** 7 different architectural approaches
- **Data Utilization:** Effective use of multiple external datasets
- **Ensemble Optimization:** Sophisticated voting mechanism

#### Key Success Factors
1. **Architectural Diversity:** Multiple DeBERTa variants prevent single-point failures
2. **Data Augmentation:** Strategic name swaps and additions improve generalization
3. **Knowledge Transfer:** Distillation enhances weaker model performance
4. **Optuna Optimization:** Automated hyperparameter tuning for ensemble weights
5. **Postprocessing:** CV-based refinement methods for final predictions

## Training Methodology

### Multi-Model Training Pipeline
1. **Base Model Training:** Standard DeBERTa-v3-large fine-tuning
2. **Architecture Variants:** Each model incorporates unique architectural modifications
3. **Knowledge Transfer:** Distillation from best-performing models
4. **Data Augmentation:** Systematic enhancement of training diversity
5. **Ensemble Integration:** Combining predictions for optimal performance

### Training Commands
```bash
# Multi-Sample Dropout Model
python train_multi_dropouts.py

# BiLSTM Enhanced Model
python train_bilstm.py

# Knowledge Distillation Model
python train_distil.py

# Augmented Data Models
python train_exp073.py
python train_exp076.py
```

## Implementation Details

### Setup Process
```bash
git clone https://github.com/Saximn/tiktok-techjam-2025.git
cd tiktok-techjam-2025
pip install -r requirements.txt
sh ./setup_datasets.sh
```

### Inference Process
- Detailed inference notebook
- Processes test data through all five models
- Applies ensemble logic for final predictions
- Includes post-processing for optimal results


## Key Innovations

1. **Architectural Diversity:** Multiple DeBERTa variants prevent overfitting to single architecture
2. **Strategic Augmentation:** Name swaps and additions improve real-world performance  
3. **Knowledge Distillation:** Leverages best models to improve weaker variants
4. **Ensemble Strategy:** Combines strengths of different approaches
5. **NaN Prevention:** Specific initialization strategies for stable training

## Future Enhancements

### Potential Improvements
1. **Model Efficiency:** Explore lighter models for production deployment
2. **Real-time Processing:** Optimize inference speed for streaming data
3. **Domain Adaptation:** Extend to other educational data types
4. **Multi-language Support:** Expand beyond English educational content
5. **Active Learning:** Incorporate human feedback for continuous improvement

### Production Considerations
- Model versioning and deployment strategies
- Monitoring and alerting for model performance
- A/B testing framework for model updates
- Compliance with educational data privacy regulations

## Conclusion

This solution demonstrates the power of ensemble learning with diverse DeBERTa architectures for PII detection in Tiktok videos. The combination of architectural variants, strategic data augmentation, and knowledge distillation creates a robust system capable of accurately identifying personal information while maintaining high performance across various text types.

The solution's success validates the approach of using multiple model variations rather than relying on a single architecture, providing both improved accuracy and enhanced robustness for real-world applications in educational data privacy protection.

