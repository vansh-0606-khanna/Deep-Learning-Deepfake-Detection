# Deepfake Detection Using Deep Learning Models

### A Comparative Study on the RWFS and RVK10K Datasets

![Project Poster](Detailed%20Poster.png)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Objectives and Contributions](#2-objectives-and-contributions)
3. [Repository Structure](#3-repository-structure)
4. [Datasets](#4-datasets)
5. [Methodology](#5-methodology)
6. [Model Architectures](#6-model-architectures)
7. [Results](#7-results)
8. [Cross-Dataset Validation](#8-cross-dataset-validation)
9. [Output Summaries](#9-output-summaries)
10. [Discussion](#10-discussion)
11. [Limitations](#11-limitations)
12. [Future Scope](#12-future-scope)
13. [Getting Started](#13-getting-started)
14. [Authors](#14-authors)
15. [References](#15-references)

---

## 1. Overview

The rapid growth of generative models has made it easy to produce highly realistic manipulated face images, commonly known as deepfakes. These images threaten privacy, security and trust in digital media, which makes reliable automatic detection an important research problem.

This project presents a comparative study of several machine learning and deep learning approaches for deepfake image detection. Models ranging from simple fully connected networks to modern vision transformers are trained and evaluated under the same experimental conditions on two image datasets:

| Dataset | Total Images | Description |
| --- | --- | --- |
| RWFS | 7,000 | Real and fake face images |
| RVK10K | 10,000 | Real and fake face images |
| **Total** | **17,000** | Both datasets evaluated |

The evaluated models are an Artificial Neural Network (ANN), an improved ANN, a Convolutional Neural Network (CNN), LeNet, GoogLeNet, VGGNet, EfficientNet-B0, a DINOv2 vision transformer, and a hybrid RNN + LSTM model.

**Key finding:** EfficientNet-B0 achieved the best accuracy of **92%**, ahead of the custom CNN (80%) and the baseline ANN (72%). Compound-scaled convolutional architectures with efficient feature extraction suit deepfake detection on datasets of this size better than fully connected, sequential or large pretrained transformer models.

---

## 2. Objectives and Contributions

The objective of this work is to compare a wide range of architectures under identical conditions and to study how well each one generalises beyond its training data. The main contributions are:

- A comparative evaluation of eight model families for image-based deepfake detection on RWFS (7,000 images) and RVK10K (10,000 images).
- A study of an improved ANN design against the baseline ANN.
- Evidence that EfficientNet-B0 gives the best accuracy (92%), ahead of the CNN (80%) and the ANN (72%).
- An analysis of why DINOv2 and RNN + LSTM underperform in this setting.
- A two-way cross-dataset validation protocol between RWFS and RVK10K, together with a discussion of zero-shot and other learning paradigms.
- A clear statement of limitations and future scope.

---

## 3. Repository Structure

| File | Description |
| --- | --- |
| `DL_Project.ipynb` | Jupyter notebook containing the implementation, training and evaluation of the models |
| `Dataset links.txt` | Links to the datasets used in this study |
| `Detailed Poster.png` | Project poster summarising the study |
| `Detailed analysis with future scopes and Gaps.docx` | Full written analysis including methodology, results, limitations and future scope |
| `Latest Research paper(2023-26).xlsx` | Spreadsheet of recent research papers (2023 to 2026) reviewed for this study |
| `Output Summary 1.png` | Output summary of the experiments (part 1) |
| `Output Summary 2.png` | Output summary of the experiments (part 2) |
| `Output Summary 3.png` | Output summary of the experiments (part 3) |

---

## 4. Datasets

Two image datasets were used in the experiments. Each image is labelled as real or fake, so the task is treated as **binary classification**.

| Dataset | Total Images | Description |
| --- | --- | --- |
| RWFS | 7,000 | Real and fake face images |
| RVK10K | 10,000 | Real and fake face images |

Download links for both datasets are provided in [`Dataset links.txt`](Dataset%20links.txt).

**Preprocessing**

- All images were resized to the input resolution required by each network.
- Pixel values were normalised.
- Data was split into training, validation and test subsets.
- Data augmentation (horizontal flipping, small rotations, brightness and contrast changes, and slight blurring) was applied to the training set to reduce overfitting and improve robustness to image quality and compression.

---

## 5. Methodology

The overall workflow consists of image acquisition, preprocessing, model training, evaluation and final classification.

```
Image Acquisition --> Preprocessing --> Model Training --> Evaluation --> Real / Fake Classification
```

### Learning Strategies

Besides the choice of architecture, the way a model is trained and adapted strongly affects its detection ability.

| Strategy | Idea | Status in This Work |
| --- | --- | --- |
| Training from scratch | Learn all weights from the given data | Used (ANN, CNN, LeNet) |
| Transfer learning | Reuse pretrained weights and adapt them | Used (VGG, GoogLeNet, EfficientNet-B0) |
| Zero-shot learning | Predict on unseen data or classes without retraining | Used (DINOv2, cross-dataset) |
| Few-shot learning | Adapt to a new generator using a handful of labelled samples | Possible |
| Self-supervised learning | Learn features from unlabelled images, then classify | Partly (DINOv2 pretraining) |
| Fine-tuning of transformers | Update all or some transformer layers on the task | Possible |
| Ensemble learning | Combine the predictions of several models | Possible |
| Frequency-domain analysis | Detect GAN fingerprints in the Fourier spectrum | Possible |
| Multimodal detection | Use audio, text and video cues together | Possible |
| Adversarial training | Train against perturbed or attacked images | Possible |

**Transfer learning.** A network pretrained on a large dataset such as ImageNet is reused and its learned filters are adapted to the new task. This is valuable when the dataset contains only a few thousand images. Pretrained VGGNet, GoogLeNet and EfficientNet-B0 backbones were used with a new binary classification head.

**Zero-shot learning.** In this study, zero-shot ideas appear in two forms: DINOv2 features were evaluated in a frozen setting with a light classifier, and every model was tested on an unseen second dataset in the cross-dataset experiment.

---

## 6. Model Architectures

### 6.1 Artificial Neural Network (ANN) and Improved ANN
The baseline ANN flattens the image into a vector and passes it through fully connected hidden layers with ReLU activation, followed by a sigmoid output. Flattening discards spatial structure, so accuracy is limited to about 72%. The improved ANN adds batch normalisation, dropout regularisation and a tuned learning rate to stabilise training and reduce overfitting.

### 6.2 Convolutional Neural Network (CNN)
A custom CNN with stacked convolution, pooling and fully connected layers was trained. Convolutional filters capture local textures and edge inconsistencies common in manipulated faces, which raised accuracy to about 80%.

### 6.3 LeNet, GoogLeNet and VGGNet
- **LeNet** is a shallow CNN suited to small inputs.
- **VGGNet** uses deep stacks of 3x3 convolutions and provides strong features, but has a large number of parameters.
- **GoogLeNet** uses inception modules that process features at multiple scales in parallel.

### 6.4 EfficientNet-B0
The baseline of the EfficientNet family. It uses mobile inverted bottleneck (MBConv) blocks with squeeze-and-excitation and a compound scaling rule that balances network depth, width and input resolution. With about 5.3 million parameters it is much smaller than VGG, yet it extracts rich multi-level features. It gave the best accuracy of **92%** in this study.

### 6.5 DINOv2 Transformer
A vision transformer pretrained with self-supervised learning. Its representations are strong for general vision tasks but are not tuned to the subtle, low-level artefacts of manipulation, and transformers generally need large amounts of data to adapt.

### 6.6 RNN + LSTM
A recurrent model with LSTM layers applied on top of extracted features. Recurrent networks are designed for temporal sequences, but the datasets contain still images with no frame order, so sequential modelling offers little benefit.

---

## 7. Results

### In-Dataset Accuracy

| S.No. | Model | Accuracy (%) | Remark |
| --- | --- | --- | --- |
| 1 | ANN | 72 | Baseline, no spatial features |
| 2 | Improved ANN | 76 | Regularisation helps |
| 3 | CNN | 80 | Learns local features |
| 4 | LeNet | 74 | Too shallow |
| 5 | GoogLeNet | 85 | Multi-scale features |
| 6 | VGGNet | 83 | Heavy, many parameters |
| 7 | DINOv2 (Transformer) | 68 | Needs more data |
| 8 | RNN + LSTM | 65 | Unsuited to still images |
| 9 | **EfficientNet-B0** | **92** | **Best performance** |

EfficientNet-B0 outperformed the CNN by 12 percentage points and the ANN by 20 percentage points.

---

## 8. Cross-Dataset Validation

Testing on the same dataset used for training can overestimate real-world performance, because models may learn dataset-specific cues such as background, lighting, compression or image source instead of true manipulation artefacts. To measure generalisation, a two-way cross-dataset validation was carried out.

- **Experiment A:** train and validate on RWFS (7,000 images), then test on RVK10K (10,000 images).
- **Experiment B:** train and validate on RVK10K, then test on RWFS.
- The same preprocessing, image size, optimiser and number of epochs were used in both experiments, so differences come from the data and not from the settings.

| Model | Train RWFS, Test RVK10K (%) | Train RVK10K, Test RWFS (%) | In-Dataset Accuracy (%) |
| --- | --- | --- | --- |
| ANN | 58 | 56 | 72 |
| CNN | 66 | 64 | 80 |
| EfficientNet-B0 | 79 | 77 | 92 |
| DINOv2 (zero-shot) | 60 | 58 | 68 |

All models lose accuracy on the unseen dataset. EfficientNet-B0 keeps the highest cross-dataset accuracy (79% and 77%), while the ANN drops the most in absolute terms (about 14 to 16 points).

---

## 9. Output Summaries

The following images show the output summaries generated from the experiments.

### Output Summary 1
![Output Summary 1](Output%20Summary%201.png)

### Output Summary 2
![Output Summary 2](Output%20Summary%202.png)

### Output Summary 3
![Output Summary 3](Output%20Summary%203.png)

---

## 10. Discussion

EfficientNet-B0 performed best for three main reasons:

1. Convolutional layers preserve spatial structure, which is essential for spotting blending boundaries and texture artefacts.
2. Compound scaling and squeeze-and-excitation attention let the network emphasise informative channels while keeping the model compact, which lowers the risk of overfitting on 7,000 to 10,000 images.
3. Pretrained ImageNet weights give a strong starting point.

The ANN and improved ANN lack any mechanism for local feature extraction. LeNet is too shallow, and VGGNet is heavy for the amount of data. The DINOv2 transformer and the RNN + LSTM models gave lower accuracy because transformers are data-hungry and the LSTM expects sequential input, whereas the datasets consist of independent images. These findings show that a bigger or more fashionable model does not always give better results.

The cross-dataset experiment adds a second view of the results: a high in-dataset score does not guarantee good generalisation, and the drop in accuracy on the unseen dataset is the more honest measure of real-world usefulness. The zero-shot use of DINOv2 shows that general features alone are not enough for this task, and that supervised adaptation to manipulation artefacts is needed.

---

## 11. Limitations

- **Dataset size:** 7,000 and 10,000 images are small compared with large benchmarks, which limits how well deep models, especially transformers, can be trained.
- **Number of datasets:** only two datasets were used, so conclusions about generalisation to other generators and new deepfake techniques are limited.
- **Still images only:** temporal cues from video, which are important for many deepfakes, were not used.
- **Metrics:** the comparison is based mainly on accuracy. Precision, recall, F1-score and AUC would give a fuller picture, particularly if the classes are imbalanced.
- **Robustness:** models were not tested against adversarial attacks, heavy compression or low-resolution social media images.
- **Hyperparameter tuning:** hyperparameters were not exhaustively tuned for every model, so some weaker results may improve with more tuning.
- **Model scale:** larger models such as EfficientNet-B3 to B7 and fully fine-tuned transformers were not explored because of computational limits.

---

## 12. Future Scope

- Fine-tune DINOv2 and other vision transformers fully on larger datasets, and explore hybrid CNN-transformer models.
- Test larger EfficientNet variants and ensembles of EfficientNet, CNN and transformer models.
- Apply few-shot and meta-learning so a detector can adapt quickly to a new generator with very few labelled examples.
- Extend the work to video using real frame sequences with LSTM or temporal transformers, and add audio-visual consistency checks.
- Add frequency-domain features to detect GAN fingerprints hidden in the spatial domain.
- Evaluate on more datasets and generators, including diffusion-model images, for stronger cross-dataset validation.
- Improve robustness with adversarial training and tests under compression, noise and resizing.
- Add explainability methods such as Grad-CAM to show which facial regions influence the decision.
- Optimise the best model for mobile and real-time use, and build a web or mobile application for end users.

---

## 13. Getting Started

### Prerequisites

- Python 3
- Jupyter Notebook or JupyterLab (Google Colab also works)
- The Python libraries imported at the top of `DL_Project.ipynb`

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/vansh-0606-khanna/<repository-name>.git
   cd <repository-name>
   ```
2. Open `Dataset links.txt` and download the RWFS and RVK10K datasets.
3. Place the datasets in a folder and update the dataset paths in the first cells of `DL_Project.ipynb`.
4. Install any missing libraries imported in the notebook using `pip`.
5. Launch Jupyter and run the notebook from top to bottom:
   ```bash
   jupyter notebook DL_Project.ipynb
   ```

---

## 14. Authors

**Vansh Khanna** and **Prachi**
Department of Computer Science and Engineering, The NorthCap University, Gurugram, India

---

## 15. References

1. M. Tan and Q. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in Proc. ICML, 2019.
2. M. Oquab et al., "DINOv2: Learning robust visual features without supervision," Trans. Machine Learning Research, 2024.
3. K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," in Proc. ICLR, 2015.
4. C. Szegedy et al., "Going deeper with convolutions," in Proc. IEEE CVPR, 2015.
5. Y. LeCun, L. Bottou, Y. Bengio and P. Haffner, "Gradient-based learning applied to document recognition," Proc. IEEE, vol. 86, no. 11, 1998.
6. S. Hochreiter and J. Schmidhuber, "Long short-term memory," Neural Computation, vol. 9, no. 8, 1997.
7. I. Goodfellow et al., "Generative adversarial nets," in Proc. NeurIPS, 2014.
8. A. Rossler et al., "FaceForensics++: Learning to detect manipulated facial images," in Proc. IEEE ICCV, 2019.
