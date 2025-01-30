# Rethinking Bottlenecks in Safety Fine-Tuning of Vision Language Models
[Yi Ding*](https://dripnowhy.github.io/)<sup style="color: #e17e7e;">1</sup><sup>,</sup><sup style="color: #3f7fba;">2</sup>, [Lijun Li*](https://scholar.google.com/citations?user=394j5K4AAAAJ&hl=zh-CN)<sup style="color: #e17e7e;">1</sup>, [Bing Cao](https://bcaosudo.github.io/)<sup>†</sup><sup style="color: #3f7fba;">2</sup>,  [Jing Shao](https://amandajshao.github.io/)<sup>†</sup><sup style="color: #e17e7e;">1</sup>

<sup style="color: #e17e7e;">1</sup>Shanghai Artificial Intelligence Laboratory, <sup style="color: #3f7fba;">2</sup>Tianjin University

<sup>*</sup>Equal contribution <sup>†</sup>Corresponding author

<a href='https://github.com/DripNowhy/MIS'><img alt="Static Badge" src="https://img.shields.io/badge/Paper-arXiv-red"></a> <a href='https://dripnowhy.github.io/MIS/'><img src='https://img.shields.io/badge/Project-Page-green'></a> <a href='https://huggingface.co/datasets/kzhou35/mssbench/tree/main'><img alt="Static Badge" src="https://img.shields.io/badge/%F0%9F%A4%97-Dataset-blue"> <a href='https://huggingface.co/datasets/kzhou35/mssbench/tree/main'><img alt="Static Badge" src="https://img.shields.io/badge/%F0%9F%A4%97-Model-blue">

</a>
</a>

![Teaser figure](./static/images/pipeline.png)

## 🎙️ News
📅[2025-01-30] 🧨 Our Dataset, MIRage series VLMs are released now! 🧨

## 📝 Introduction
<div style="text-align: center;">
  <img src="./static/images/motivation.png" alt="Introduction figure" style="width: 70%;">
</div>

Large Vision-Language Models (VLMs) have achieved remarkable performance across a wide range of tasks. However, their deployment in safety-critical domains poses significant challenges. Existing safety fine-tuning methods, which focus on textual or multimodal content, fall short in addressing challenging cases or disrupt the balance between helpfulness and harmlessness. Our evaluation highlights a safety reasoning gap: these methods lack safety visual reasoning ability, leading to such bottlenecks. To address this limitation and enhance both visual perception and reasoning in safety-critical contexts, we propose a novel dataset that integrates multi-image inputs with safety Chain-of-Thought (CoT) labels as fine-grained reasoning logic to improve model performance. Specifically, we introduce the Multi-Image Safety (MIS) dataset, an instruction-following dataset tailored for multi-image safety scenarios, consisting of training and test splits. Our experiments demonstrate that fine-tuning InternVL2.5-8B with MIS significantly outperforms both powerful open-source models and API-based models in challenging multi-image tasks requiring safety-related visual reasoning. This approach not only delivers exceptional safety performance but also preserves general capabilities without any trade-offs. Specifically, fine-tuning with MIS increases average accuracy by 0.83% across five general benchmarks and reduces the Attack Success Rate (ASR) on multiple safety benchmarks by a large margin.

## 📊 Dataset
![Dataset figure](./static/images/dataset.png)
You can download our [MIS dataset](https://huggingface.co/collections/Tuwhy/mis-679ae8748aa3744dfb0d453e) from Huggingface 🤗.