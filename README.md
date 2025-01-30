# Rethinking Bottlenecks in Safety Fine-Tuning of Vision Language Models
[Yi Ding*](https://dripnowhy.github.io/)<sup style="color: #e17e7e;">1</sup><sup>,</sup><sup style="color: #3f7fba;">2</sup>, [Lijun Li*](https://scholar.google.com/citations?user=394j5K4AAAAJ&hl=zh-CN)<sup style="color: #e17e7e;">1</sup>, [Bing Cao](https://bcaosudo.github.io/)<sup>†</sup><sup style="color: #3f7fba;">2</sup>,  [Jing Shao](https://amandajshao.github.io/)<sup>†</sup><sup style="color: #e17e7e;">1</sup>

<sup style="color: #e17e7e;">1</sup>Shanghai Artificial Intelligence Laboratory, <sup style="color: #3f7fba;">2</sup>Tianjin University

<sup>*</sup>Equal contribution <sup>†</sup>Corresponding author

<a href='https://github.com/DripNowhy/MIS'><img alt="Static Badge" src="https://img.shields.io/badge/Paper-arXiv-red"></a> <a href='https://dripnowhy.github.io/MIS/'><img src='https://img.shields.io/badge/Project-Page-green'></a> <a href='https://huggingface.co/collections/Tuwhy/mis-679ae8748aa3744dfb0d453e'><img alt="Static Badge" src="https://img.shields.io/badge/%F0%9F%A4%97-Dataset-blue"> <a href='https://huggingface.co/collections/Tuwhy/mirage-679875ff7bb3855234c94b42'><img alt="Static Badge" src="https://img.shields.io/badge/%F0%9F%A4%97-Model-blue">

</a>

![Teaser figure](./static/images/pipeline.png)

<span style="color:red">📢 <strong><i>Please consider citing or 🌟 MIS if our repository is helpful to your work!</i></strong></span>

## 🎙️ News
📅[2025-01-30] 🧨 Our Dataset, MIRage series VLMs are released now! 🧨

## 📌 Content
- [Introduction](#-introduction)
- [Dataset](#-dataset)
- [Models](#️-models)
- [Evaluation](#-evaluation)

## 📝 Introduction
<p align="center">
  <img src="./static/images/motivation.png" alt="Motivation of MIS" width="70%">
</p>

Large Vision-Language Models (VLMs) have achieved remarkable performance across a wide range of tasks. However, their deployment in safety-critical domains poses significant challenges. Existing safety fine-tuning methods, which focus on textual or multimodal content, fall short in addressing challenging cases or disrupt the balance between helpfulness and harmlessness. Our evaluation highlights a safety reasoning gap: these methods lack safety visual reasoning ability, leading to such bottlenecks. To address this limitation and enhance both visual perception and reasoning in safety-critical contexts, we propose a novel dataset that integrates multi-image inputs with safety Chain-of-Thought (CoT) labels as fine-grained reasoning logic to improve model performance. Specifically, we introduce the Multi-Image Safety (MIS) dataset, an instruction-following dataset tailored for multi-image safety scenarios, consisting of training and test splits. Our experiments demonstrate that fine-tuning InternVL2.5-8B with MIS significantly outperforms both powerful open-source models and API-based models in challenging multi-image tasks requiring safety-related visual reasoning. This approach not only delivers exceptional safety performance but also preserves general capabilities without any trade-offs. Specifically, fine-tuning with MIS increases average accuracy by 0.83% across five general benchmarks and reduces the Attack Success Rate (ASR) on multiple safety benchmarks by a large margin.

## 📊 Dataset
![Dataset figure](./static/images/dataset.png)
You can download our [MIS dataset](https://huggingface.co/collections/Tuwhy/mis-679ae8748aa3744dfb0d453e) from Huggingface 🤗.

## 🏜️ Models
![MIRage](./static/images/model_fig.png)
<p align="center">
    <b>MIRage: M</b>ulti-<b>I</b>mage <b>R</b>e<b>a</b>sonin<b>g</b> Saf<b>e</b>ty Fine-Tuning
</p>

- You can download InternVL2.5-8B fine-tuned with our MIRage and MIS training data from [here](https://huggingface.co/Tuwhy/InternVL2.5-8B-MIRage) 🤗.
- You can download Qwen2-VL-7B-Instruct fine-tuned with our MIRage and MIS training data from [here](https://huggingface.co/Tuwhy/Qwen2-VL-7B-MIRage) 🤗.

## 📐 Evaluation

### Inference
- Clone our MIS repo:
  ```
  git clone https://github.com/DripNowhy/MIS.git
  cd MIS
  ```
- Data Preparation: First, download our [MIS test set](#-dataset). Then, organize your data following the structure below:
  ```
  ├── easy_image
  │   ├── 1
  │   │   └── object1.png
  │   │   └── object2.png
  │   └── ...
  ├── hard_image
  │   ├── 1
  │   │   └── object1.png
  │   │   └── object2.png
  │   └── ...
  ├── real_image
  │   ├── 1
  │   │   └── object1.png
  │   │   └── object2.png
  │   └── ...
  ├── mis_easy.json
  ├── mis_hard.json
  └── mis_real.json
  ```


- For Qwen2-VL series, InternVL2.5 series, Phi3.5-Vision-Instruct, Idefics3-8B, LLaVA-OneVision-72b-Chat-hf models. We recommend you to deploy VLMs using [vLLM](https://github.com/vllm-project/vllm). 
  ```
  pip install vllm
  pip install qwen_vl_utils
  bash scripts/inf_vllm.sh
  ```

- For LLaVA-NeXT-Interleave, first, install the LLaVA environment by following the instructions in the [LLaVA-NeXT Official Repository](https://github.com/LLaVA-VL/LLaVA-NeXT/tree/main). Once the LLaVA environment is set up, you can run inferences on the model using the following code:
  ```
  bash scripts/inf_llava.sh
  ```

- For DeepSeek-VL2, first, install the deepseek environment by following the instructions in the [DeepSeek-VL2 Official Repository](https://github.com/deepseek-ai/DeepSeek-VL2). Once the deepseek environment is set up, you can run inferences on the model using the following code:
  ```
  bash scripts/inf_deepseek.sh
  ```

### Evaluation
Now, you can use GPT-4o as evaluator to do the evaluation. Make sure you have fulfilled your openai api in `evaluation/gpt_eval.py`.
```
bash scripts/eval_all.sh
```

<h2 id="citation">🌟 Citation</h2>

still tuning~

<hr>