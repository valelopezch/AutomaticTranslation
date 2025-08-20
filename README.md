# AutomaticTranslation

This repository contains projects focused on **Automatic Translation** using pre-trained language models, finetuning strategies, and evaluation metrics. The experiments cover both **speech-to-text translation** and **code-to-text translation**, exploring modern architectures such as Whisper, NLLB, M2M100, PLBART, and CodeT5.

📄 **Note:** All detailed reports are written in **English**. This README provides a structured summary of the projects.

---

## 📂 Projects Overview

### 1. Portuguese → English Speech Translation (Common Voice Dataset)
- **Dataset:** [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets).  
- **Methods:**  
  - Speech transcription using **Whisper** (base, small, medium).  
  - Speech translation with Whisper, evaluated using **BLEU** and **COMET** metrics.  
  - Cascade translation pipelines with **NLLB** and **M2M100**.  
  - Finetuning with **Optuna** hyperparameter search and text normalization.  
- **Best Results:**  
  - Whisper (medium) baseline: **WER = 7%**, BLEU = 51.8, COMET = 86.38.  
  - Cascade with NLLB finetuned + text normalization: **BLEU = 54.0**, **COMET = 87.3**:contentReference[oaicite:2]{index=2}.  

---

### 2. Python Code → Docstring Translation (CodeXGLUE Dataset)
- **Dataset:** [CodeXGLUE: Code-to-Text](https://huggingface.co/datasets/google/code_x_glue_ct_code_to_text).  
- **Methods:**  
  - Baselines with general and code-specific pre-trained models:  
    - **NLLB** (Meta, multilingual translation).  
    - **LLAMA-2** (general LLM).  
    - **CodeT5-large** (Salesforce, optimized for code tasks).  
    - **PLBART-large** (UCLANLP, trained on code + NL).  
  - Evaluation using **BLEU**, **COMET**, and **ROUGE** (1, 2, L).  
  - Finetuning experiments with limited data (200 train samples).  
  - Parameter-efficient tuning with **LoRA** and **PrefixTuning**.  
- **Best Results:**  
  - Finetuned PLBART: **BLEU = 85.9**, COMET = 88.35%, ROUGE-L = 92.38.  
  - LoRA on PLBART: **BLEU = 90.3**, COMET = 89.77%, ROUGE-L = 94.49:contentReference[oaicite:3]{index=3}.  

---

## ⚙️ Tech Stack

- **Frameworks & Libraries:**  
  - [Hugging Face Transformers](https://huggingface.co/transformers/) – model loading and finetuning  
  - [OpenAI Whisper](https://github.com/openai/whisper) – speech recognition & translation  
  - [Optuna](https://optuna.org/) – hyperparameter optimization  
  - `torch`, `datasets`, `evaluate` – training, dataset handling, evaluation  
  - `numpy`, `pandas` – data manipulation  
  - `sacrebleu`, `comet`, `rouge-score` – translation metrics  

- **Environments:**  
  - Google Colab with GPU acceleration  
  - Jupyter Notebooks for experimentation  

---

## 📑 Reports

Detailed methodologies, results, and analysis are available in the **English-language reports** included in this repository:

- `Informe_A3_TA.pdf` – Portuguese → English Speech Translation (Whisper, NLLB, M2M100).  
- `Informe_A2_TA.pdf` – Python Code → Docstring Translation (CodeXGLUE, PLBART, CodeT5).  

---

## ✨ Key Takeaways

- Whisper medium strikes a balance between speed and accuracy for speech recognition and translation.

- Cascade pipelines with NLLB provide strong results, especially after finetuning with clean text.

- Code-to-text translation benefits significantly from specialized models like PLBART and CodeT5.

- Parameter-efficient finetuning (LoRA) allows adapting large models to domain-specific tasks with fewer resources.

