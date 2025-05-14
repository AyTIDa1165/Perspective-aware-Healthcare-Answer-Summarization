# PerSpectraMed: Adaptable Perspective Summarization for Healthcare QA via LoRA-Tuned Language Models

## 📈 Overview

**PerSpectraMed** addresses the challenge of summarizing complex and multi-perspective discussions in healthcare community question-answering (CQA) forums. These forums often contain varied viewpoints, including factual clarifications, treatment suggestions, personal experiences, and follow-up queries. The goal of this project is to generate concise, perspective-aware summaries that retain critical medical information.

We benchmark zero-shot inference on SOTA large language models (LLMs) like **Gemini 2.0 Flash**,  **Gemini 3 27B** and **ChatGPT 4o-mini** to establish a baseline. We propose our own solution using a fine-tuned **Gemma2 model** using **Low-Rank Adaptation (LoRA)**. Additionally, we incorporate a **span prediction mechanism** to accurately identify and supply relevant input segments to the language models for effective summarization.

---

## 📂 PUMA Dataset

https://drive.google.com/drive/folders/1yUNr5na2dQq2FUqdGkcinsXVemp-Ba7Q?usp=sharing

---

## ⚖️ LoRA Weights

https://drive.google.com/drive/folders/1UHb0DIAUn1qWu32nFb39SRnBG65AlL3P?usp=sharing

---

## ✨ Key Contributions

* 🔹 Baseline integration of zero-shot and prefix-tuned LLMs for perspective control
* 🔹 LoRA-based fine-tuning of Gemma2 for efficient training
* 🔹 Span prediction module to maintain essential clinical details
* 🔹 Evaluation on curated healthcare CQA data using ROUGE, BLEU, METEOR, and BERTScore

---

## 🚪 Features

* ✅ Perspective-based summarization
* ✅ Parameter-efficient fine-tuning with LoRA
* ✅ Retention of key medical spans
* ✅ Easy-to-extend evaluation pipeline

---

## 📊 Results

| **Model**                     | **R-1 F1** | **R-2 F1** | **R-L F1** | **BLEU** | **METEOR** | **BERTScore** |
|-------------------------------|------------|------------|------------|----------|------------|---------------|
| **Gemma 3 27B (Zero-shot)**    | 26.7       | 8.4        | 24.1       | 0.0433   | 0.3411     | 0.8735        |
| **Gemini 2.0 Flash (Zero-shot)**| 31.6       | 11.5       | 28.7       | 0.0659   | 0.3712     | 0.8862        |
| **PLASMA**                     | 23.2       | 7.3        | 21.3       | 0.040    | 0.244      | 0.869         |
| **Gemma2 + LoRA + Span**       | **28.9**   | **9.9**    | **26.1**   | **0.0528**| **0.2859** | **0.8803**     |

**Note**: Our LoRA-based method with span prediction outperforms the zero-shot baselines (Gemma 3 27B, Gemini 2.0 Flash) and improves over prefix tuning across multiple metrics.

---

## 🔬 Evaluation

To evaluate the generated summaries, follow these steps:

### 1. Format Predictions

- Format the generated summaries into a CSV file with the following columns: `uri`, `question`, `output`, `reference`.
- Name the file as `predictions_<model_name>.csv` and place it in the `Output` directory.

### 2. Run Evaluation Notebook

- Navigate to the `Summarization` directory and locate the `eval.ipynb` notebook.
- Set the variable `MODEL_NAME` to `<model_name>` within the notebook.
- Execute all cells in the notebook to evaluate the model's performance.

Supported metrics:

* ROUGE-1, ROUGE-2, ROUGE-L
* BLEU-4
* METEOR
* BERTScore

---

## 🧰 Future Work

- **Reinforcement Learning**: Incorporate user feedback (e.g., reading durations, upvotes) for dynamic summary refinement.
- **Perspective Expansion**: Integrate additional perspectives such as empathy, sentiment, or sarcasm detection—especially relevant in mental health or delicate medical discussions.
- **Broader Domain Testing**: Validate transferability to subdomains like pediatrics, oncology, or rare diseases, possibly combining retrieval augmentation for specialized terminologies.
- **Integration with GPT, Deepseek**: Investigate advanced generative models or retrieval-based hybrids to further boost fluency and factual grounding.

---

## 🧑‍💼 Authors

* **Aarya Gupta** — [aarya22006@iiitd.ac.in](mailto:aarya22006@iiitd.ac.in)
* **Aditya Aggarwal** — [aditya22028@iiitd.ac.in](mailto:aditya22028@iiitd.ac.in)
* **Arpan Verma** — [arpan22105@iiitd.ac.in](mailto:arpan22105@iiitd.ac.in)

---

## 📅 Citation

For the baseline experiments, please refer to the following repository:
- **PUMA-PLASMA-ACL**: [GitHub Link to Model](https://github.com/GauriNaik826/PUMA-PLASMA-ACL)

```
@inproceedings{gupta2025perspectramed,
  title={PerSpectraMed: Adaptable Perspective Summarization for Healthcare QA via LoRA-Tuned Language Models},
  author={Gupta, Aarya and Aggarwal, Aditya and Verma, Arpan},
  year={2025},
  note={Under submission}
}
```

---

## 📄 License

This project is intended for academic use only. Please refer to the LICENSE file for more information.
