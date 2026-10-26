# 2026 Generative AI — LLM & RAG Project Portfolio

Hands-on projects from the **Generative Artificial Intelligence** course at National Yang Ming Chiao Tung University (Spring 2026). The focus is on **applied LLM engineering**: retrieval-augmented generation, prompt engineering, parameter-efficient fine-tuning of open-source models, local model serving, and rigorous evaluation — plus two diffusion models trained from scratch.

All code is in [`generative-ai-coursework/`](generative-ai-coursework/); every project has its own README with method, reproduce commands, and results.

## Skills at a glance

| Area | What I built | Projects |
|---|---|---|
| **RAG pipelines** | Sentence-window chunking → HyDE query expansion → hybrid **BM25 + BGE dense** retrieval fused with RRF → **cross-encoder reranking** → grounded answer generation; PDF parsing with PyMuPDF | [HW2](generative-ai-coursework/hw2-rag-document-qa/), [HW3](generative-ai-coursework/hw3-hallucination-classifier/) |
| **Prompt engineering** | Few-shot + Chain-of-Thought prompts with TF-IDF-retrieved exemplars; HyDE (hypothetical document) prompting; calibration examples for confusable classes; strict output-format constraints and answer post-processing | [HW1](generative-ai-coursework/hw1-medical-qa-lora/), [HW2](generative-ai-coursework/hw2-rag-document-qa/), [HW3](generative-ai-coursework/hw3-hallucination-classifier/) |
| **LLM fine-tuning (PEFT)** | **LoRA** on Llama-3.2-1B (Transformers + PEFT, 4-bit bitsandbytes); **QLoRA** on Qwen2.5-3B (Unsloth), checkpoint selection by dev Macro-F1, minority-class oversampling | [HW1](generative-ai-coursework/hw1-medical-qa-lora/), [HW3](generative-ai-coursework/hw3-hallucination-classifier/) |
| **Open-source models** | Llama-3.2-1B/3B-Instruct, Qwen2.5-3B-Instruct, BGE embeddings (`bge-large-en-v1.5`) and reranker (`bge-reranker-v2-m3`), Stable-Diffusion VAE, CLIP | all |
| **Model serving & APIs** | Served Llama-3.2-3B locally with **Ollama** and called it through its REST API (retries, timeouts, fallback); high-throughput HTTP data collection with `httpx` + thread pools | [HW2](generative-ai-coursework/hw2-rag-document-qa/), [HW4](generative-ai-coursework/hw4-ptt-stock-crawler/) |
| **Evaluation & data analysis** | Accuracy / Macro-F1 / IoU / FID / CLIP-score; confusion matrices and training curves; ablations (multi-query, MMR, reranker on/off); found and fixed a mismatch between the course scorer's Inception model and the reference statistics | all |
| **Engineering practice** | Reproducible CLIs with `argparse`, pinned `requirements.txt`, caching of parsed/embedded data, profiling-driven optimization (BeautifulSoup → lxml: 4m52s → 79s), documented design decisions | all |

## Projects

### LLM applications

| Project | Summary | Result |
|---|---|---|
| [**HW2 · RAG Document QA**](generative-ai-coursework/hw2-rag-document-qa/) | Per-paper question answering: HyDE + hybrid BM25/BGE retrieval + BGE reranker, answers from Llama-3.2-3B served via Ollama | Evidence-grounded answers with retrieved passages |
| [**HW3 · Hallucination Type Classification**](generative-ai-coursework/hw3-hallucination-classifier/) | Classify hallucination types in AI-written peer reviews: PDF evidence retrieval (BM25 + dense + reranker) feeding a QLoRA-tuned Qwen2.5-3B | 5-class classifier, Macro-F1 checkpoint selection |
| [**HW1 · Medical Multiple-Choice QA**](generative-ai-coursework/hw1-medical-qa-lora/) | Pathology MCQ with Llama-3.2-1B: TF-IDF few-shot retrieval + CoT prompting + LoRA fine-tuning | **80.7 %** test accuracy |

### Data collection

| Project | Summary | Result |
|---|---|---|
| [**HW4 · PTT Stock Board Crawler**](generative-ai-coursework/hw4-ptt-stock-crawler/) | Single-file CLI crawling a full year of forum posts with four analysis subcommands; `httpx` keep-alive, `lxml` XPath, 24-thread pool | IoU **1.00 / 1.00 / 1.00 / 0.99**; 28.5 k posts in 28 s |

### Generative vision (from scratch)

| Project | Summary | Result |
|---|---|---|
| [**HW5 · Latent Diffusion Faces**](generative-ai-coursework/hw5-latent-diffusion-faces/) | Latent DDPM U-Net trained from scratch on SD-VAE latents, augmentation + EMA, 256×256 | **FID 9.84** |
| [**HW6 · Text-Conditional Diffusion**](generative-ai-coursework/hw6-text-conditional-diffusion/) | Hand-written EDM U-Net (27.8 M params), Heun sampler, classifier-free guidance, CLIP text conditioning, 64×64 | **FID 44.27**, CLIP-T 0.272 |

<p align="center">
  <img src="generative-ai-coursework/hw5-latent-diffusion-faces/results/samples_grid.png" width="48%" alt="HW5 samples">
  <img src="generative-ai-coursework/hw6-text-conditional-diffusion/results/samples_grid.png" width="36%" alt="HW6 samples">
</p>

## Tech stack

**Python · PyTorch · Hugging Face Transformers / PEFT · Unsloth · bitsandbytes · sentence-transformers · rank_bm25 · PyMuPDF · Ollama · open_clip · diffusers (VAE only) · httpx · lxml · scikit-learn · pandas**

## Notes

- Model weights (45 MB – 1 GB) and course datasets are not included; each README documents how to retrain. HW5 weights are linked from Google Drive.
- Assignment specs are summarized in each project README; the original course PDFs are not redistributed.

---
**Yu-Ren Jou (周宥任)** · Dept. of Electronics and Electrical Engineering, NYCU
