# Visual Jailbreak Attack on Aligned Vision-Language Models (LLaVA)

This project aims to **reproduce visual adversarial jailbreak attacks** against aligned Vision-Language Models (VLMs), following the methodology of the AAAI 2024 paper *"Visual Adversarial Examples Jailbreak Aligned Large Language Models"*.

We test whether carefully constructed adversarial images can bypass alignment mechanisms (e.g., instruction refusal) in open-source VLMs like **LLaVA-v1.5-7B**.

---

## What this project does

- Load an aligned VLM (LLaVA-1.5 7B) with visual input support
- Provide a harmless image + harmful prompt → model should refuse
- Optimize an adversarial image using PGD
- Test whether the model now **outputs harmful content** (i.e., jailbreak)
- Evaluate output with manual inspection or automatic toxicity detection
