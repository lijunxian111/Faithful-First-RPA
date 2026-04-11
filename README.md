# Faithful-First-RPA

[🏆ACL'26] Official Repo for Faithful-First Reasoning, Planning, and Acting for Multimodal LLMs
> *”A perceptually faithful model reasons only over what is visually observable; it does not “see” what the image does not reveal.“*

[<strong>Junxian Li*</strong>](https://lijunxian111.github.io), [Xinyue Xu*](https://scholar.google.com/citations?hl=en&user=9AthkQ0AAAAJ), [Sai Ma*](https://scholar.google.com/citations?hl=en&user=9fef3AYAAAAJ), [Di Zhang](https://github.com/trotsky1997), [Sichao Li^](https://sichao-li.github.io/)

</div>


<div align="center">
  <h2 align="center">
  <a href="https://github.com/lijunxian111/Faithful-First-RPA/" target='_blank' style="text-decoration: none;"><img src="https://visitor-badge.laobi.icu/badge?page_id=https://github.com/lijunxian111/Faithful-First-RPA/"></a>
  <a href="https://arxiv.org/abs/2511.08409" style="display: inline-block; text-align: center;">
      <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2511.08409-b31b1b.svg?style=flat">
  </a>
  <a href="https://github.com/lijunxian111/Faithful-First-RPA/stargazers" target='_blank' style="text-decoration: none;"><img src="https://img.shields.io/github/stars/lijunxian111/Faithful-First-RPA"></a>
  <a href=""><img src="https://img.shields.io/badge/License-MIT-green.svg"></a>
</div>

#### 🔥🔥🔥 News

- **2026-04-08:** This repo is released.  
- **2026-04-08:** Our paper is accepted by ACL 2026.

---

> **Abstract:** Multimodal Large Language Models (MLLMs) frequently suffer from unfaithfulness, generating reasoning chains that drift from visual evidence or contradict final predictions. We propose Faithful-First Reasoning, Planning, and Acting (RPA) framework in which FaithEvi provides step-wise and chain-level supervision by evaluating the faithfulness of intermediate reasoning, and FaithAct uses these signals to plan and execute faithfulness-aware actions during inference. Experiments across multiple multimodal reasoning benchmarks show that faithful-first RPA improves perceptual faithfulness by up to 24% over prompt-based and tool-augmented reasoning frameworks, without degrading task accuracy. Our analysis shows that treating faithfulness as a guiding principle perceptually faithful reasoning trajectories and mitigates hallucination behavior. This work thereby establishes a unified framework for both evaluating and enforcing faithfulness in multimodal reasoning. 
---

**Train**  
In the ```train_pope``` folder, you can find codes for training the clip classifier.  

---

**Evaluation**  
```
CUDA_VISIBLE_DEVICES=X python eval_f_steps.py  
```
One 48G GPU is enough!  

---

**Faithful Reasoning**  
```
CUDA_VISIBLE_DEVICES=X python qwen_faith_inference.py
CUDA_VISIBLE_DEVICES=X python internvl_faith_inference.py
CUDA_VISIBLE_DEVICES=X python llava_faith_inference.py  
```
One 48G GPU is enough!  

---

### Citation

```ruby
@article{li2025faithact,
  title={FaithAct: Faithfulness Planning and Acting in MLLMs},
  author={Li, Junxian and Xu, Xinyue and Ma, Sai and Li, Sichao},
  journal={arXiv preprint arXiv:2511.08409},
  year={2025}
}
```
