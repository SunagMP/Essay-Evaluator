# Essay-Evaluator
This project is an AI-powered essay evaluation tool tailored for IPS, IAS, KPSC aspirants. It leverages Large Language Models (LLMs) to help candidates practice and improve their essay-writing skills with detailed, multi-criteria feedback.

---

## 📌 Project Objective

1. The LLM generates an essay topic.
2. The user (aspirant) writes an essay based on the generated topic.
3. The LLM evaluates the essay on the following criteria:
   - ✅ **Language Control**
   - ✅ **Clarity of Thought (CoT)**
   - ✅ **Depth of Analysis (DoA)**
4. A score (out of 10) is assigned to each criterion.
5. Based on these scores, a final summary and average score is generated.
6. The structured feedback is returned to the user for self-improvement.

---

## 💡 Features

- 🎯 Automatic essay topic generation
- 🧠 LLM-based evaluation on key academic writing traits
- 📝 Feedback includes:
  - Criterion-wise score (Language, CoT, DoA)
  - Constructive suggestions
  - Overall summary and average score
- ✅ Validated output using Pydantic schema

---

## 🧪 Tech Stack

- Python 🐍
- LangGraph
- LangChain 🧱
- Gemini LLM API 💬
- Pydantic (for structured parsing) ✅

---

## Contact

- Made by Sunag M P
- GitHub : @SunagMP

---

## 📦 Installation

Clone the repo:

```bash
git clone https://github.com/YOUR_USERNAME/essay-evaluator.git
cd essay-evaluator
