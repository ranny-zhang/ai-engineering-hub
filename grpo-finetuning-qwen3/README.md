# GRPO Fine-tuning on Fireworks Training API

This project demonstrates how to fine-tune **Qwen3-8B** for structured JSON invoice extraction using GRPO (Group Relative Policy Optimization) via the Fireworks Training API. The training loop runs from a local notebook. The model trains on remote GPUs managed by Fireworks.

The fine-tuned model scores **82% schema-valid accuracy** on a held-out eval set, beating both the base Qwen3-8B (62%) and GPT-4.1 (58%) on the same task.

![Eval Chart](eval_chart.png)

---

## Setup and installations

**Get API Keys**:
- [Fireworks AI](https://fireworks.ai) — needed for training and inference. Requires RLOR (training) access. Store it as `FIREWORKS_API_KEY` in a `.env` file.
- [OpenRouter](https://openrouter.ai) — needed for base model and GPT-4.1 eval. Store it as `OPENROUTER_API_KEY` in a `.env` file.

Refer to `.env.example` for the structure of the file. You will also need your Fireworks account ID stored as `FIREWORKS_ACCOUNT_ID`.

**Clone the Fireworks cookbook**:
```bash
git clone https://github.com/fw-ai/cookbook.git
```

**Install Dependencies**:

Ensure you have Python 3.10 or later installed.

```bash
uv venv
source .venv/bin/activate
uv pip install python-dotenv jsonschema openai fireworks-ai matplotlib
uv pip install -e "cookbook/training[training]"
uv pip install eval-protocol nest_asyncio
```

Select the virtual environment as the kernel in the notebook.

**Run the notebook**:

Open and run `grpo_json_extraction.ipynb` end-to-end. The notebook covers:

1. Reward function that scores JSON completions against a schema
2. Dataset upload to Fireworks
3. GRPO training loop against remote GPUs
4. Baseline eval on base Qwen3-8B
5. Post-training eval on the fine-tuned model
6. GPT-4.1 comparison eval
7. Inference on the deployed model

---

## 📬 Stay Updated with Our Newsletter!

**Get a FREE Data Science eBook** 📖 with 150+ essential lessons in Data Science when you subscribe to our newsletter! Stay in the loop with the latest tutorials, insights, and exclusive resources. [Subscribe now!](https://join.dailydoseofds.com)

[![Daily Dose of Data Science Newsletter](https://github.com/patchy631/ai-engineering/blob/main/resources/join_ddods.png)](https://join.dailydoseofds.com)

---

## Contribution

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.