# Hugging Face Learning Journey: Fine-Tuning DistilBERT for Sentiment Analysis

A 3-day hands-on project exploring the Hugging Face ecosystem, ending with a **fine-tuned DistilBERT model** that classifies movie reviews as positive or negative and is published on the Hugging Face Hub.

**Model on the Hub:** [techuser45/my-distilbert-imdb](https://huggingface.co/techuser45/my-distilbert-imdb)

## Results

| Stage | Test accuracy |
|---|---|
| Before fine-tuning | 49.2% (about a coin flip) |
| After epoch 1 | 84.6% |
| After epoch 2 | **87.4%** |

Trained on a small slice of data (2,000 reviews) for 2 epochs in Google Colab on a free T4 GPU. Training took under a minute.

## What's in this project

| Part | What I did |
|---|---|
| **1. Concepts** | Hugging Face vs traditional ML, and the main components: Hub, Models, Datasets, Transformers library, Tokenizers, Pipelines, Spaces |
| **2. Pipelines** | Ran sentiment analysis and question answering with a few lines of code |
| **3. Multimodal model** | Loaded Janus-Pro-1B in Colab to answer questions about an image |
| **4. Fine-tuning** | Fine-tuned DistilBERT on IMDB reviews with the `Trainer` API |
| **5. Publishing** | Saved the model and pushed it to the Hugging Face Hub |

## Tech stack

- Python
- Hugging Face `transformers`, `datasets`, `accelerate`
- PyTorch
- Google Colab (T4 GPU)

## Fine-tuning workflow

1. **Load data**: `stanfordnlp/imdb`, shuffled, then a 2,000-review train slice and a 500-review test slice
2. **Tokenize**: `distilbert/distilbert-base-uncased` tokenizer, truncated and padded to 256 tokens
3. **Load model**: `AutoModelForSequenceClassification` with `num_labels=2`
4. **Train**: `Trainer` with learning rate `2e-5`, batch size 16, 2 epochs, mixed precision (`fp16`)
5. **Evaluate**: accuracy on the held-out test slice, before and after training
6. **Save and publish**: `save_pretrained`, then `push_to_hub`

## Quick start

Open the notebook in Google Colab, switch the runtime to **T4 GPU**, and run the cells in order.

```python
!pip install transformers datasets accelerate
```

### Use the published model

```python
from transformers import pipeline

clf = pipeline("text-classification", model="techuser45/my-distilbert-imdb")
print(clf("I really enjoyed this film!"))
```

Expected output: a `POSITIVE` label with a confidence score.

## Problems I ran into (and how I fixed them)

| Problem | Cause | Fix |
|---|---|---|
| `Repository id must be 'namespace/name'` | The dataset short name `imdb` is outdated | Use the full address `stanfordnlp/imdb` |
| `ValueError: mutable default <class 'dict'>` when loading Janus-Pro | A very new `transformers` version clashed with Janus's older code | Pin an older version (`transformers==4.47.1`) and restart the session |
| `401 Invalid user token` when logging in | The access token was copied with extra characters | Re-copy it, store it in Colab Secrets, and use `.strip()` |

## Key things I learned

- A **pre-trained model** already understands language, and **fine-tuning** only teaches it a specific task, so small data and short training can still give strong results.
- The **tokenizer and model must match**.
- Custom models like Janus need both their **GitHub code** and the **Hub weights**.
- Never put access tokens in code. Store them in Colab Secrets.

## Limitations

- Trained on only 2,000 movie reviews, so accuracy would likely improve with more data and longer training.
- Reviews are truncated to 256 tokens, so very long reviews lose their ending.
- The model only knows two labels (positive and negative), so neutral text gets forced into one of them.
- It was trained on movie reviews and may perform worse on other kinds of text.

## Next steps

- Build a Gradio demo and host it on Hugging Face Spaces
- Train on more data and compare results
- Write a fuller model card on the Hub

## Author

**Latika Sekhri**
B.Tech Computer Engineering, Guru Nanak Dev University, Amritsar
