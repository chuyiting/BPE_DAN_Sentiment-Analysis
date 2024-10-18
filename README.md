# A character based implementation of BPE tokenizer, and Deep Averging Network [1] on Sentiment Analysis

This project includes two parts. Part one contains a **Deep Averaging Network** that supports both Glove embedding and randomly initialized embedding through `torch.nn.Embedding`. Part two contains a **character based BPE or CPE to be precise tokenizer**. The efficacy of the network is shown by training on a Sentiment Analysis Dataset.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Reference](#references)

## Installation

1. **Download the code:**
   ```bash
   git clone https://github.com/chuyiting/BPE_DAN_Sentiment-Analysis.git
   ```
2. **create virtual environment and install dependencies**

   ```bash
   conda env create -f environment.yml
   conda activate dan
   ```

## Usage

**Run the project**

```bash
python main.py --model BOW
```

```bash
python main.py --model DAN --embedding=data/glove.6B.300d-relativized.txt
```

```bash
python main.py --model DAN --tokenizer=tokenizer/model/cpe1000
```

- Two models available **BOW** & **DAN**
- For **DAN** model, you can specify either`--tokenizer=path_to_tokenizer` or `--embedding=path_to_embedding`. If no tokenizer is specified, a default word-based tokenizer will be used. If no embedding is specified, default 100 dimentional embedding will be created using `torch.nn.Embedding`. Note that CPE tokenizer does not support Glove embedding.
- For tokenizer, please specify path up until its base name. Do not include extension. For eample `--tokenizer=tokenizer/model/cpe1000`. Do not put `--tokenizer=tokenizer/model/cpe1000.model`

## References

[1]: Iyyer, M., Manjunatha, V., Boyd-Graber, J., & Daumé III, H. (2015). Deep Unordered Composition Rivals Syntactic Methods for Text Classification. In _Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)_ (pp. 1681–1691). Association for Computational Linguistics. [https://aclanthology.org/P15-1162](https://aclanthology.org/P15-1162).
