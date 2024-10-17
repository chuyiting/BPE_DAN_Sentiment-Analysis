# CSE256 PA1 By Eddy

This project includes two parts. Part one contains a **Deep Averaging Network** that supports both Glove embedding and randomly initialized embedding through `torch.nn.Embedding`. Part two contains a **character based BPE or CPE to be precise tokenizer**. The efficacy of the network is shown by training on a Sentiment Analysis Dataset.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Command-Line Flags](#command-line-flags)
- [Models](#models)
- [DAN Model Options](#dan-model-options)

## Installation

1. **Download the code:**
2. **create virtual environment and install dependencies**

   ```bash
   conda env create -f environment.yml
   conda activate dan
   ```

3. **Run the project**
   ```bash
   python main.py --model BOW
   ```
   - Two models available **BOW** & **DAN**
   - For **DAN** model, you can specify either`--tokenizer=path_to_tokenizer` or `--embedding=path_to_embedding`. If no tokenizer is specified, a default word-based tokenizer will be used. If no embedding is specified, default 100 dimentional embedding will be created using `torch.nn.Embedding`. Note that CPE tokenizer does not support Glove embedding.
