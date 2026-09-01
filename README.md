# FDER

The repo is the official implementation for the paper: **FDER: Frequency-Decoupled Enhanced Retrieval for Long-term Time Series Forecasting**.

[[Paper]](TODO) [[Code]](https://github.com/Gardeniae/FDER)

## Introduction

Long look-back windows provide rich historical contexts for long-term time series forecasting. However, longer inputs also introduce redundant information and make historical pattern matching more challenging.

FDER is motivated by two observations:

* **Multi-periodicity is difficult to disentangle in the time domain.**
  Real-world time series often contain multiple coexisting cycles, such as short-term fluctuations, daily patterns, weekly patterns, and long-range trends. Time-domain trend-seasonal decomposition may mix these components into coarse trend or seasonal terms.

* **Time-domain retrieval can suffer from phase shifts and hard alignment.**
  Similar periodic patterns may appear at different relative positions within fixed-length patches. Directly comparing raw temporal patches may therefore underestimate their similarity.

To address these issues, FDER reorganizes long historical contexts from a **frequency-decoupled retrieval** perspective. It separates regular frequency structures from residual dynamics and selectively retrieves useful historical evidence for future prediction.

<p align="center">
<img src="./figures/architecture.png" alt="FDER Architecture" align=center />
</p>

## Overall Architecture

FDER decomposes historical information into complementary predictive components and aggregates them for final forecasting.

The framework contains four prediction branches:

* **Low-frequency trend branch** models slowly evolving temporal changes.
* **Dominant-frequency branch** captures salient periodic components across different scales.
* **Historical spectral similarity branch** retrieves globally similar frequency patterns from historical patches.
* **Event-aligned residual retrieval branch** recalls local residual responses for short-term perturbations and irregular variations.

The final prediction is obtained by additive fusion:

```text
Y_pred = Y_low + Y_dom + Y_sim + Y_evt
```

Each branch plays a distinct role:

| Branch  | Purpose                      |
| ------- | ---------------------------- |
| `Y_low` | Trend continuation           |
| `Y_dom` | Multi-periodic extrapolation |
| `Y_sim` | Global spectral reference    |
| `Y_evt` | Local event-like response    |

This design allows FDER to jointly exploit regular spectral structures and local dynamic cues within a unified forecasting framework.

## Usage

### 1. Environment

Create a Python environment:

```bash
conda create -n fder python=3.8
conda activate fder
```

Install the required dependencies:

```bash
pip install torch numpy pandas scikit-learn matplotlib
```

Please install the PyTorch version that matches your CUDA environment.

### 2. Data Preparation

Please organize the datasets under the `./datasets/` directory:

```text
datasets/
├── ETT-small/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── weather/
│   └── weather.csv
├── solar/
│   └── solar_AL.txt
└── exchange_rate/
    └── exchange_rate.csv
```

Make sure the paths in the running scripts are consistent with your local dataset directory.

### 3. Run FDER

We provide running scripts under `./script/`.

```bash
bash ./script/fder_etth1.sh
bash ./script/fder_etth2.sh
bash ./script/fder_ettm1.sh
bash ./script/fder_ettm2.sh
bash ./script/fder_weather.sh
bash ./script/fder_solar.sh
bash ./script/fder_exchange.sh
```

You can specify the GPU device as follows:

```bash
CUDA_VISIBLE_DEVICES=0 bash ./script/fder_ettm2.sh
```

## Example Command

A typical command for running FDER is:

```bash
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./datasets/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_1920_96_FDER \
  --model FDER \
  --data ETTm2 \
  --features M \
  --seq_len 1920 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --train_epochs 15 \
  --learning_rate 0.005 \
  --batch_size 32 \
  --itr 1
```

## Important Arguments

### Forecasting Arguments

| Argument          | Description                            |
| ----------------- | -------------------------------------- |
| `--seq_len`       | Input look-back window length          |
| `--label_len`     | Start token length                     |
| `--pred_len`      | Forecasting horizon                    |
| `--features`      | Forecasting setting: `M`, `S`, or `MS` |
| `--enc_in`        | Number of input variables              |
| `--dec_in`        | Number of decoder input variables      |
| `--c_out`         | Number of output variables             |
| `--train_epochs`  | Number of training epochs              |
| `--learning_rate` | Learning rate                          |
| `--batch_size`    | Batch size                             |

### FDER-specific Arguments

| Argument       | Description                                                 |
| -------------- | ----------------------------------------------------------- |
| `--revise_len` | Patch length used for historical refinement                 |
| `--stride`     | Patching stride; `-1` means `revise_len // 2`               |
| `--topk`       | Number of retrieved historical patches                      |
| `--top_m`      | Number of top frequency bins used in the retrieval key      |
| `--n_low`      | Number of low-frequency bins for trend modeling             |
| `--n_dom`      | Number of dominant-frequency bins for periodic modeling     |
| `--evt_d`      | Bottleneck dimension of the event residual adapter          |
| `--tau_evt`    | Event-strength threshold for residual event retrieval       |
| `--p_evt`      | Number of top residual scores used for robust event scoring |

## Repository Structure

```text
FDER/
├── data_provider/        # Data loading and preprocessing
├── exp/                  # Experiment pipeline
├── figures/              # Figures used in README and paper
├── layers/               # Neural network layers
├── models/               # Model implementation
│   └── FDER.py
├── script/               # Running scripts
├── utils/                # Utility functions
├── run.py                # Main entry
└── README.md
```

## Citation

If you find this repository useful, please consider citing our work:

```bibtex
@article{fder2026,
  title   = {FDER: Frequency-Decoupled Enhanced Retrieval for Long-term Time Series Forecasting},
  author  = {Anonymous},
  journal = {Under Review},
  year    = {2026}
}
```

## Acknowledgement

We appreciate the following repositories for their valuable codebases and experimental pipelines:

* Time-Series-Library
* PatchTST
* iTransformer
* FEDformer
* Autoformer
* Informer

## Contact

If you have any questions or suggestions, please feel free to contact:

```text
Your Name: your.email@example.com
```
