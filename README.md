# VTT code

- Code implementation for : [Transformer-based multivariate time series anomaly detection using inter-variable attention mechanism](https://www.sciencedirect.com/science/article/pii/S0950705124001424?ref=pdf_download&fr=RR-2&rr=88e52cc379d63158)
- 위 코드 기반으로 작성

# Environments (requirements.txt 참고)
* Python >= 3.9
* cuda == 11.8
* Pytorch == 2.7.0+cu118
* 필수 패키지
    * omegaconf : config 관리
    * einops : 텐서 변환 연산자

# Data Preparation
### Notice
- **dataset_preprocessing.ipynb**로 웨이퍼 데이터(P_FG_60S_1.parquet)와 Score table 데이터(score_results.parquet)를 통해 전처리 진행
- drystrip_dataset 폴더 안에 웨이퍼 데이터 파일과 score table 데이터 파일을 위치시켜야 함


# 코드 실행
```
    # For Training & Inference together
    python main.py --train --test --model VTTSAT

```

* `vis_test_results_allstep.ipynb`
    * 시각화를 위한 코드
    * `2024_06_10_SKT_Market_Top_AI과정.ipynb`의 xai part의 코드 부분 활용
    
# 폴더 구조
```sh
.
├── LICENSE
├── README.md 
├── image
├── dataset_preprocessing.ipynb # For data preprocessing
└── src
    ├── data_provider    # data load, preprocessing, dataloader setting
    │   ├── dataloader.py
    │   └── dataset.py
    ├── layers    # layers for models (attention, embedding, etc.)
    │   ├── Attention.py
    │   ├── Embed.py
    │   └── Transformer_Enc.py
    ├── models
    │   ├── VTTPAT.py
    │   └── VTTSAT.py
    ├── utils    # utils
    │   ├── metrics.py   # metrics for inference
    │   ├── tools.py    # adjust learning rate, visualization, early stopping
    │   └── utils.py    # seed setting, load model, version build, progress bar, check points, log setting
    ├── scripts    # utils
    ├── config.yaml    # configure
    │   ├── run.sh
    │   └── test.sh
    ├── main.py    # main code
    └── model.py    # model build (build, train, validation,test, inference)
   

5 directories, 18 files

```
