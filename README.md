# VTT code

- Paper : [Transformer-based multivariate time series anomaly detection using inter-variable attention mechanism](https://www.sciencedirect.com/science/article/pii/S0950705124001424?ref=pdf_download&fr=RR-2&rr=88e52cc379d63158)
- Code repo : [KBS2024-VTT](https://github.com/hwk0702/KBS2024-VTT)
- 위 코드 기반으로 작성

# Environments (environment.txt 참고)
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
- config.yaml을 통해, hyperparameter 설정 가능
- 아래 코드 실행 시, logs 폴더 안에 아래 파일들 저장됨
    - `*.pth` : best train loss의 model parameter 파일
    - `arguments.json`: argument 설정 정보
    - `model_params.json` : model hyperparameters 정보
    - `inference_result_with_metadata.h5` : test셋에 대한 웨이퍼 별 메타 데이터 및 **inference 결과**를 함께 저장한 데이터 파일
        - 이 데이터를 이용하여, `notebook/vis_test_results.ipynb` 코드로 시각화 진행 가능

```
    # For Training & Inference together
    python main.py --train --test --model VTTSAT

```

# 시각화 코드 
- `vis_test_results.ipynb`
    - 기능
        - VTT score 계산 
        - inference 결과들에 대해서 시계열 plot 시각화
        - attention difference map 시각화
    - 코드 실행을 위해, 모델 test test에 대한 inference 결과 데이터 필요(`inference_result_with_metadata.h5`) 
    - [KBS2024-VTT](https://github.com/hwk0702/KBS2024-VTT/tree/main/notebook)의 `2024_06_10_SKT_Market_Top_AI과정.ipynb` 코드 활용하여 작성됨

    
# 폴더 구조
```sh
.
├── LICENSE
├── README.md 
├── image
├── dataset_preprocessing.ipynb # For data preprocessing
├── drystrip_dataset     # 웨이퍼 데이터 파일(.parquet), score table 데이터 파일을 위치시켜야 함
├── data                 # dataset_preprocessing.ipynb를 통해 전처리된 데이터셋 저장됨
└── src    # 코드 메인 폴더
    ├── data_provider    # Wafer Dataset 클래스, dataloader 함수
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
    ├── config.yaml    # config 세팅에 사용됨
    ├── main.py    # main code
    └── model.py    # model build (build, train, validation,test, inference)
   

```
