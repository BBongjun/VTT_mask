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

## ⚙️ 사용법 요약

1. `drystrip_dataset/` 폴더 안에 아래 두 파일을 위치시키기
   - `P_FG_60S_1.parquet` (웨이퍼 시계열 데이터)
   - `score_results.parquet` (score table)

2. 데이터 전처리:
   - `notebook/dataset_preprocessing.ipynb` 실행 → `data/` 폴더에 전처리된 데이터가 저장됨

3. 모델 학습/테스트:

```bash
    # 학습 및 테스트 한 번에 실행
    python main.py --train --test --model VTTSAT
```

- config.yaml을 통해, 데이터 파일 경로 설정, hyperparameter 설정 가능
- 아래 코드 실행 시, `src/logs/` 폴더 안에 아래 파일들 저장됨
    - `*.pth` : best train loss의 model parameter 파일
    - `arguments.json`: argument 설정 정보
    - `model_params.json` : model hyperparameters 정보
    - `inference_result_with_metadata.h5` : test셋에 대한 웨이퍼 별 메타 데이터 및 **inference 결과**를 함께 저장한 데이터 파일
        - 이 데이터를 이용하여, `notebook/vis_test_results.ipynb` 코드로 시각화 진행 가능

# 시각화 코드 
- `notebook/vis_test_results.ipynb`
    - 기능
        - inference 결과들에 대해서 시계열 plot 시각화
        - VTT score 기반 가장 높은/낮은 웨이퍼들을 활용하여 시계열 및 attention difference map 시각화
    - 코드 실행을 위해, test셋에 대한 inference 결과 데이터 필요(`./logs/*/inference_result_with_metadata.h5`) 
    - [KBS2024-VTT](https://github.com/hwk0702/KBS2024-VTT/tree/main/notebook)의 `2024_06_10_SKT_Market_Top_AI과정.ipynb` 코드 기반으로 작성됨

    
# 폴더 구조
```sh

├── notebook/                   
    ├── dataset_preprocessing.ipynb # 원본 웨이퍼 데이터 전처리 jupyter notebook
    └── vis_test_results.ipynb  # inference 결과를 활용하여 시계열 시각화 및 attetnion difference map 시각화            
├── drystrip_dataset/           # 웨이퍼 데이터(.parquet), score table(.parquet) 위치
├── data/                       # 전처리된 데이터 저장 경로
└── src/                        # 메인 코드 디렉토리
    ├── config.yaml             # 실험 하이퍼파라미터 설정
    ├── main.py                 # 메인 코드 (train/test 선택 가능)
    ├── model.py                # 모델 build : train/test/inference 함수 정의
    ├── data_provider/
    │ └── dataset.py            # WaferDataset 정의 및 DataLoader 함수
    ├── layers/                 # layers for models (attention, embedding, etc.)
    │ ├── Attention.py          # Variable/Temporal Attention 모듈
    │ ├── Embed.py              
    │ └── Transformer_Enc.py    
    ├── models/
    │ ├── VTTPAT.py             
    │ └── VTTSAT.py 
    ├── utils/
    │ ├── metrics.py            # 평가 지표 계산 함수, 현재 프로젝트에서는 사용 X
    │ ├── tools.py              # 학습에 사용되는 learning rate 스케쥴러, early stopping 등 함수
    │ └── utils.py              # seed 설정, logging, 체크포인트 저장 등
```
