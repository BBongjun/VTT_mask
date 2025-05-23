import torch
from omegaconf import OmegaConf
import yaml
from datetime import datetime
import warnings
import argparse
import json
import pdb
import os

from utils.utils import set_seed, log_setting, version_build
from data_provider.waferdataset import get_dataloader
from model import build_model

warnings.filterwarnings('ignore')


def main(sweep_config=None):
    """
    main experiment

    Parameters
    ----------
    sweep_config : dict
    """
    # arg parser
    model_name = args.model
    dataset = args.dataname

    # process init
    now = datetime.now()
    group_name = f'{dataset}-{model_name}'
    process_name = f'{group_name}-{now.strftime("%Y%m%d_%H%m%S")}'

    # savedir
    logdir = os.path.join(config['log_dir'], f'{dataset}/{model_name}')
    savedir = version_build(logdir=logdir, is_train=args.train, resume=args.resume)
    logger = log_setting(savedir, f'{now.strftime("%Y%m%d_%H%m%S")}')

    # save arguments
    json.dump(vars(args), open(os.path.join(savedir, 'arguments.json'), 'w'), indent=4)

    # multiple GPU init
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # set seed
    set_seed(args.seed)

    model_params = config[model_name]
    scale = config['scale']

    # save model parameters
    json.dump(dict(model_params), open(os.path.join(savedir, 'model_params.json'), 'w'), indent=4)

    # model type init
    if model_name in ('VTTSAT', 'VTTPAT'):
        model_type = 'reconstruction'
    elif model_name in (''):
        model_type = 'prediction'
    else:
        model_type = 'classification'

    logger.info(f'Process {process_name} start!')

    data_info = OmegaConf.create(config[args.dataname])

    # load data
    trainloader, validloader, testloader = get_dataloader(data_info = data_info,
                                                          loader_params = config['loader_params'])        

    # model build
    model = build_model(args, model_params, savedir)
    logger.info('Model build success!!')

    # training
    if args.train:
        logger.info('Training Start!')
        history = model.train(trainloader, validloader, testloader, use_val=config['loader_params']['use_val'])
        for i in range(len(history['train_loss'])):
            train_loss = history['train_loss'][i]
            if config['loader_params']['use_val']:
                valid_loss = history['validation_loss'][i]
                # precisio n= history['precision'][i]
                # recall = history['recall'][i]
                # roc_auc = history['roc_auc'][i]
                # f1 = history['f1'][i]
                logger.info(f"Epoch: {i + 1} Train Loss: {train_loss:.7f} Vali Loss: {valid_loss:.7f} ")
            else:
                logger.info(f"Epoch: {i + 1} Train Loss: {train_loss:.7f}")
                        # f"precision: {precision:.4f} recall: {recall:.4f} f1: {f1:.4f} ROC_AUC: {roc_auc:.4f}")
        logger.info('Model training success!!')
        

    # test
    if args.test:
        dist = model.inference_unlabeled(testloader)
        # history = model.test(validloader, testloader)
        # for i in range(len(history['precision'])):
        #     precision = history['precision'][i]
        #     recall = history['recall'][i]
        #     roc_auc = history['roc_auc'][i]
        #     f1 = history['f1'][i]
            # logger.info(f"Result -> precision: {precision:.4f} recall: {recall:.4f} f1: {f1:.4f} ROC_AUC: {roc_auc:.4f}")

        logger.info('Model test success!!')

    torch.cuda.empty_cache()


if __name__ == '__main__':
    # set arguments
    parser = argparse.ArgumentParser()

    # 모드 선택
    parser.add_argument('--train', action='store_true', help='training model')
    parser.add_argument('--test', action='store_true', help='anomaly scoring')
    parser.add_argument('--resume', type=int, default=None, help='version number to re-train or test model')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['VTTSAT', 'VTTPAT'],
                        help="model (VTTSAT or VTTPAT)")

    # 데이터 설정
    parser.add_argument('--dataname', type=str, default='vtt_all_step', help='wafer dataset name')
    parser.add_argument('--window_size', type=int, default=350, help='window size for data loader')
    parser.add_argument('--slide_size', type=int, default=1, help='slide step size for data loader')

    # 학습 설정
    parser.add_argument('--epochs', type=int, default=2000, help='number of epochs')
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for dataloader')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'mae'],
                        help='select loss function')

    # 실행 환경
    parser.add_argument('--seed', type=int, default=72, help="set randomness")
    parser.add_argument('--use_gpu', type=bool, default=True, help='use GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU number')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple GPUs')
    parser.add_argument('--devices', type=str, default='0,1', help='device IDs of multiple GPUs')
    parser.add_argument('--configure', type=str, default='config.yaml', help='YAML configuration file')

    args = parser.parse_args()

    # Load YAML configuration
    with open(args.configure) as f:
        config = OmegaConf.load(f)
    
    # 실행
    main()

