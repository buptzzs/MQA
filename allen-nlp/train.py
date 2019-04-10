from model import MultiStepParaRankModel
from self_attentive import SelfAttentive
from qangaroo import MyQangarooReader
from decoder import Decoder, SANDecoder
from allennlp.commands import train
import argparse

parser = argparse.ArgumentParser(description='Train By AllenNlp Trainer')

parser.add_argument('--param', '-p',type=str, default='./model.jsonnet')
parser.add_argument('--save_dir', '-s',type=str, default='./checkpoint')

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    train.train_model_from_file(args.param, args.save_dir, force=True)