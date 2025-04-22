import argparse
import random
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_minigptv2

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args_list = ["--cfg-path", "eval_configs/minigptv2_eval.yaml", "--gpu-id", 0]
    args = parser.parse_args(args_list)
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

# Model Initialization
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))

img_list = []
conv = CONV_VISION_minigptv2.copy()

# upload image
img_path = "/home/isarlab/PycharmProjects/miniGPT-4/examples_v2/office.jpg"
start = time.time()
llm_message = chat.upload_img(img_path, conv, img_list)
# print(llm_message)
print('Uploading image time:', time.time()-start)

# ask question
start1 = time.time()
question = 'describe the image as detailed as possible in one short sentence'
chat.ask(question, conv)
print('Asking time:', time.time()-start1)
start2 = time.time()
chat.encode_img(img_list)  # after this we have img_list[0] that is the embedding torch.Size([1, 256, 4096]),
# where 256 = 1024 / 4 i.e. they group 4 visual patches into one token, which is 4096-long
print('Encoding time:', time.time()-start2)

# get answer
start3 = time.time()
answer, _, output_emb = chat.answer(conv, img_list, return_out_emb=True)
print('Answering time:', time.time()-start3)
print('Total elapsed time:', time.time()-start)
print(answer)
input_llm = img_list[0].squeeze(0)
print('Last Hidden State:', output_emb.shape)
'''
OUTPUTS NEEDED FOR DATASET CONSTRUCTION
1. Embedding post ViT-Linear -> input_llm, [M, 4096]
2. Embedding pre tokenizer ma post decoder layers di Llama -> output_emb (ultimo hidden state, [N, 4096])
3. Testo diretto da passare al tokenizzatore di SD-Turbo cosi da non cambiare l'architettura di SD-Turbo -> answer 

N.B. M è la dimensione del numero di token visuali, i.e M = 4xP dove P è una patch 14x14 pixel dell'immagine 
resizata a 448x448, ossia 32x32 patch nell'immagine = 1024, che diviso 4 fa appunto 256 = M
N.B. N è la dimensione dei token di uscita dal LLM, che poi saranno l'input del Tokenizer
'''
