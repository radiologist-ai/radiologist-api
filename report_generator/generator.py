from argparse import Namespace

import torch

from .models import LAMRGModel_v12, Tokenizer


args = Namespace(
    image_dir='data/mimic/r2gen/images/images224/files',
    ann_path='report_generator/annotation.json',
    label_path='data/mimic/finding/chexpert_labeled.csv',
    image_size=256,
    crop_size=224,
    dataset_name='mimic_cxr',
    max_seq_length=80,
    threshold=3,
    num_workers=4,
    batch_size=16,
    version='12',
    visual_extractor='resnet101',
    visual_extractor_pretrained=True,
    pretrain_cnn_file=None,
    d_model=512,
    d_ff=512,
    d_vf=2048,
    num_heads=8,
    num_memory_heads=8,
    num_layers=3,
    num_labels=14,
    dropout=0.1,
    logit_layers=1,
    bos_idx=0,
    eos_idx=0,
    pad_idx=0,
    use_bn=0,
    drop_prob_lm=0.1,
    sample_method='beam_search',
    beam_size=3,
    temperature=1.0,
    sample_n=1,
    group_size=1,
    output_logsoftmax=1,
    decoding_constraint=0,
    block_trigrams=1,
    n_gpu=1,
    epochs=30,
    save_dir='results\\mimic_cxr\\V12_resnet101_labelloss_rankloss_10th_try_balanced_train_20240329-235144',
    expe_name='10th_try_balanced_train',
    record_dir='records/',
    save_period=1,
    monitor_mode='max',
    monitor_metric='BLEU_4',
    early_stop=50,
    label_smoothing=0.1,
    alpha=0.5,
    grad_clip=5,
    test_steps=500,
    label_loss=True,
    rank_loss=True,
    optim='Adam',
    lr_ve=5e-05,
    lr_ed=0.0001,
    weight_decay=5e-05,
    amsgrad=True,
    lr_scheduler='StepLR',
    step_size=1,
    gamma=0.8,
    seed=456789,
    gpu='0',
    resume=None,
    num_slots=60,
    rm_num_heads=8,
    rm_d_model=512,
    cfg='config/mimic_resnet.yml',
    set_cfgs=[]
)


class ReportGenerator:
    def __init__(self, model_path: str):
        self.args = args
        self.model_path = model_path
        self.tokenizer = Tokenizer(self.args)
        self.model = LAMRGModel_v12(self.args, self.tokenizer)
        print(self.model.load_state_dict(torch.load(self.model_path)['state_dict']))
        self.model.eval()
#     TODO methods for report generation

