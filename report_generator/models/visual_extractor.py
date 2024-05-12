import torch
import torch.nn as nn
import torchvision.models as models


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.args = args
        print(f"=> creating model '{args.visual_extractor}'")
        if 'resnet' in args.visual_extractor:
            self.visual_extractor = args.visual_extractor
            self.pretrained = args.visual_extractor_pretrained
            model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
            modules = list(model.children())[:-2]
            self.model = nn.Sequential(*modules)
            self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
            self.classifier = nn.Linear(2048, args.num_labels)

        else:
            raise NotImplementedError

        # load pretrained visual extractor
        if args.pretrain_cnn_file and args.pretrain_cnn_file != "":
            print(f'Load pretrained CNN model from: {args.pretrain_cnn_file}')
            checkpoint = torch.load(args.pretrain_cnn_file, map_location='cuda:{}'.format(args.gpu))
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            print(f'Load pretrained CNN model from: official pretrained in ImageNet')

    def forward(self, images):
        if 'resnet' in self.visual_extractor:
            patch_feats = self.model(images)
            avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
            labels = self.classifier(avg_feats)
        else:
            raise NotImplementedError
        batch_size, feat_size, _, _ = patch_feats.shape
        # print("patch feats shape", patch_feats.shape)
        # print("avg_feats shape", avg_feats.shape)

        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        # print("patch feats  aftershape", patch_feats.shape)
        return patch_feats, avg_feats, labels
