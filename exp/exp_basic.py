import os
import torch
from models import Autoformer, TimesNet, DLinear, PatchTST, iTransformer, \
    IRPA, IRPA_fft, iTransformer_IRPA, Autoformer_IRPA, TimesNet_IRPA, PatchTST_IRPA, FDER, FDER_demo, GTR, \
    RAFT, TimeMixer, TimeFilter, FEDformer, PaiFilter, TexFilter

try:
    from models import FDER_AttnFusion
except ImportError:
    FDER_AttnFusion = None

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet_IRPA': TimesNet_IRPA,
            'PatchTST_IRPA': PatchTST_IRPA,
            'iTransformer_IRPA': iTransformer_IRPA,
            'Autoformer_IRPA': Autoformer_IRPA,
            'IRPA': IRPA,
            'IRPA_fft': IRPA_fft,
            'FDER': FDER,
            'FDER_demo': FDER_demo,
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'DLinear': DLinear,
            'PatchTST': PatchTST,
            'iTransformer': iTransformer,
            'GTR': GTR,
            'RAFT': RAFT,
            'TimeMixer': TimeMixer,
            'TimeFilter': TimeFilter,
            'FEDformer': FEDformer,
            'FilterNet': PaiFilter,
            'PaiFilter': PaiFilter,
            'TexFilter': TexFilter,
        }
        if FDER_AttnFusion is not None:
            self.model_dict['FDER_AttnFusion'] = FDER_AttnFusion
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
