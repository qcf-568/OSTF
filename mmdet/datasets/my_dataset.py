import mmengine
import pickle
import json
from .base_det_dataset import BaseDetDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class MyDataset(BaseDetDataset):

    METAINFO = {
       'classes': ('tampered'),
        'palette': [(220, 20, 60)]
    }

    def load_data_list(self,):
        with open(self.ann_file, 'rb') as f:
            ann_list = pickle.load(f)

        data_infos = []
        for i, (k,v) in enumerate(ann_list.items()):
            
            width = v['w']
            height = v['h']
            instances = []

            for anns in v['b']:
                instance = {}
                instance['bbox'] = list(anns)
                instance['bbox_label'] = 0
                instances.append(instance)

            data_infos.append(
                dict(
                    img_path=k,
                    img_id=i,
                    width=width,
                    height=height,
                    instances=instances
                ))

        return data_infos
