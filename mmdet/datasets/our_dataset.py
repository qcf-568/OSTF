import mmengine
import pickle
import json
from .base_det_dataset import BaseDetDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class OurDataset(BaseDetDataset):

    METAINFO = {
       'classes': ('authentic', 'tampered'),
       'palette': [(66, 66, 86), (88, 88, 88)]
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
                instance['bbox'] = list(anns[:4])
                instance['bbox_label'] = anns[4]
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
