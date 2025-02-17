import mmengine
import pickle
import json
import numpy as np
from .base_det_dataset import BaseDetDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class PTDataset(BaseDetDataset):

    METAINFO = {
       'classes': ('authentic', 'tampered'),
       'palette': [(66, 66, 86), (88, 88, 88)]
    }

    def load_data_list(self,):
        with open(self.ann_file, 'rb') as f:
            ann_list = pickle.load(f)

        data_infos = []
        for i, (k,v) in enumerate(ann_list.items()):
            v3 = v[3]
            width = v[0]
            height = v[1]
            minsize = min(width, height)
            maxsize = max(width, height)
            r1 = (768/minsize)
            r2 = (1536/maxsize)
            rmin = max(r1,r2)
            rmin2 = (rmin**2)
            instances = []
            yxs = 0
            assert (len(v3)==len(v[2])), '%d_%d'%(len(v3), len(v[2]))
            for ai,anns in enumerate(v[2]):
                instance = {}
                ((y1,x1),(y2,x2)) = anns[3]
                bh = anns[2]
                bw = anns[1]
                ba = anns[0]
                instance['bbox'] = (y1,x1,y2,x2)
                instance['bbox_label'] = 0 # np.zeros((len(instance['bbox']),),dtype=np.uint8)
                instance['bbox_whs'] = (bh,bw,(v3[ai]*rmin2))
                instance['bbox_pts'] = anns[4]
                instances.append(instance)
                if ((bh>=16) and (bw>=16) and (ba>=256)):
                    yxs = (yxs+1)
            if yxs==0:
                continue
            data_infos.append(
                dict(
                    img_path=k,
                    msk_path=k.replace('/img/','/msk/').replace('.jpg','.png'),
                    img_id=i,
                    width=width,
                    height=height,
                    instances=instances
                ))

        return data_infos
