from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS

# @DATASETS.register_module()
# class NWPUInsSegDataset(CocoDataset):
#     METAINFO = {
#         'classes': ['airplane', 'ship', 'storage_tank', 'baseball_diamond',
#                     'tennis_court', 'basketball_court', 'ground_track_field',
#                     'harbor', 'bridge', 'vehicle'],
#         'palette': [(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
#                     (0, 60, 100), (0, 80, 100), (0, 0, 230),
#                     (119, 11, 32), (0, 255, 0), (0, 0, 255)]
#     }

@DATASETS.register_module()
class NWPUInsSegDataset(CocoDataset):
    METAINFO = {
        'classes': [
            '_background_', 'Calanus sinicus', 'Sagitta crassa', 'Themisto gracilipes', 'Penilia avirostris', "Centropages abdominalis", "Acartia pacifica",
            "Centropages tenuiremis", "Pontellopsis tenuicauda", "Calanopia thompsoni", "Sugiura chengshanense", "Ophioplutues larva early", "Eirene menoni", 
            "Euphausia pacifica", "Evadne tergestina", "Muggiaea atlantica", "Paracalanus parvus", "Oithona plumifera", "Pleurobrachia globosa", "Clytia folleata", 
            "Obelia dichotoma", "Ectopleura bimanatus", "Doliolum denticulatum", "Oikopleura longicauda", "Tornaria larva", "Polychaeta larva early", 
            "Polychaeta larva later", "Turritopsis nutricula", "Proboscidactyla flavicirrata", "Fritillaria formica", "Labidocera rotunda", "Alima larva", 
            "Megalopa larva", "Brachyura zoea larva", "Ophioplutues larva later", "Fish eggs", "Fish larva", "Actinotrocha larva", "Trochophora larva", ],
        'palette': [
            (0, 0, 0), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30), 
            (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), 
            (120, 166, 157), (110, 76, 0), (174, 57, 255), (197, 226, 255), (72, 0, 118), (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182), 
            (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255), (134, 134, 103), 
            (145, 148, 174), (255, 208, 186)]
    }



@DATASETS.register_module()
class WHUInsSegDataset(CocoDataset):
    METAINFO = {
        'classes': ['building'],
        'palette': [(0, 255, 0)]
    }


@DATASETS.register_module()
class SSDDInsSegDataset(CocoDataset):
    METAINFO = {
        'classes': ['ship'],
        'palette': [(0, 0, 255)]
    }