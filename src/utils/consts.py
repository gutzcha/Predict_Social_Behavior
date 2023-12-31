import numpy as np
import os.path as osp

area_number_to_name_map = {
    1:   {'x': -2.4, 'y': 3.18, 'z': -8.5, 'name': 'MeAD', 'description': 'MeAD description'},
    2:   {'x': -3.2, 'y': 3, 'z': -9.10, 'name': 'MePV', 'description': 'MePV medial amygdaloid nucleus, posteroventral part'},
    3:   {'x': -4, 'y': 3.6, 'z': -9.10, 'name': 'AHiAL', 'description': 'AHiAL amygdalohippocampal area, anterolateral part'},
    4:   {'x': -4, 'y': 2.6, 'z': -9.10, 'name': 'AHiPM', 'description': 'AHiPM amygdalohippocampal area, posteromedial part'},
    5:   {'x': -2.2, 'y': 3.5, 'z': -9.10, 'name': 'BMA', 'description': 'BMA Basomedial amygdaloid nucleus, anterior part'},
    6:   {'x': -3.4, 'y': 4, 'z': -9.10, 'name': 'BL', 'description': 'BL Basolateral amygdaloid nucleus posterior and ventral'},
    7:   {'x': -1, 'y': 3.8, 'z': -9.10, 'name': 'CxA', 'description': 'CxA Cortex-amygdala transition zone'},
    8:   {'x': -2, 'y': 4.8, 'z': -9.10, 'name': 'Pir ly3', 'description': 'Pir Anterior layer3 piriform cortex'},
    9:   {'x': -4, 'y': 4.2, 'z': -8.60, 'name': 'BMP', 'description': 'BMP Basomedial amygdaloid nucleus, posterior part'},
    10:  {'x': -1.4, 'y': 3, 'z': -9.10, 'name': 'LOT', 'description': 'LOT nucleus of the lateral olfactory tract'},
    11:  {'x': -3.4, 'y': 3.4, 'z': -8.60, 'name': 'MePD', 'description': 'MePD medial amygdaloid nucleus, posterodorsal part'},
    12:  {'x': -3.4, 'y': 4, 'z': -8.60, 'name': 'STIA', 'description': 'STIA bed nucleus of the stria terminalis, intraamygdaloid'},
    13:  {'x': -2.4, 'y': 3.8, 'z': -8.60, 'name': 'CeA', 'description': 'CeC/CeL/CeM Central amygdaloid nucleus,capsular part/ Lateral/ Medial division'},
    14:  {'x': -1, 'y': 3, 'z': -9.10, 'name': 'AA', 'description': 'AA/Ven/ACO anterior amygdaloid area,ventral endopiriform claustrum,anterior cortical amygdaloid area'},
    15:  {'x': -0.4, 'y': 3, 'z': -8.60, 'name': 'VP', 'description': 'VP ventral pallidum'},
    16:  {'x': -1.4, 'y': 3, 'z': -8.60, 'name': 'EA', 'description': 'EA extended amygdala'},
    17:  {'x': -1, 'y': 4, 'z': -8.10, 'name': 'IPAC', 'description': 'IPAC interstitial nucleus of the posterior limb of the anterior commisure'},
    19:  {'x': 2.4, 'y': 1.6, 'z': -6.6, 'name': 'AcbC', 'description': 'AcbC Accumbens nucleus, core'},
    52:  {'x': -4, 'y': 6, 'z': -7.6, 'name': 'PRh', 'description': 'PRh perirhinal cortex'},
    53:  {'x': -3, 'y': 5, 'z': -7.6, 'name': 'LaDL', 'description': 'LaDL lateral amygdaloid nucleus, dorsolateral part'},
    54:  {'x': -3.8, 'y': 1.4, 'z': -6.82, 'name': 'VPPC', 'description': 'VPPC ventral posterior nucleus of the thalamus, parvicellular part'},
    55:  {'x': -3, 'y': 1.4, 'z': -6.82, 'name': 'VM', 'description': 'VM ventromedial thalamic nucleus'},
    51:  {'x': -2.2, 'y': 0.8, 'z': -7.6, 'name': 'ZI', 'description': 'ZI/ZIR Zona incerta/Zona incerta Rostral/Ventral part'},
    35:  {'x': -2.2, 'y': 2, 'z': -7.6, 'name': 'PLH', 'description': 'PLH Peduncular part of Lateral hypothalamus'},
    36:  {'x': -3, 'y': 3, 'z': -6.9, 'name': 'VPL', 'description': 'VPL Ventral Posterolateral thalamic nuclei'},
    27:  {'x': -1, 'y': 4, 'z': -7.6, 'name': 'CPu', 'description': 'CPu caudate putamen (striatum)'},
    56:  {'x': -3, 'y': 3, 'z': -8.1, 'name': 'Opt', 'description': 'Opt optic tract'},
    57:  {'x': -2.4, 'y': 2.4, 'z': -8.1, 'name': 'ic', 'description': 'ic internal capsule'},
    58:  {'x': -1.4, 'y': 1.6, 'z': -8.1, 'name': 'Mfb', 'description': 'Mfb medial forebrain bundle'},
    43:  {'x': -4.6, 'y': 2, 'z': -8.4, 'name': 'cp', 'description': 'cp cerebral peduncle'},
    60:  {'x': 0, 'y': 3.6, 'z': -8.1, 'name': 'acp', 'description': 'acp anterior commissure, posterior limb'},
    100: {'x': 2, 'y': 5.6, 'z': -6.1, 'name': 'DI Layer 2 or 3', 'description': 'DI Dysgranular insular cortex Layer 2 or 3'},
    101: {'x': 3, 'y': 5, 'z': -6.1, 'name': 'AI Layer 1 ventral', 'description': 'AI Agranular Insular Cortex Layer 1 ventral'},
    102: {'x': 2, 'y': 5.6, 'z': -6.1, 'name': 'DI Layer 5 or 6', 'description': 'DI Dysgranular insular cortex Layer 5 or 6'},
    103: {'x': 1, 'y': 6.4, 'z': -6.1, 'name': 'GI Layer 5 or 6', 'description': 'GI Granular Insular cortex Layer 5 or 6'},
    104: {'x': 1, 'y': 6.4, 'z': -6.1, 'name': 'GI Layer 5', 'description': 'GI Granular insular Cortex Layer 5'},
    63:  {'x': -1, 'y': 3, 'z': -8.6, 'name': 'LDB', 'description': 'LDB Lateral nucleus of the diagonal band'},
    111: {'x': -2.4, 'y': 3.4, 'z': -8.60, 'name': 'MeD', 'description': 'MeD medial amygdaloid nucleus, Dorsal part'},
    112: {'x': -3, 'y': 4.2, 'z': -8.80, 'name': 'BMA', 'description': 'BMA Basomedial amygdaloid nucleus'},
    113: {'x': -4, 'y': 3, 'z': -9.10, 'name': 'AHiA', 'description': 'AHiA amygdalohippocampal area'}
}


area_name_to_number_map = {v['name']: {'x': v['x'], 'y': v['y'], 'z': v['z'], 'description': v['description'], 'number': k} for k, v in area_number_to_name_map.items()}


NODES = [
    'Ear_left',
    'Ear_right',
    'Nose',
    'Neck',
    'Trunk',
    'Lateral_right',
    'Lateral_left',
    'Tail_base']

NODES2IND = {n:ind for ind,n in enumerate(NODES)}

SKELETON = [
    ('Neck', 'Ear_left'),
    ('Neck', 'Ear_right'),
    ('Neck', 'Nose'),
    ('Trunk', 'Neck'),
    ('Trunk', 'Lateral_right'),
    ('Trunk', 'Lateral_left'),
    ('Trunk', 'Tail_base')
]

EDGES = np.array([(NODES2IND[e1], NODES2IND[e2]) for e1, e2 in SKELETON]).reshape(-1,2)
FILES_MAPPING_PATH = osp.join('..', 'assets', 'file_to_session_mapping_20230930-123448.xlsx')
#
# if __name__ == "__main__":
#     # DEBUG
#     import pandas as pd
#     import os
#     print(os.getcwd())
#     # df = pd.read_excel(FILES_MAPPING_PATH)
#     # print(df)