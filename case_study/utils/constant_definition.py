from enum import Enum


class ToplevelPathwayNameEnum(Enum):
    DISEASE = "Disease"
    IMMUNE_SYSTEM = "Immune System"
    METABOLISM = "Metabolism"
    SIGNAL_TRANSDUCTION = "Signal Transduction"


class MessageTextEnum(Enum):
    UNKNOWN = "unknown"


class ReactionDirectionEnum(Enum):
    INPUT_FLAG = -1
    OUTPUT_FLAG = 1
    REGULATOR_FLAG = 0


class FileNameEnum(Enum):
    ATTRIBUTE_STID_FILE_NAME = 'components-all.txt'
    ATTRIBUTE_NAME_FILE_NAME = 'components-all-names.txt'
    NODE_STID_FILE_NAME = 'nodes.txt'
    NODE_NAME_FILE_NAME = 'nodes-names.txt'
    EDGE_STID_FILE_NAME = 'edges.txt'
    EDGE_NAME_FILE_NAME = 'edges-names.txt'
    RELATIONSHIP_FILE_NAME = 'relationship.txt'
    PAIR_OF_NODE_AND_ATTRIBUTE_FILE_NAME = 'components-mapping.txt'


class ModelPathEnum(Enum):
    ATTRIBUTE_LEVEL_TASK = 'attribute_level_task'
    EDGE_LEVEL_TASK = 'edge_level_task'

    GCN_MODEL_NAME = 'gcn'
    HGNN_MODEL_NAME = 'hgnn'
    HGNNP_MODEL_NAME = 'hgnnp'
    MF_MODEL_NAME = 'mf'

    DISEASE_DATASET_NAME = 'Disease'
    IMMUNE_SYSTEM_DATASET_NAME = 'Immune System'
    METABOLISM_DATASET_NAME = 'Metabolism'
    SIGNAL_TRANSDUCTION_DATASET_NAME = 'Signal Transduction'

    EDGE_LEVEL_INPUT_DIRECTION = 'input'
    EDGE_LEVEL_OUTPUT_DIRECTION = 'output'

    BEST_EDGE_LEVEL_GCN_DISEASE_INPUT_MODEL = 'GCN_Disease_input link prediction dataset_0.005.bin'
    BEST_EDGE_LEVEL_GCN_DISEASE_OUTPUT_MODEL = 'GCN_Disease_output link prediction dataset_0.01.bin'

    BEST_EDGE_LEVEL_GCN_IMMUNE_SYSTEM_INPUT_MODEL = 'GCN_Immune System_input link prediction dataset_0.05.bin'
    BEST_EDGE_LEVEL_GCN_IMMUNE_SYSTEM_OUTPUT_MODEL = 'GCN_Immune System_output link prediction dataset_0.05.bin'

    BEST_EDGE_LEVEL_GCN_METABOLISM_INPUT_MODEL = 'GCN_Metabolism_input link prediction dataset_0.01.bin'
    BEST_EDGE_LEVEL_GCN_METABOLISM_OUTPUT_MODEL = 'GCN_Metabolism_output link prediction dataset_0.01.bin'

    BEST_EDGE_LEVEL_GCN_SIGNAL_TRANSDUCTION_INPUT_MODEL = 'GCN_Signal Transduction_input link prediction dataset_0.005.bin'
    BEST_EDGE_LEVEL_GCN_SIGNAL_TRANSDUCTION_OUTPUT_MODEL = 'GCN_Signal Transduction_output link prediction dataset_0.05.bin'

    BEST_EDGE_LEVEL_HGNN_DISEASE_INPUT_MODEL = 'HGNN_Disease_input link prediction dataset_0.01.bin'
    BEST_EDGE_LEVEL_HGNN_DISEASE_OUTPUT_MODEL = 'HGNN_Disease_output link prediction dataset_0.005.bin'

    BEST_EDGE_LEVEL_HGNN_IMMUNE_SYSTEM_INPUT_MODEL = 'HGNN_Immune System_input link prediction dataset_0.05.bin'
    BEST_EDGE_LEVEL_HGNN_IMMUNE_SYSTEM_OUTPUT_MODEL = 'HGNN_Immune System_output link prediction dataset_0.05.bin'

    BEST_EDGE_LEVEL_HGNN_METABOLISM_INPUT_MODEL = 'HGNN_Metabolism_input link prediction dataset_0.01.bin'
    BEST_EDGE_LEVEL_HGNN_METABOLISM_OUTPUT_MODEL = 'HGNN_Metabolism_output link prediction dataset_0.01.bin'

    BEST_EDGE_LEVEL_HGNN_SIGNAL_TRANSDUCTION_INPUT_MODEL = 'HGNN_Signal Transduction_input link prediction dataset_0.01.bin'
    BEST_EDGE_LEVEL_HGNN_SIGNAL_TRANSDUCTION_OUTPUT_MODEL = 'GCN_Signal Transduction_output link prediction dataset_0.05.bin'

    BEST_EDGE_LEVEL_HGNNP_DISEASE_INPUT_MODEL = 'HGNNP_Disease_input link prediction dataset_0.005.bin'
    BEST_EDGE_LEVEL_HGNNP_DISEASE_OUTPUT_MODEL = 'HGNNP_Disease_output link prediction dataset_0.05.bin'

    BEST_EDGE_LEVEL_HGNNP_IMMUNE_SYSTEM_INPUT_MODEL = 'HGNNP_Immune System_input link prediction dataset_0.05.bin'
    BEST_EDGE_LEVEL_HGNNP_IMMUNE_SYSTEM_OUTPUT_MODEL = 'HGNNP_Immune System_output link prediction dataset_0.05.bin'

    BEST_EDGE_LEVEL_HGNNP_METABOLISM_INPUT_MODEL = 'HGNNP_Metabolism_input link prediction dataset_0.005.bin'
    BEST_EDGE_LEVEL_HGNNP_METABOLISM_OUTPUT_MODEL = 'HGNNP_Metabolism_output link prediction dataset_0.005.bin'

    BEST_EDGE_LEVEL_HGNNP_SIGNAL_TRANSDUCTION_INPUT_MODEL = 'HGNNP_Signal Transduction_input link prediction dataset_0.01.bin'
    BEST_EDGE_LEVEL_HGNNP_SIGNAL_TRANSDUCTION_OUTPUT_MODEL = 'HGNNP_Signal Transduction_output link prediction dataset_0.05.bin'


class ModelHyperParameter(Enum):
    BEST_EDGE_LEVEL_GCN_DISEASE_INPUT_MODEL_EMB_SIZE = 256
    # todo 64
    BEST_EDGE_LEVEL_GCN_DISEASE_OUTPUT_MODEL_EMB_SIZE = 256

    BEST_EDGE_LEVEL_GCN_IMMUNE_SYSTEM_INPUT_MODEL_EMB_SIZE = 256
    BEST_EDGE_LEVEL_GCN_IMMUNE_SYSTEM_OUTPUT_MODEL_EMB_SIZE = 256

    BEST_EDGE_LEVEL_GCN_METABOLISM_INPUT_MODEL_EMB_SIZE = 256

    # todo 128
    BEST_EDGE_LEVEL_GCN_METABOLISM_OUTPUT_MODEL_EMB_SIZE = 256

    BEST_EDGE_LEVEL_GCN_SIGNAL_TRANSDUCTION_INPUT_MODEL_EMB_SIZE = 256
    BEST_EDGE_LEVEL_GCN_SIGNAL_TRANSDUCTION_OUTPUT_MODEL_EMB_SIZE = 256

    BEST_EDGE_LEVEL_HGNN_DISEASE_INPUT_MODEL_EMB_SIZE = 256
    # todo 64
    BEST_EDGE_LEVEL_HGNN_DISEASE_OUTPUT_MODEL_EMB_SIZE = 256

    BEST_EDGE_LEVEL_HGNN_IMMUNE_SYSTEM_INPUT_MODEL_EMB_SIZE = 256
    # todo 64
    BEST_EDGE_LEVEL_HGNN_IMMUNE_SYSTEM_OUTPUT_MODEL_EMB_SIZE = 256

    # todo 128
    BEST_EDGE_LEVEL_HGNN_METABOLISM_INPUT_MODEL_EMB_SIZE = 256
    # todo 128
    BEST_EDGE_LEVEL_HGNN_METABOLISM_OUTPUT_MODEL_EMB_SIZE = 256

    BEST_EDGE_LEVEL_HGNN_SIGNAL_TRANSDUCTION_INPUT_MODEL_EMB_SIZE = 256
    # todo 64
    BEST_EDGE_LEVEL_HGNN_SIGNAL_TRANSDUCTION_OUTPUT_MODEL_EMB_SIZE = 256

    BEST_EDGE_LEVEL_HGNNP_DISEASE_INPUT_MODEL_EMB_SIZE = 256
    BEST_EDGE_LEVEL_HGNNP_DISEASE_OUTPUT_MODEL_EMB_SIZE = 256

    BEST_EDGE_LEVEL_HGNNP_IMMUNE_SYSTEM_INPUT_MODEL_EMB_SIZE = 256
    # todo 128
    BEST_EDGE_LEVEL_HGNNP_IMMUNE_SYSTEM_OUTPUT_MODEL_EMB_SIZE = 256

    BEST_EDGE_LEVEL_HGNNP_METABOLISM_INPUT_MODEL_EMB_SIZE = 256
    BEST_EDGE_LEVEL_HGNNP_METABOLISM_OUTPUT_MODEL_EMB_SIZE = 256

    BEST_EDGE_LEVEL_HGNNP_SIGNAL_TRANSDUCTION_INPUT_MODEL_EMB_SIZE = 256
    BEST_EDGE_LEVEL_HGNNP_SIGNAL_TRANSDUCTION_OUTPUT_MODEL_EMB_SIZE = 256


