import os

from case_study.utils.constant_definition import ModelPathEnum, ModelHyperParameter


class ModelSelector:
    def __init__(self):
        self.__path_dict: dict[str, dict] = {
            ModelPathEnum.EDGE_LEVEL_TASK.value: {
                ModelPathEnum.GCN_MODEL_NAME.value: {
                    ModelPathEnum.DISEASE_DATASET_NAME.value: {
                        ModelPathEnum.EDGE_LEVEL_INPUT_DIRECTION.value: ModelPathEnum.BEST_EDGE_LEVEL_GCN_DISEASE_INPUT_MODEL.value,
                        ModelPathEnum.EDGE_LEVEL_OUTPUT_DIRECTION.value: ModelPathEnum.BEST_EDGE_LEVEL_GCN_DISEASE_OUTPUT_MODEL.value
                    },
                    ModelPathEnum.IMMUNE_SYSTEM_DATASET_NAME.value: {
                        ModelPathEnum.EDGE_LEVEL_INPUT_DIRECTION.value: ModelPathEnum.BEST_EDGE_LEVEL_GCN_IMMUNE_SYSTEM_INPUT_MODEL.value,
                        ModelPathEnum.EDGE_LEVEL_OUTPUT_DIRECTION.value: ModelPathEnum.BEST_EDGE_LEVEL_GCN_IMMUNE_SYSTEM_OUTPUT_MODEL.value
                    },
                    ModelPathEnum.METABOLISM_DATASET_NAME.value: {
                        ModelPathEnum.EDGE_LEVEL_INPUT_DIRECTION.value: ModelPathEnum.BEST_EDGE_LEVEL_GCN_METABOLISM_INPUT_MODEL.value,
                        ModelPathEnum.EDGE_LEVEL_OUTPUT_DIRECTION.value: ModelPathEnum.BEST_EDGE_LEVEL_GCN_METABOLISM_OUTPUT_MODEL.value
                    },
                    ModelPathEnum.SIGNAL_TRANSDUCTION_DATASET_NAME.value: {
                        ModelPathEnum.EDGE_LEVEL_INPUT_DIRECTION.value: ModelPathEnum.BEST_EDGE_LEVEL_GCN_SIGNAL_TRANSDUCTION_INPUT_MODEL.value,
                        ModelPathEnum.EDGE_LEVEL_OUTPUT_DIRECTION.value: ModelPathEnum.BEST_EDGE_LEVEL_GCN_SIGNAL_TRANSDUCTION_OUTPUT_MODEL.value
                    }
                },
                ModelPathEnum.HGNN_MODEL_NAME.value: {
                    ModelPathEnum.DISEASE_DATASET_NAME.value: {
                        ModelPathEnum.EDGE_LEVEL_INPUT_DIRECTION.value: ModelPathEnum.BEST_EDGE_LEVEL_HGNN_DISEASE_INPUT_MODEL.value,
                        ModelPathEnum.EDGE_LEVEL_OUTPUT_DIRECTION.value: ModelPathEnum.BEST_EDGE_LEVEL_HGNN_DISEASE_OUTPUT_MODEL.value
                    },
                    ModelPathEnum.IMMUNE_SYSTEM_DATASET_NAME.value: {
                        ModelPathEnum.EDGE_LEVEL_INPUT_DIRECTION.value: ModelPathEnum.BEST_EDGE_LEVEL_HGNN_IMMUNE_SYSTEM_INPUT_MODEL.value,
                        ModelPathEnum.EDGE_LEVEL_OUTPUT_DIRECTION.value: ModelPathEnum.BEST_EDGE_LEVEL_HGNN_IMMUNE_SYSTEM_OUTPUT_MODEL.value
                    },
                    ModelPathEnum.METABOLISM_DATASET_NAME.value: {
                        ModelPathEnum.EDGE_LEVEL_INPUT_DIRECTION.value: ModelPathEnum.BEST_EDGE_LEVEL_HGNN_METABOLISM_INPUT_MODEL.value,
                        ModelPathEnum.EDGE_LEVEL_OUTPUT_DIRECTION.value: ModelPathEnum.BEST_EDGE_LEVEL_HGNN_METABOLISM_OUTPUT_MODEL.value
                    },
                    ModelPathEnum.SIGNAL_TRANSDUCTION_DATASET_NAME.value: {
                        ModelPathEnum.EDGE_LEVEL_INPUT_DIRECTION.value: ModelPathEnum.BEST_EDGE_LEVEL_HGNN_SIGNAL_TRANSDUCTION_INPUT_MODEL.value,
                        ModelPathEnum.EDGE_LEVEL_OUTPUT_DIRECTION.value: ModelPathEnum.BEST_EDGE_LEVEL_HGNN_SIGNAL_TRANSDUCTION_OUTPUT_MODEL.value
                    }
                },
                ModelPathEnum.HGNNP_MODEL_NAME.value: {
                    ModelPathEnum.DISEASE_DATASET_NAME.value: {
                        ModelPathEnum.EDGE_LEVEL_INPUT_DIRECTION.value: ModelPathEnum.BEST_EDGE_LEVEL_HGNNP_DISEASE_INPUT_MODEL.value,
                        ModelPathEnum.EDGE_LEVEL_OUTPUT_DIRECTION.value: ModelPathEnum.BEST_EDGE_LEVEL_HGNNP_DISEASE_OUTPUT_MODEL.value
                    },
                    ModelPathEnum.IMMUNE_SYSTEM_DATASET_NAME.value: {
                        ModelPathEnum.EDGE_LEVEL_INPUT_DIRECTION.value: ModelPathEnum.BEST_EDGE_LEVEL_HGNNP_IMMUNE_SYSTEM_INPUT_MODEL.value,
                        ModelPathEnum.EDGE_LEVEL_OUTPUT_DIRECTION.value: ModelPathEnum.BEST_EDGE_LEVEL_HGNNP_IMMUNE_SYSTEM_OUTPUT_MODEL.value
                    },
                    ModelPathEnum.METABOLISM_DATASET_NAME.value: {
                        ModelPathEnum.EDGE_LEVEL_INPUT_DIRECTION.value: ModelPathEnum.BEST_EDGE_LEVEL_HGNNP_METABOLISM_INPUT_MODEL.value,
                        ModelPathEnum.EDGE_LEVEL_OUTPUT_DIRECTION.value: ModelPathEnum.BEST_EDGE_LEVEL_HGNNP_METABOLISM_OUTPUT_MODEL.value
                    },
                    ModelPathEnum.SIGNAL_TRANSDUCTION_DATASET_NAME.value: {
                        ModelPathEnum.EDGE_LEVEL_INPUT_DIRECTION.value: ModelPathEnum.BEST_EDGE_LEVEL_HGNNP_SIGNAL_TRANSDUCTION_INPUT_MODEL.value,
                        ModelPathEnum.EDGE_LEVEL_OUTPUT_DIRECTION.value: ModelPathEnum.BEST_EDGE_LEVEL_HGNNP_SIGNAL_TRANSDUCTION_OUTPUT_MODEL.value
                    }
                }
            }
        }

        self.__hyper_parameter_dict: dict[str, dict] = {
            ModelPathEnum.EDGE_LEVEL_TASK.value: {
                ModelPathEnum.GCN_MODEL_NAME.value: {
                    ModelPathEnum.DISEASE_DATASET_NAME.value: {
                        ModelPathEnum.EDGE_LEVEL_INPUT_DIRECTION.value: ModelHyperParameter.BEST_EDGE_LEVEL_GCN_DISEASE_INPUT_MODEL_EMB_SIZE.value,
                        ModelPathEnum.EDGE_LEVEL_OUTPUT_DIRECTION.value: ModelHyperParameter.BEST_EDGE_LEVEL_GCN_DISEASE_OUTPUT_MODEL_EMB_SIZE.value
                    },
                    ModelPathEnum.IMMUNE_SYSTEM_DATASET_NAME.value: {
                        ModelPathEnum.EDGE_LEVEL_INPUT_DIRECTION.value: ModelHyperParameter.BEST_EDGE_LEVEL_GCN_IMMUNE_SYSTEM_INPUT_MODEL_EMB_SIZE.value,
                        ModelPathEnum.EDGE_LEVEL_OUTPUT_DIRECTION.value: ModelHyperParameter.BEST_EDGE_LEVEL_GCN_IMMUNE_SYSTEM_OUTPUT_MODEL_EMB_SIZE.value
                    },
                    ModelPathEnum.METABOLISM_DATASET_NAME.value: {
                        ModelPathEnum.EDGE_LEVEL_INPUT_DIRECTION.value: ModelHyperParameter.BEST_EDGE_LEVEL_GCN_METABOLISM_INPUT_MODEL_EMB_SIZE.value,
                        ModelPathEnum.EDGE_LEVEL_OUTPUT_DIRECTION.value: ModelHyperParameter.BEST_EDGE_LEVEL_GCN_METABOLISM_OUTPUT_MODEL_EMB_SIZE.value
                    },
                    ModelPathEnum.SIGNAL_TRANSDUCTION_DATASET_NAME.value: {
                        ModelPathEnum.EDGE_LEVEL_INPUT_DIRECTION.value: ModelHyperParameter.BEST_EDGE_LEVEL_GCN_SIGNAL_TRANSDUCTION_INPUT_MODEL_EMB_SIZE.value,
                        ModelPathEnum.EDGE_LEVEL_OUTPUT_DIRECTION.value: ModelHyperParameter.BEST_EDGE_LEVEL_GCN_SIGNAL_TRANSDUCTION_OUTPUT_MODEL_EMB_SIZE.value
                    }
                },
                ModelPathEnum.HGNN_MODEL_NAME.value: {
                    ModelPathEnum.DISEASE_DATASET_NAME.value: {
                        ModelPathEnum.EDGE_LEVEL_INPUT_DIRECTION.value: ModelHyperParameter.BEST_EDGE_LEVEL_HGNN_DISEASE_INPUT_MODEL_EMB_SIZE.value,
                        ModelPathEnum.EDGE_LEVEL_OUTPUT_DIRECTION.value: ModelHyperParameter.BEST_EDGE_LEVEL_HGNN_DISEASE_OUTPUT_MODEL_EMB_SIZE.value
                    },
                    ModelPathEnum.IMMUNE_SYSTEM_DATASET_NAME.value: {
                        ModelPathEnum.EDGE_LEVEL_INPUT_DIRECTION.value: ModelHyperParameter.BEST_EDGE_LEVEL_HGNN_IMMUNE_SYSTEM_INPUT_MODEL_EMB_SIZE.value,
                        ModelPathEnum.EDGE_LEVEL_OUTPUT_DIRECTION.value: ModelHyperParameter.BEST_EDGE_LEVEL_HGNN_IMMUNE_SYSTEM_OUTPUT_MODEL_EMB_SIZE.value
                    },
                    ModelPathEnum.METABOLISM_DATASET_NAME.value: {
                        ModelPathEnum.EDGE_LEVEL_INPUT_DIRECTION.value: ModelHyperParameter.BEST_EDGE_LEVEL_HGNN_METABOLISM_INPUT_MODEL_EMB_SIZE.value,
                        ModelPathEnum.EDGE_LEVEL_OUTPUT_DIRECTION.value: ModelHyperParameter.BEST_EDGE_LEVEL_HGNN_METABOLISM_OUTPUT_MODEL_EMB_SIZE.value
                    },
                    ModelPathEnum.SIGNAL_TRANSDUCTION_DATASET_NAME.value: {
                        ModelPathEnum.EDGE_LEVEL_INPUT_DIRECTION.value: ModelHyperParameter.BEST_EDGE_LEVEL_HGNN_SIGNAL_TRANSDUCTION_INPUT_MODEL_EMB_SIZE.value,
                        ModelPathEnum.EDGE_LEVEL_OUTPUT_DIRECTION.value: ModelHyperParameter.BEST_EDGE_LEVEL_HGNN_SIGNAL_TRANSDUCTION_OUTPUT_MODEL_EMB_SIZE.value
                    }
                },
                ModelPathEnum.HGNNP_MODEL_NAME.value: {
                    ModelPathEnum.DISEASE_DATASET_NAME.value: {
                        ModelPathEnum.EDGE_LEVEL_INPUT_DIRECTION.value: ModelHyperParameter.BEST_EDGE_LEVEL_HGNNP_DISEASE_INPUT_MODEL_EMB_SIZE.value,
                        ModelPathEnum.EDGE_LEVEL_OUTPUT_DIRECTION.value: ModelHyperParameter.BEST_EDGE_LEVEL_HGNNP_DISEASE_OUTPUT_MODEL_EMB_SIZE.value
                    },
                    ModelPathEnum.IMMUNE_SYSTEM_DATASET_NAME.value: {
                        ModelPathEnum.EDGE_LEVEL_INPUT_DIRECTION.value: ModelHyperParameter.BEST_EDGE_LEVEL_HGNNP_IMMUNE_SYSTEM_INPUT_MODEL_EMB_SIZE.value,
                        ModelPathEnum.EDGE_LEVEL_OUTPUT_DIRECTION.value: ModelHyperParameter.BEST_EDGE_LEVEL_HGNNP_IMMUNE_SYSTEM_OUTPUT_MODEL_EMB_SIZE.value
                    },
                    ModelPathEnum.METABOLISM_DATASET_NAME.value: {
                        ModelPathEnum.EDGE_LEVEL_INPUT_DIRECTION.value: ModelHyperParameter.BEST_EDGE_LEVEL_HGNNP_METABOLISM_INPUT_MODEL_EMB_SIZE.value,
                        ModelPathEnum.EDGE_LEVEL_OUTPUT_DIRECTION.value: ModelHyperParameter.BEST_EDGE_LEVEL_HGNNP_METABOLISM_OUTPUT_MODEL_EMB_SIZE.value
                    },
                    ModelPathEnum.SIGNAL_TRANSDUCTION_DATASET_NAME.value: {
                        ModelPathEnum.EDGE_LEVEL_INPUT_DIRECTION.value: ModelHyperParameter.BEST_EDGE_LEVEL_HGNNP_SIGNAL_TRANSDUCTION_INPUT_MODEL_EMB_SIZE.value,
                        ModelPathEnum.EDGE_LEVEL_OUTPUT_DIRECTION.value: ModelHyperParameter.BEST_EDGE_LEVEL_HGNNP_SIGNAL_TRANSDUCTION_OUTPUT_MODEL_EMB_SIZE.value
                    }
                }
            }
        }

    def select_model_path(self, edge_or_node_level_task_name: str, model_name: str, toplevel_pathway_name: str,
                          direction: str) -> str:

        # os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
        model_path_root: str = os.path.join(
            os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'static', 'models')
        if edge_or_node_level_task_name in self.__path_dict.keys():
            if model_name in self.__path_dict[edge_or_node_level_task_name].keys():
                if toplevel_pathway_name in self.__path_dict[edge_or_node_level_task_name][model_name].keys():
                    if direction in self.__path_dict[edge_or_node_level_task_name][model_name][
                        toplevel_pathway_name].keys():
                        filename: str = \
                            self.__path_dict[edge_or_node_level_task_name][model_name][toplevel_pathway_name][direction]
                        model_path = os.path.join(model_path_root, filename)
                        return model_path
        return ''

    def select_model_hyper_parameter_emb_size(self, edge_or_node_level_task_name: str, model_name: str,
                                              toplevel_pathway_name: str, direction: str) -> int:
        if edge_or_node_level_task_name in self.__hyper_parameter_dict.keys():
            if model_name in self.__hyper_parameter_dict[edge_or_node_level_task_name].keys():
                if toplevel_pathway_name in self.__hyper_parameter_dict[edge_or_node_level_task_name][model_name].keys():
                    if direction in self.__hyper_parameter_dict[edge_or_node_level_task_name][model_name][
                        toplevel_pathway_name].keys():
                        hyper_parameter_emb_size: int = self.__hyper_parameter_dict[edge_or_node_level_task_name][model_name][toplevel_pathway_name][direction]
                        return hyper_parameter_emb_size

        return 64

