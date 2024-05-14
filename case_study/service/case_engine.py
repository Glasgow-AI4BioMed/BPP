from typing import List
from case_study.bean.bean_collection import Edge, Node, Relationship
from case_study.bean.data_comparator import DataComparator
from case_study.bean.data_version import DataWithVersion
from case_study.service.edge_service import edge_service
from case_study.service.model_engine import filter_regulator_relationships, ModelEngine
from case_study.service.node_service import node_service

from case_study.service.primary_secondary_entity_engine import PrimarySecondaryEntityEngine


class Case:
    def __init__(self, dataset_name: str, reaction: Edge, target: Node):
        self.dataset_name = dataset_name
        self.reaction = reaction
        self.target = target
        # {Model : { Indicators : Scores} }
        self.score_dict: dict[str, dict[str, float]] = dict()

    def get_score(self, model, indicator):
        return self.score_dict[model][indicator]

    def __set_score_dict_via_case_study_result_on_single_model(self, model_name: str, case_study_result: dict):
        for indicator_name, score in case_study_result.items():
            if model_name not in self.score_dict.keys():
                self.score_dict[model_name] = dict()

            self.score_dict[model_name][indicator_name] = score

    def set_score_dict(self, model_name: str, case_study_result: dict):
        self.__set_score_dict_via_case_study_result_on_single_model(model_name, case_study_result)

    def __repr__(self):
        return f"Case(dataset_name='{self.dataset_name}', reaction='{self.reaction}', target='{self.target}', score_dict={self.score_dict})"


class CaseEngine:
    def __init__(self, dataset):
        self.dataset: str = dataset
        self.cases_pool: list[Case] = list()
        self.res: dict[str, dict[str, float]] = dict()

    def add_single_case(self, dataset_name, relationship: Relationship, model_name_list: List[str],
                        case_study_result_list: List[dict]):
        case = Case(dataset_name, relationship.edge, relationship.node)

        for index, model_name in enumerate(model_name_list):
            case.set_score_dict(model_name, case_study_result_list[index])

        self.__add_case_to_pool(case)

        return case

    def __add_case_to_pool(self, case: Case):
        self.cases_pool.append(case)

    def __clean_res(self):
        self.res = dict()

    def __generate_inner_res_dict(self, model_name, indicator_name):
        if model_name not in self.res.keys():
            self.res[model_name] = dict()

        if indicator_name not in self.res[model_name].keys():
            self.res[model_name][indicator_name] = 0.0

    def __contains_ndcg_string_check(self, string):
        lowercase_string = string.lower()
        lowercase_substring = "ndcg"
        return lowercase_substring in lowercase_string

    def print_case_pool(self):
        for case in self.cases_pool:
            is_predict_out: bool = False
            for model_name, indicator_dict in case.score_dict.items():
                for indicator_name, score in indicator_dict.items():
                    if (not self.__contains_ndcg_string_check(indicator_name)) and score > 0:
                        is_predict_out = True
                        break
            if is_predict_out:
                print(str(case))
            else:
                print("Fail to predict out: " + str(case))

    def calculate(self):
        self.__clean_res()

        for case in self.cases_pool:
            for model_name, indicator_dict in case.score_dict.items():
                for indicator_name, score in indicator_dict.items():
                    self.__generate_inner_res_dict(model_name, indicator_name)
                    self.res[model_name][indicator_name] += score

        for model_name, indicator_dict in self.res.items():
            for indicator_name, score in indicator_dict.items():
                self.res[model_name][indicator_name] = self.res[model_name][indicator_name] / len(self.cases_pool)

    def print_res(self):
        print("- -" + self.dataset)
        print("- -" + "The number of cases is {}".format(len(self.cases_pool)))
        for model_name, indicator_dict in self.res.items():
            print("- - - -" + model_name)
            for indicator_name, score in self.res[model_name].items():
                # print("- - - -" + "- - - -" + indicator_name + ": " + '{.2f}'.format(score))
                print("- - - -" + "- - - -" + indicator_name + ": " + '{:.4f}'.format(float(score)))


def case_study(toplevel_pathway_name: str, relationship: Relationship, data_version: DataWithVersion, model: str):
    # print("\033[91m{0}\033[0m".format("CASE STUDY ON"))
    # print("\033[91m{0}\033[0m".format(str(relationship)))
    # print("\033[91m{0}\033[0m".format("Model = " + model))
    edge_index = relationship.edge.index
    target_node_index = relationship.node.index
    edge = edge_service.get_edge_from_dataset_based_on_index(toplevel_pathway_name, edge_index, data_version)
    node = node_service.get_node_from_dataset_based_on_index(toplevel_pathway_name, target_node_index, data_version)
    list_of_selected_edges: list[Edge] = list()
    list_of_selected_edges.append(edge)

    list_of_target_nodes: list[Node] = list()
    list_of_target_nodes.append(node)

    direction = ""
    if relationship.direction == 1:
        direction = "output"
    elif relationship.direction == -1:
        direction = "input"
    else:
        # relationship.direction == 0
        direction = "output"

    res = ModelEngine(data_version_name=data_version.data_version_name, task_name="raw",
                      toplevel_pathway_name=toplevel_pathway_name, model=model,
                      direction=direction).case_study_predict(list_of_selected_edges,
                                                              list_of_target_nodes)

    # print(res)

    return res


def case_study_on_dataset(dataset_name: str):
    """
    :param dataset_name: "Disease" /
    :return:
    """
    data_old = DataWithVersion("data")
    # data_old = DataVersion("data")
    data_new = DataWithVersion("data_version_85")
    data_comparator = DataComparator(old_data=data_old, new_data=data_new)
    new_relationships = data_comparator.choose_dataset(
        dataset_name).get_relationships_newly_added_with_node_and_edge_in_old_data()
    new_relationships = filter_regulator_relationships(new_relationships)
    num_of_relationships = len(new_relationships)
    print("Number of New Relationships in " + dataset_name + " is {0}.".format(num_of_relationships))

    dataset_name_lowercase = dataset_name.lower()

    case_engine = CaseEngine(dataset_name_lowercase)

    for relationship in new_relationships:
        # print(relationship)
        case_gcn = case_study(toplevel_pathway_name=dataset_name, relationship=relationship, data_version=data_old,
                              model="gcn")
        case_hgnn = case_study(toplevel_pathway_name=dataset_name, relationship=relationship,
                               data_version=data_old, model="hgnn")
        case_hgnnp = case_study(toplevel_pathway_name=dataset_name, relationship=relationship,
                                data_version=data_old, model="hgnnp")

        case_engine.add_single_case(dataset_name, relationship, ['gcn', 'hgnn', 'hgnnp'],
                                    [case_gcn, case_hgnn, case_hgnnp])

    case_engine.print_case_pool()

    case_engine.calculate()

    case_engine.print_res()


def case_study_on_multi_dataset(dataset_name_list: List[str]):
    """
    :param dataset_name: "Disease" /
    :return:
    """
    data_old = DataWithVersion("data")
    # data_old = DataVersion("data")
    data_new = DataWithVersion("data_version_85")
    data_comparator = DataComparator(old_data=data_old, new_data=data_new)

    new_relationships_in_multi_datasets: list[dict] = []
    case_engine_name_for_multi_datasets: str = "_".join(dataset_name_list)

    for dataset_name in dataset_name_list:
        new_relationships = data_comparator.choose_dataset(
            dataset_name).get_relationships_newly_added_with_node_and_edge_in_old_data()
        new_relationships = filter_regulator_relationships(new_relationships)
        num_of_relationships = len(new_relationships)
        print("Number of New Relationships in " + dataset_name + " is {0}.".format(num_of_relationships))
        for new_relationship in new_relationships:
            new_relationships_in_multi_datasets.append(
                {"new_relationship": new_relationship, "dataset_name": dataset_name})

    case_engine_name_lowercase = case_engine_name_for_multi_datasets.lower()

    case_engine = CaseEngine(case_engine_name_lowercase)

    for relationship_dict in new_relationships_in_multi_datasets:
        dataset_name = relationship_dict["dataset_name"]
        relationship = relationship_dict["new_relationship"]
        # print(relationship)
        case_gcn = case_study(toplevel_pathway_name=dataset_name, relationship=relationship, data_version=data_old,
                              model="gcn")
        case_hgnn = case_study(toplevel_pathway_name=dataset_name, relationship=relationship,
                               data_version=data_old, model="hgnn")
        case_hgnnp = case_study(toplevel_pathway_name=dataset_name, relationship=relationship,
                                data_version=data_old, model="hgnnp")

        case_engine.add_single_case(dataset_name, relationship, ['gcn', 'hgnn', 'hgnnp'],
                                    [case_gcn, case_hgnn, case_hgnnp])

    case_engine.print_case_pool()

    case_engine.calculate()

    case_engine.print_res()


def case_study_on_primary_and_secondary_entities(toplevel_pathway_name: str, dataset_name: str,
                                                 data_version: DataWithVersion, threshold_degree: int):
    degree2nodes_dict = node_service.get_degree2node_dict(toplevel_pathway_name, data_version)
    primary_nodes_indexes, secondary_nodes_indexes = PrimarySecondaryEntityEngine().get_primary_secondary_entities(
        degree2nodes_dict, threshold_degree)

    data_old = DataWithVersion("data")
    # data_old = DataVersion("data")
    data_new = DataWithVersion("data_version_85")
    data_comparator = DataComparator(old_data=data_old, new_data=data_new)
    new_relationships = data_comparator.choose_dataset(
        dataset_name).get_relationships_newly_added_with_node_and_edge_in_old_data()
    new_relationships = filter_regulator_relationships(new_relationships)
    num_of_relationships = len(new_relationships)
    print("Number of New Relationships in " + dataset_name + " is {0}.".format(num_of_relationships))

    dataset_name_lowercase = dataset_name.lower()

    case_engine_primary = CaseEngine(dataset_name_lowercase)
    case_engine_secondary = CaseEngine(dataset_name_lowercase)

    for relationship in new_relationships:
        # print(relationship)
        case_gcn = case_study(toplevel_pathway_name=dataset_name, relationship=relationship, data_version=data_old,
                              model="gcn")
        case_hgnn = case_study(toplevel_pathway_name=dataset_name, relationship=relationship,
                               data_version=data_old, model="hgnn")
        case_hgnnp = case_study(toplevel_pathway_name=dataset_name, relationship=relationship,
                                data_version=data_old, model="hgnnp")

        if relationship.node_index in primary_nodes_indexes:
            case_engine_primary.add_single_case(dataset_name, relationship, ['gcn', 'hgnn', 'hgnnp'],
                                                [case_gcn, case_hgnn, case_hgnnp])

        elif relationship.node_index in secondary_nodes_indexes:
            case_engine_secondary.add_single_case(dataset_name, relationship, ['gcn', 'hgnn', 'hgnnp'],
                                                  [case_gcn, case_hgnn, case_hgnnp])

    print(
        "-----------------------------case study on primary entities of " + dataset_name + " -----------------------------")
    case_engine_primary.print_case_pool()

    case_engine_primary.calculate()

    case_engine_primary.print_res()

    print(
        "-----------------------------case study on secondary entities of " + dataset_name + " -----------------------------")
    case_engine_secondary.print_case_pool()

    case_engine_secondary.calculate()

    case_engine_secondary.print_res()
