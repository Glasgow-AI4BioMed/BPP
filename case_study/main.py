from case_study.bean.data_version import DataWithVersion
from case_study.service.case_engine import case_study_on_dataset, case_study_on_multi_dataset
from case_study.service.model_engine import ModelEngine

from itertools import product, groupby


def create_table():
    import matplotlib.pyplot as plt

    # 数据
    categories = ['Disease', 'Metabolism', 'Immune System', 'Signal Transduction']
    thresholds = [6, 7, 6, 7]
    primary_percentages = [98.11, 94.21, 98.28, 98.17]
    secondary_percentages = [1.89, 5.79, 1.72, 1.83]

    # 创建新的figure和axes对象
    fig, ax = plt.subplots()

    # 设置标题
    ax.set_title("Entity Percentages by Category")

    # 设置不显示axes内容
    ax.axis('off')

    # 数据和列名
    columns = ['Category', 'Degree Threshold', 'Primary Entities (%)', 'Secondary Entities (%)']
    cell_data = [categories, thresholds, primary_percentages, secondary_percentages]
    table_data = list(zip(*cell_data))

    # 在axes上添加表格
    table = ax.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center', colColours=['#f5f5f5'] * 4)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # 设置表格大小

    plt.show()


def evaluate_on_primary_entities_and_secondary_entities_sweep(data_version: DataWithVersion, toplevel_pathway_name: str,
                                                              threshold_degree: int):
    task_name_group: list[str] = ["input link prediction dataset", "output link prediction dataset"]
    model_group: list[str] = ["gcn", "hgnn", "hgnnp"]
    direction_dict: dict[str, str] = {"input link prediction dataset": "input",
                                      "output link prediction dataset": "output"}

    print("\n\n********************* " + toplevel_pathway_name + " *********************\n\n")
    print(f"Threshold Degree = {threshold_degree}\n")
    # print("Threshold Degree = {}".format(threshold_degree))

    prev_task_name = None
    name_dict: dict[str, str] = {"input link prediction dataset": "Input Link Prediction Task",
                                 "output link prediction dataset": "Output Link Prediction Task"}
    # product 生成所有可能的组合
    # groupby 依照task_name 去分组
    """ output: 
    Task Name: input
    ('input', 'model1')
    ('input', 'model1')
    ('input', 'model2')
    ('input', 'model2')
    ----
    Task Name: output
    ('output', 'model1')
    ('output', 'model1')
    ('output', 'model2')
    ('output', 'model2')
    """
    for task_name, group in groupby(product(task_name_group, model_group),
                                    key=lambda x: x[0]):
        # if task_name != prev_task_name:
        #     print(f"------ Task: {name_dict[task_name]} ------")
        #     prev_task_name = task_name
        print(f"\n------ Task: {name_dict[task_name]} ------\n")
        for _, model in group:
            evaluate_on_primary_entities_and_secondary_entities_assist(data_version=data_version, task_name=task_name,
                                                                       toplevel_pathway_name=toplevel_pathway_name,
                                                                       model=model, direction=direction_dict[task_name],
                                                                       threshold_degree=threshold_degree)


def evaluate_on_primary_entities_and_secondary_entities_assist(data_version: DataWithVersion, task_name: str,
                                                               toplevel_pathway_name: str,
                                                               model: str, direction: str,
                                                               threshold_degree: int):
    """
    :param data_version: data_old = DataVersion("data")
    :param task_name: "input link prediction dataset" or "output link prediction dataset"
    :param toplevel_pathway_name: "Disease", "Metabolism", "Immune System", "Signal Transduction"
    :param model: "gcn", "hgnn", "hgnnp"
    :param direction: "input" or "output"
    :param threshold_degree
    :return:
    """

    print(f"\nEvaluating using Model: {model}, Direction: {direction}")

    model_engine = ModelEngine(
        data_version_name=data_version.data_version_name, task_name=task_name,
        toplevel_pathway_name=toplevel_pathway_name, model=model,
        direction=direction)

    reactions_with_primary_entities_masked, reactions_with_secondary_entities_masked = model_engine.get_reactions_with_entities_masked_for_testing(
        data_version=data_version,
        threshold_degree=threshold_degree)

    for entity_type, reactions_masked in [("Primary", reactions_with_primary_entities_masked),
                                          ("Secondary", reactions_with_secondary_entities_masked)]:
        print(f"For {entity_type} Entities: ")
        print(f"Num of {entity_type} Entities: {len(reactions_masked)}")
        print(model_engine.evaluate_on_list_of_reactions(reactions_masked))
        print("--------------------------------------------------------------")

    # print("For Primary Entities: ")
    # print("Num of Primary Entities: {}".format(len(reactions_with_primary_entities_masked)))
    # print(model_engine.evaluate_on_list_of_reactions(reactions_with_primary_entities_masked))
    #
    # print("--------------------------------------------------------------")
    #
    # print("For Secondary Entities: ")
    # print("Num of Secondary Entities: {}".format(len(reactions_with_secondary_entities_masked)))
    # print(model_engine.evaluate_on_list_of_reactions(reactions_with_secondary_entities_masked))

    # print(res)


if __name__ == '__main__':
    # degree process
    # data_old = DataVersion("data")
    # primary_secondary_entity_process(toplevel_pathway_name="Disease", data_version=data_old, threshold_degree=6)

    # primary_secondary_entity_process(toplevel_pathway_name="Metabolism", data_version=data_old, threshold_degree=7)

    # primary_secondary_entity_process(toplevel_pathway_name="Immune System", data_version=data_old, threshold_degree=6)

    # primary_secondary_entity_process(toplevel_pathway_name="Signal Transduction", data_version=data_old, threshold_degree=7)

    # create_table()

    case_study_on_multi_dataset(['Disease', 'Immune System', 'Metabolism', 'Signal Transduction'])

    # case_study_on_dataset('Disease')
    #
    # case_study_on_dataset('Immune System')
    #
    # case_study_on_dataset('Metabolism')
    #
    # case_study_on_dataset('Signal Transduction')

    # Case Study on primary and secondary entities

    # data_old = DataWithVersion("data")
    # case_study_on_primary_and_secondary_entities(toplevel_pathway_name="Disease", dataset_name="Disease", data_version=data_old, threshold_degree=6)

    # case_study_on_primary_and_secondary_entities(toplevel_pathway_name="Metabolism", dataset_name="Metabolism",
    #                                              data_version=data_old, threshold_degree=6)

    # case_study_on_primary_and_secondary_entities(toplevel_pathway_name="Immune System", dataset_name="Immune System", data_version=data_old, threshold_degree=6)

    # case_study_on_primary_and_secondary_entities(toplevel_pathway_name="Signal Transduction",
    #                                              dataset_name="Immune System", data_version=data_old,
    #                                              threshold_degree=6)

    # evaluate_on_primary_entities_and_secondary_entities_sweep(data_version=data_old, toplevel_pathway_name="Disease", threshold_degree=6)

    # evaluate_on_primary_entities_and_secondary_entities_sweep(data_version=data_old, toplevel_pathway_name="Metabolism",
    #                                                           threshold_degree=7)

    # evaluate_on_primary_entities_and_secondary_entities_sweep(data_version=data_old,
    #                                                           toplevel_pathway_name="Immune System",
    #                                                           threshold_degree=6)

    # evaluate_on_primary_entities_and_secondary_entities_sweep(data_version=data_old,
    #                                                           toplevel_pathway_name="Signal Transduction",
    #                                                           threshold_degree=7)
