from typing import Dict, Tuple, List

from case_study.bean.data_version import DataWithVersion
from case_study.service.edge_service import EdgeService
from case_study.service.node_service import node_service
import matplotlib.pyplot as plt


class PrimarySecondaryEntityEngine:

    def __init__(self):
        self.edge_service = EdgeService()
        pass

    def primary_secondary_entity_process(self, toplevel_pathway_name: str, data_version: DataWithVersion,
                                         threshold_degree: int):
        print("Dataset: " + toplevel_pathway_name)

        degree2nodes_dict = node_service.get_degree2node_dict(toplevel_pathway_name, data_version)

        for degree, nodes in degree2nodes_dict.items():
            print("Degree: " + str(degree) + "|||||" + "Num of Nodes: " + str(len(nodes)))

        # 从字典中提取数据
        degrees = list(degree2nodes_dict.keys())
        num_of_nodes = [len(nodes) for nodes in degree2nodes_dict.values()]

        plt.figure(figsize=(10, 6))

        # 使用loglog来绘制log-log图
        plt.loglog(degrees, num_of_nodes, 'o-', label='Degree Distribution')

        plt.title('Degree Distribution on log-log scale')
        plt.xlabel('Degree')
        plt.ylabel('Number of Nodes')
        plt.legend()
        plt.grid(True, which="both", ls="--", c='0.65')  # 添加网格线

        plt.show()

        self.classify_entities(toplevel_pathway_name=toplevel_pathway_name, degree2nodes_dict=degree2nodes_dict,
                               threshold_degree=threshold_degree, data_version=data_version)

    def get_primary_secondary_entities(self, degree2nodes_dict: Dict[int, List[int]], threshold_degree: int) -> Tuple[
        List[int], List[int]]:
        primary_nodes: list[int] = []
        secondary_nodes: list[int] = []

        for degree, nodes in degree2nodes_dict.items():
            if degree <= threshold_degree:
                primary_nodes.extend(nodes)
            else:
                secondary_nodes.extend(nodes)

        return primary_nodes, secondary_nodes

    # Press the green button in the gutter to run the script.
    # 我手动选一个度数，然后小于等于这个度数的为primary entity，大于这个度数的是secondary entity
    def classify_entities(self, toplevel_pathway_name: str, degree2nodes_dict: Dict[int, List[int]],
                          threshold_degree: int,
                          data_version: DataWithVersion):
        """
        根据提供的度数阈值, 将实体分类为primary和secondary, 并返回它们的百分比。
        """

        primary_nodes, secondary_nodes = self.get_primary_secondary_entities(degree2nodes_dict, threshold_degree)

        # Print header
        print("=" * 80)
        print(f"{'Node Name':<60} {'stId':<20}")
        print("=" * 80)

        for node_index in secondary_nodes:
            node = node_service.get_node_from_dataset_based_on_index(toplevel_pathway_name=toplevel_pathway_name,
                                                                     index=node_index, data_version=data_version)
            print(f"{node.name:<60} {node.stId:<20}")

        # # data = []
        # for node_index in secondary_nodes:
        #     node = node_service.get_node_from_dataset_based_on_index(toplevel_pathway_name=toplevel_pathway_name, index=node_index, data_version=data_old)
        #     # print(f"{node.name:<50} {node.stId}")
        #     # data.append([node.name, node.stId])

        # print(tabulate(data, headers=["Node Name", "stId"], tablefmt="grid"))

        total_nodes = len(primary_nodes) + len(secondary_nodes)

        primary_percentage = len(primary_nodes) / total_nodes * 100
        secondary_percentage = len(secondary_nodes) / total_nodes * 100

        print(f"Degree Threshold: {threshold_degree}")
        print(f"Primary Entities Percentage: {primary_percentage:.2f}%")
        print(f"Secondary Entities Percentage: {secondary_percentage:.2f}%")

        # 绘制饼状图
        labels = ['Primary Entities', 'Secondary Entities']
        sizes = [primary_percentage, secondary_percentage]
        colors = ['#ff9999', '#66b2ff']
        explode = (0.1, 0)  # 将第一个部分（Primary Entities）分离出来

        plt.figure(figsize=(8, 8))
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
        plt.title('Distribution of Primary and Secondary Entities')
        plt.show()

        return primary_percentage, secondary_percentage




