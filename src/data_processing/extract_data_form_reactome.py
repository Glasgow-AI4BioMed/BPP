# from neo4j import GraphDatabase
import re

import numpy as np
# from py2neo import Graph, Node, Relationship
from py2neo import Graph
import os
import time
from pyecharts import options as opts
from pyecharts.charts import Bar


# match p = (n:PhysicalEntity) - [:input|output] - () where n.speciesName = 'Homo sapiens'  and n.stId = 'R-HSA-9626061' return p limit 100
# match p = (n:PhysicalEntity) - [:input|output] - () where n.speciesName = 'Homo sapiens'  and n.stId = 'R-HSA-9626045' return p limit 100

# match p = (n) - [:hasComponent] - () where n.stId = 'R-HSA-9626061' return p
# match p = (n) - [:hasComponent] - () where n.stId = 'R-HSA-9626045' return p

# 我们现在发现，我们没有办法根据name去反过来找id，所以，我们先根据pathway，找到所有的reaction_id，去重，然后根据去重后的reaction_id找到它们对应的reaction_name
# 这些 reaction_name 仍然有重复的，这是因为 两个不同的 reaction_id 可能对应 同一个name，这是因为这个reaction 实际上有完全相同的 input 和 output节点，只是这些节点的id不同，因为它们位于不同的细胞环境中
# 我们选择直接丢掉这些重复的name，也就是如果有多个reaction id对应同一个reaction name的情况，我们只保留一个reaction id，
# 我们会构建一个 reaction_id -> reaction_name的字典，然后把这个字典反过来，得到一个 reaction_name -> reaction_id的字典，第二个字典是一一对应的关系
# 我们保存一份reaction_name list 用于打印
# 注意，我们这样的行为并不会使节点数目减少，因为我们后面本来就会把处于不同环境中的相同物质(它们id不同)视为同一节点
# 我们现在抛弃了某个reaction_id，但保留的reaction_id里面已经拿到了同样的节点，那你可能会疑惑，它们节点的id那不一样啊，会不会你抛弃的节点其实还参加了别的反应呢？
# 这样你不就少拿了一些 relationship 吗？ 但实际上如果这个节点还参加了别的反应，那我们在别的reaction_id中就可以拿到它啊，如果好巧不巧，这个reaction_id也同名了，那我们恰好又把它参与的reaction_id抛弃掉了
# 但即使这样，我们保留下来的reaction_id的参与节点，也会有它的替代品(同一物质处于不同环境)，我们的节点关系并不会丢失

# 我们会利用这些留下了来的reaction_id 去寻找对应的entity_id，拿到entity_id后，我们也会把entity_id先去一次重，然后根据去重后的entity_id找到entity_name
# 这些entity_name会有一种新的是否重复的判定方法，就是它们如果是同一种物质，但是处于不同环境，那他们也是相同的，我们每拿到一个entity_name就会去除它的环境字段，得到一个entity_name_X
# 然后我们会建立一个 字典 entity_id -> entity_name_X，同时把它反过来生成一个字典 entity_name_X -> entity_id，第二个字典是一一对应的
# 这些entity_name_X中是有很多重复的，然后我们把所有的entity_name_X放在一起，去重后得到我们的entity_name
# 然后我们去根据已有的方法找到一个reaction_id对应的所有relationship : entity_id, reaction_id, direction
# 然后根据字典 entity_id -> entity_name_X，我们找到了这个entity_id对应的entity_name_X，并确定这个entity_name在 去重后的entity_name_X 列表中的位置
# 同理，我们根据字典reaction_id -> reaction_name，找到reaction_id 对应的 reaction_name以及它在reaction_name list 中的位置
# 这样以后，我们就拿到了所有的reactions entities 和对应的 relationships

# 对于regulation，我们在找到了reaction_id后，根据 reaction_id 去寻找 regulation_id 也就是entity_id，然后会把它们添加到总的entity id中，并如28行所示去建立entity_id -> entity_name_X 字典
# 但是这种关系怎么储存呢？我们会去寻找regulation_relationship: entity_id, reaction_id ，不存方向
# 然后我们还是根据entity_id -> entity_name_X字典，找到entity_id对应的entity_name_X和其下标，然后reaction_id id 同理
# 然后我们生成三个东西：
# entity_name_X_index, reaction_nameP_index
# entity_name_X_index, reaction_nameP_index -1
# entity_name_X_index, reaction_nameP_index 1
# 第一个是为了单独保存，指示regulation关系
# 后面两个加入到relationship中

# 最后我们还剩下components没解决
# 我们现在已经得到了所有的entity id，但是这些entity id 对应着的名字可能有重复，也就是说可能是多个entity id对应一个entity name
# 我们解决它的方法就是和之前一样，借助之前得到的没有任何重复的list: entity_name_X，然后借助字典：entity_name_X -> entity_id
# 这个字典是一一对应的关系
# 我们可以拿到一组绝不会重复的entity_id
# 然后可以根据这些entity id 找到很多的 component id，
# 我们建立 component_relationship: entity id, component id 这是一个list
# 我们根据这些component id 查到它们对应的 component name，注意，多个component id 可能对应同一个component name
# 建立字典 component id -> component name
#


class PathWayProcessor:
    """The processor which deals with the pathway, assistance class for ReactomeProcessor, we won't use it directly

    PathWayProcessor class has methods for pathways in Reactome database(https://reactome.org/)
    it provides method 'get_all_top_level_pathways' to get all the top pathway id in Reaqctome
    it is a tool member of class ReactomeProcessor, which is for final execution on extracting and store data from reactome,
    and analyze them to gain some statistical information. You can simply regard it as a inner class of ReactomeProcessor
    In fact, we don't need to use this class manually, as the ReactomeProcessor will call the methods of PathWayProcessor automatically

    Attributes:
        __graph:  The inner Reactome graph object, offering a connection to Reactome database, relying on py2neo package
    """

    def __init__(self, graph: Graph):
        """ Initialize method

        initialize the class PathWayProcessor, we input a graph object to be the value of inner __graph object

        :param
            graph(Graph): the graph object of class Graph, offering a connection to Reactome database
        """
        self.__graph = graph

    # get_all_top_level_pathways(self):
    # output:
    def get_all_top_level_pathways_ids(self) -> list[str]:
        """Method to get all the top level pathway id

        This method offers ways to get all the pathway ids from Reactome database, and return them as a list

        :return:
            toplevel_pathways(list[str]):  a list of top level pathways' id
        """
        # TopLevelPathway
        # "MATCH (n:TopLevelPathway) WHERE n.speciesName='Homo sapiens' RETURN n.schemaClass"
        # "MATCH (n) WHERE any(label in labels(n) WHERE label in ['TopLevelPathway', 'Pathway']) AND n.speciesName='Homo sapiens' RETURN n LIMIT 20"
        # "MATCH (n) WHERE any(label in labels(n) WHERE label in ['TopLevelPathway']) AND n.speciesName='Homo sapiens' RETURN n.stId, n.displayName"
        __instruction__ = "MATCH (n) WHERE any(label in labels(n) WHERE label in ['TopLevelPathway']) AND n.speciesName='Homo sapiens' RETURN n.stId"
        toplevel_pathways = self.__graph.run(__instruction__).to_ndarray()

        # Here, we means reducing output to one dimension
        toplevel_pathways = toplevel_pathways.flatten(order='C').tolist()

        return toplevel_pathways

    def get_all_top_level_pathways_names(self) -> list[str]:
        all_top_level_pathways_ids = self.get_all_top_level_pathways_ids()
        all_top_level_pathways_names: list[str] = list()
        for pathway_stId in all_top_level_pathways_ids:
            __instruction__ = "MATCH (n:Pathway) WHERE n.stId = '" + str(pathway_stId) + "' RETURN n.displayName"
            pathways = self.__graph.run(__instruction__).to_ndarray()
            pathways = pathways.flatten(order='C')
            if pathways.size == 1:
                all_top_level_pathways_names.append(pathways[0])
            else:
                print("sorry, we can't find pathway with stId = '" + str(pathway_stId) + "'")

        return all_top_level_pathways_names

    def get_top_level_pathway_id_by_name(self, pathway_name):
        all_top_level_pathways_names: list[str] = self.get_all_top_level_pathways_names()
        all_top_level_pathways_ids: list[str] = self.get_all_top_level_pathways_ids()
        if pathway_name not in all_top_level_pathways_names:
            print("sorry, we can't find pathway with name = '" + str(pathway_name) + "'")
            return ""
        else:
            index = all_top_level_pathways_names.index(pathway_name)
            pathway_id = all_top_level_pathways_ids[index]

        return pathway_id


# The processor which deals with the reaction
class ReactionProcessor:
    """The ReactionProcessor class is a processor which deals with the reaction

    We won't use this class manually, as the class ReactomeProcessor will call the methods of this class automatically.
    The ReactionProcessor class provides the methods to get the reactions form Reactome dataset,
    and analyze these reactions to get input, output or all the entities of single reaction or a set of reactions.
     'get_reactions_from_pathway'
    to get reactions from a single pathway of Reactome database based on a pathway's Id,
    the method 'get_all_reactions_of_homo_sapiens_in_Reactome' is to get all the reactions of homo sapiens in Reactome database
    the method 'get_all_reactions_of_homo_sapiens_based_on_all_top_pathways_in_Reactome' is to get all the reactions based on all the top level pathways,
    as there are a very small amount of reactions with no pathways, we just use this method to quit these reactions
    the method '

    Attributes:
        __graph:    The inner Reactome graph object, offering a connection to Reactome database, relying on py2neo package
        entity_id_index:    the index of entity element in a relationship object(entity id, reaction id, direction(-1/1))
        reaction_id_index:  the index of reaction element in a relationship object
        direction_index:    the index of direction of the entity to a reaction in a relationship object, 1 means the entity is output of the reaction, -1 means the entity is input of the reaction
    """

    def __init__(self, graph: Graph):
        """The initialization method of ReactionProcessor

        The method is to set the value of inner __graph object, and create an object of ReactionProcessor,
        it also set the default value for index of entity, reaction, direction of a relationship

        :param
            graph: The inner Reactome graph object, offering a connection to Reactome database, relying on py2neo package
        """
        self.__graph = graph
        # self.id_index = 0
        # self.name_index = 1
        self.entity_id_index = 0
        self.reaction_id_index = 1
        self.direction_index = 2

    def get_reactions_ids_from_single_pathway(self, pathway_stId):
        """ Method to get a list of reactions ids
        This method is to get a list of reaction ids based on a single pathway id
        :param pathway_stId: a single pathway id
        :return reaction_ids: a list of reaction ids
        """
        # we used to return "reaction_stId, reaction_displayName", but now only "reaction_stId"
        __instruction = "MATCH (n:Pathway)-[r:hasEvent*1..]->(m:Reaction) WHERE n.stId = '" + str(
            pathway_stId) + "' AND n.speciesName='Homo sapiens' RETURN m.stId"
        reaction_ids = self.__graph.run(__instruction).to_ndarray()
        reaction_ids = reaction_ids.flatten(order='C').tolist()

        # MATCH p = (n:TopLevelPathway)-[r:hasEvent*1..]->(m:BlackBoxEvent) WHERE n.speciesName = 'Homo sapiens' return p limit 20
        __instruction = "MATCH (n:Pathway)-[r:hasEvent*1..]->(m:BlackBoxEvent) WHERE n.stId = '" + str(
            pathway_stId) + "' AND n.speciesName='Homo sapiens' RETURN m.stId"

        black_box_event_ids = self.__graph.run(__instruction).to_ndarray()
        black_box_event_ids = black_box_event_ids.flatten(order='C').tolist()

        # We regard black_box_event as a kind of reactions, so we'll combine these two list - reaction_ids and black_box_event_ids
        # reaction_ids.extend(black_box_event_ids)
        reaction_ids = reaction_ids + black_box_event_ids

        # There'll be some duplicated reactions via our method in one pathway, so just reduce the duplicated ones
        reaction_ids = list(set(reaction_ids))

        # list of reaction_id
        return reaction_ids

    # unused method
    # 一些id 会对应一个很复杂的名字，比如	[(2'-deoxy)cytidine + ATP => (d)CMP + ADP (DCK),phosphorylation of 2'-Deoxycytidine to 2'-Deoxycytidine 5'-phosphate]
    # 这个名字中间有逗号，会被我们视为一个list of name，这样搞肯定是不行的
    # match p = (n) where n.stId='R-HSA-73598' return p limit 20
    def get_reactions_display_names_from_single_pathway(self, pathway_stId):
        __instruction = "MATCH (n:Pathway)-[r:hasEvent*1..]->(m:Reaction) WHERE n.stId = '" + str(
            pathway_stId) + "' AND n.speciesName='Homo sapiens' RETURN m.displayName"
        reaction_names = self.__graph.run(__instruction).to_ndarray()
        reaction_names = reaction_names.flatten(order='C').tolist()

        __instruction = "MATCH (n:Pathway)-[r:hasEvent*1..]->(m:BlackBoxEvent) WHERE n.stId = '" + str(
            pathway_stId) + "' AND n.speciesName='Homo sapiens' RETURN m.displayName"
        black_box_event_names = self.__graph.run(__instruction).to_ndarray()
        black_box_event_names = black_box_event_names.flatten(order='C').tolist()

        # reaction_names.extend(black_box_event_names)
        reaction_names = reaction_names + black_box_event_names

        # There'll be some duplicated reactions via our method in one pathway, so just reduce the duplicated ones
        reaction_names_set = set(reaction_names)
        reaction_names_list = list(reaction_names_set)

        # list of reaction_name
        return reaction_names_list

    def get_reaction_display_name_by_reaction_id(self, reaction_id: str) -> str:
        """
        This method is to get the display name of a reaction based on its id
        :param reaction_id: the stId of a reaction
        :return: reaction_name: the display name of a reaction
        """

        __instruction = "MATCH (n:ReactionLikeEvent) WHERE n.stId = '" + str(
            reaction_id) + "' AND n.speciesName='Homo sapiens' RETURN n.displayName"
        reaction_name = self.__graph.run(__instruction).to_ndarray().flatten(order='C').tolist()
        if reaction_name is not None and len(reaction_name) > 0:
            return reaction_name[0]
        else:
            return "Not_Found_This_Name"

    def get_reaction_name_by_reaction_id(self, reaction_id: str) -> str:
        """
        This method is to get the name of a reaction based on its id
        :param reaction_id: the stId of a reaction
        :return: reaction_name: the name of a reaction
        """

        __instruction = "MATCH (n:ReactionLikeEvent) WHERE n.stId = '" + str(
            reaction_id) + "' AND n.speciesName='Homo sapiens' RETURN n.name"
        reaction_name = self.__graph.run(__instruction).to_ndarray().flatten(order='C').tolist()
        if reaction_name is not None and len(reaction_name) > 0:
            return reaction_name[0]
        else:
            return "Not_Found_This_Name"

    def get_list_of_reaction_ids_without_duplicate_name_and_mapping_list_of_reaction_name_by_list_of_reaction_ids(
            self, reaction_ids: list):
        """
        :param reaction_ids:
        :return:
        """
        reaction_names_list = list()
        reaction_ids_list_without_duplicate_names: list[str] = list()

        reaction_name_to_list_of_reaction_ids_dict: dict[str, list[str]] = dict()
        for reaction_id in reaction_ids:
            # reaction_name = self.get_reaction_display_name_by_reaction_id(reaction_id)
            reaction_name = self.get_reaction_name_by_reaction_id(reaction_id)

            if reaction_name not in reaction_name_to_list_of_reaction_ids_dict.keys():
                reaction_name_to_list_of_reaction_ids_dict[reaction_name] = list()
                reaction_name_to_list_of_reaction_ids_dict[reaction_name].append(reaction_id)
            else:
                reaction_name_to_list_of_reaction_ids_dict[reaction_name].append(reaction_id)

            reaction_names_list.append(reaction_name)

        reaction_names_list = list(set(reaction_names_list))
        # sort the reaction_names_list
        reaction_names_list = sorted(reaction_names_list)

        for reaction_name, list_of_reaction_ids in reaction_name_to_list_of_reaction_ids_dict.items():
            list_of_reaction_ids.sort()

        # 'R-HSA-9687435'
        # 'R-HSA-9698265'
        for reaction_name in reaction_names_list:
            reaction_ids_list_without_duplicate_names.append(reaction_name_to_list_of_reaction_ids_dict.get(reaction_name)[0])

        return reaction_ids_list_without_duplicate_names, reaction_names_list

    def get_all_reactions_of_homo_sapiens_in_Reactome(self) -> list:
        reactions_ids = self.__graph.run(
            "MATCH (n:ReactionLikeEvent) WHERE n.speciesName='Homo sapiens' RETURN n.stId").to_ndarray()
        reactions_ids = reactions_ids.flatten(order='C').tolist()
        return reactions_ids

    def get_all_reactions_of_homo_sapiens_based_on_all_top_pathways_in_Reactome(self, top_pathways) -> list:
        """
        This method is to get a list of unique reaction ids in Reactome based on a list of top level pathways ids
        :param top_pathways: a list of top level pathways ids in Reactome
        :return list(reactions_set): a list of unique reaction ids based on all the top level pathways in Reactome database
        """
        reactions_set = set()
        for top_pathway_id in top_pathways:
            reactions = self.get_reactions_ids_from_single_pathway(top_pathway_id)
            for reaction in reactions:
                reactions_set.add(reaction)
        return list(reactions_set)

    def __get_input_relationship_of_single_reaction(self, reaction_stId):
        __instruction_input__ = "MATCH (n:ReactionLikeEvent)-[r:input]->(m:PhysicalEntity) WHERE n.stId = '" + str(
            reaction_stId) + "' AND n.speciesName='Homo sapiens' RETURN m.stId, n.stId"
        __input_edges__ = self.__graph.run(__instruction_input__).to_ndarray()

        if __input_edges__ is None or len(__input_edges__) == 0:
            return list()

        # get the line number of ndarray，which represents the number of input
        num_of_input_edges = __input_edges__.shape[0]

        # we build a complementary vertex based on the number of input

        # supplement_vector = np.negative(np.ones(num_of_input_edges))
        supplement_vector = np.negative(np.ones(num_of_input_edges, dtype=int))

        # we add a new column, then fill it with 0, which represents input
        __input_edges__ = np.insert(__input_edges__, 2, supplement_vector, axis=1)

        # PhysicalEntity_id, Reaction_id, 0
        return __input_edges__.tolist()

    def __get_output_relationship_of_single_reaction(self, reaction_stId):
        __instruction_output__ = "MATCH (n:ReactionLikeEvent)-[r:output]->(m:PhysicalEntity) WHERE n.stId = '" + str(
            reaction_stId) + "' AND n.speciesName='Homo sapiens' RETURN m.stId, n.stId"
        __output_edges__ = self.__graph.run(__instruction_output__).to_ndarray()

        if __output_edges__ is None or len(__output_edges__) == 0:
            return list()

        # get the line number of ndarray，which represents the number of input
        __num_of_output_edges = __output_edges__.shape[0]

        # we build a complementary vertex based on the number of input

        # supplement_vector = np.ones(__num_of_output_edges)
        supplement_vector = np.ones(__num_of_output_edges, dtype=int)

        # we add a new column, then fill it with 0, which represents input
        __output_edges__ = np.insert(__output_edges__, 2, supplement_vector, axis=1)

        # PhysicalEntity_id, Reaction_id, 1
        return __output_edges__.tolist()

    def __get_regulation_relationship_of_single_reaction(self, reaction_stId):
        # 475 * 12 = 6000 - 25 * 12 = 300 = 5700
        __instruction_regulated = "MATCH (n:ReactionLikeEvent)-[:regulatedBy]->()-[:regulator]->(m:PhysicalEntity) WHERE n.stId = '" + str(
            reaction_stId) + "' AND n.speciesName='Homo sapiens' RETURN m.stId, n.stId"

        # entity_id, reaction_id
        regulation_edges = self.__graph.run(__instruction_regulated).to_ndarray()

        if regulation_edges is None or len(regulation_edges) == 0:
            return list()

        # get the line number of ndarray，which represents the number of input
        __num_of_regulation_edges = regulation_edges.shape[0]

        # we build a complementary vertex based on the number of input
        # supplement_vector = np.zeros(__num_of_regulation_edges)
        supplement_vector = np.zeros(__num_of_regulation_edges, dtype=int)

        # we add a new column, then fill it with 0, which represents input
        regulation_edges = np.insert(regulation_edges, 2, supplement_vector, axis=1)

        return regulation_edges.tolist()

    def get_all_related_relationships_of_single_reaction(self, reaction_stId):
        # R-HSA-9613352
        # R-HSA-9646383
        # 没有input的reaction
        # match p = (n) - [:input|output] - () where n.stId = 'R-HSA-3223236' return p

        relationships_for_reaction = np.empty(shape=(0, 3))
        __input_edges = self.__get_input_relationship_of_single_reaction(reaction_stId)
        __output_edges = self.__get_output_relationship_of_single_reaction(reaction_stId)
        __regulation_edges = self.__get_regulation_relationship_of_single_reaction(reaction_stId)

        if __input_edges is not None and len(__input_edges) != 0:
            relationships_for_reaction = np.vstack((relationships_for_reaction, __input_edges))

        if __output_edges is not None and len(__output_edges) != 0:
            relationships_for_reaction = np.vstack((relationships_for_reaction, __output_edges))

        if __regulation_edges is not None and len(__regulation_edges) != 0:
            relationships_for_reaction = np.vstack((relationships_for_reaction, __regulation_edges))

        # PhysicalEntity_id, Reaction_id, 0/-1/1
        return relationships_for_reaction.tolist()

    def get_all_unique_relationships_of_set_of_reactions(self, reaction_stIds):
        """

        :param reaction_stIds:
        :return:
        """
        edges_for_set_of_reactions = np.empty(shape=(0, 3))
        for reaction_stId in reaction_stIds:
            edges_for_single_reaction = self.get_all_related_relationships_of_single_reaction(reaction_stId)
            edges_for_set_of_reactions = np.vstack((edges_for_set_of_reactions, edges_for_single_reaction))

        # reduce the duplicate ones
        # PhysicalEntity_id, Reaction_id, 0/1    -a list
        unique_edges_for_set_of_reactions = list(set(tuple(edge) for edge in edges_for_set_of_reactions))

        # every edge from tuple to list
        unique_edges_for_set_of_reactions = list(list(edge) for edge in unique_edges_for_set_of_reactions)

        return unique_edges_for_set_of_reactions

    def get_physical_entities_ids_from_single_reaction_id(self, reaction_stId) -> list:
        """
        This method is to get a list of physical entities ids based on a single reaction id
        :param reaction_stId: a list of reaction ids
        :return physical_entities_ids: a list of physical entity ids for a single reaction
        """
        # get entities from reaction id via input relationship
        __instruction_input = "MATCH (n:ReactionLikeEvent)-[r:input]->(m:PhysicalEntity) WHERE n.stId = '" + str(
            reaction_stId) + "' AND n.speciesName='Homo sapiens' RETURN m.stId"
        input_entities = self.__graph.run(__instruction_input).to_ndarray().flatten(order='C').tolist()

        # get entities from reaction id via output relationship
        __instruction_output = "MATCH (n:ReactionLikeEvent)-[r:output]->(m:PhysicalEntity) WHERE n.stId = '" + str(
            reaction_stId) + "' AND n.speciesName='Homo sapiens' RETURN m.stId"
        output_entities = self.__graph.run(__instruction_output).to_ndarray().flatten(order='C').tolist()

        # get entities from reaction id via regulated relationship
        __instruction_regulated = "MATCH (n:ReactionLikeEvent)-[r:regulatedBy]->()-[:regulator]->(m:PhysicalEntity) WHERE n.stId = '" + str(
            reaction_stId) + "' AND n.speciesName='Homo sapiens' RETURN m.stId"

        regulation_entities = self.__graph.run(__instruction_regulated).to_ndarray().flatten(order='C').tolist()

        physical_entities_ids = input_entities + output_entities + regulation_entities

        # list of PhysicalEntity_id
        return physical_entities_ids

    def get_unique_physical_entities_ids_from_list_of_reactions_ids(self, reaction_ids):
        """
        This method is to get a list of unique phisical entity ids based on a list of reaction ids
        :param reaction_ids: a list of reaction ids
        :return physical_entities_ids: a list of physical entity ids for a list of reaction ids
        """
        physical_entities_set = set()

        for reaction_stId in reaction_ids:
            # list of PhysicalEntity_id
            physical_entities_ids_single_reaction = self.get_physical_entities_ids_from_single_reaction_id(
                reaction_stId)

            for physical_entity in physical_entities_ids_single_reaction:
                physical_entities_set.add(physical_entity)

        physical_entities_ids = list(physical_entities_set)

        return physical_entities_ids

    def get_physical_entity_name_by_physical_entity_id(self, physical_entity_id) -> str:
        __instruction = "MATCH (n:PhysicalEntity) WHERE n.stId = '" + str(
            physical_entity_id) + "' AND n.speciesName='Homo sapiens' RETURN n.displayName"

        display_name = self.__graph.run(__instruction).to_ndarray().flatten(order='C').tolist()
        if display_name is None or len(display_name) == 0:
            __instruction = "MATCH (n:PhysicalEntity) WHERE n.stId = '" + str(
                physical_entity_id) + "' RETURN n.displayName"
            display_name = self.__graph.run(__instruction).to_ndarray().flatten(order='C').tolist()
            if display_name is None or len(display_name) == 0:
                return ""

        display_name = display_name[0]
        original_name = display_name
        display_name = display_name.strip()
        display_name = re.sub(u"\\[.*?\\]", "", display_name)
        display_name = display_name.strip()
        return display_name

    def get_original_physical_entity_list_and_list_of_unique_physical_entities_id_without_duplicate_name_and_list_of_mapping_names_and_original_physical_entity_id_to_no_duplicate_name_physical_entity_id_dic_from_list_of_reactions_ids(
            self, reaction_ids):
        original_physical_entity_id_to_physical_entity_name_dict = dict()
        physical_entity_name_to_list_of_entity_id: dict[str, list[str]] = dict( )
        physical_entity_name_to_physical_entity_id_dic_without_duplicate_name = dict()

        physical_entity_id_list = list()
        physical_entity_name_list = list()
        original_physical_entity_id_to_no_duplicate_name_physical_entity_id_dic = dict()

        original_physical_entity_list = self.get_unique_physical_entities_ids_from_list_of_reactions_ids(reaction_ids)

        for physical_entity_id in original_physical_entity_list:
            physical_entity_name = self.get_physical_entity_name_by_physical_entity_id(physical_entity_id)

            # There will be more than on id mapping to the same name
            original_physical_entity_id_to_physical_entity_name_dict[physical_entity_id] = physical_entity_name

            physical_entity_name_list.append(physical_entity_name)

            if physical_entity_name not in physical_entity_name_to_list_of_entity_id.keys():
                physical_entity_name_to_list_of_entity_id[physical_entity_name] = list()
                physical_entity_name_to_list_of_entity_id[physical_entity_name].append(physical_entity_id)
            else:
                physical_entity_name_to_list_of_entity_id[physical_entity_name].append(physical_entity_id)

        # sort the list of entity ids for every single entity name
        for entity_name, list_of_entity_ids in physical_entity_name_to_list_of_entity_id.items():
            list_of_entity_ids.sort()
            physical_entity_name_to_physical_entity_id_dic_without_duplicate_name[entity_name] = list_of_entity_ids[0]

        physical_entity_name_list = list(set(physical_entity_name_list))

        # sort the physical_entity_name_list
        physical_entity_name_list = sorted(physical_entity_name_list)

        for physical_entity_name in physical_entity_name_list:
            physical_entity_id_list.append(
                physical_entity_name_to_physical_entity_id_dic_without_duplicate_name[physical_entity_name])

        for original_physical_entity_id, physical_entity_name in original_physical_entity_id_to_physical_entity_name_dict.items():
            original_physical_entity_id_to_no_duplicate_name_physical_entity_id_dic[original_physical_entity_id] = \
                physical_entity_name_to_physical_entity_id_dic_without_duplicate_name[physical_entity_name]

        return original_physical_entity_list, physical_entity_id_list, physical_entity_name_list, original_physical_entity_id_to_no_duplicate_name_physical_entity_id_dic


# The processor which deals with the physical entity
class PhysicalEntityProcessor:
    # TypeDetectorOfPhysicalEntity is the inner class to help to define a specific type for the physical entity
    class TypeDetectorOfPhysicalEntity:
        complex_arr = ['Complex']
        polymer_arr = ['Polymer']
        genomeEncodedEntity_arr = ['GenomeEncodedEntity', 'EntityWithAccessionedSequence']
        entitySet_arr = ['EntitySet', 'CandidateSet', 'DefinedSet']
        simpleEntity_arr = ['SimpleEntity']
        otherEntity_arr = ['OtherEntity']
        drug_arr = ['Drug', 'ChemicalDrug', 'ProteinDrug']
        type_dic = dict(complex_type=complex_arr, polymer_type=polymer_arr,
                        genomeEncodedEntity_type=genomeEncodedEntity_arr, entitySet_type=entitySet_arr,
                        simpleEntity_type=simpleEntity_arr, otherEntity_type=otherEntity_arr, drug_type=drug_arr)

        def __init__(self):
            pass

        def __is_complex(self, input_entity_schemaClass):
            return str(input_entity_schemaClass) in self.type_dic.get('complex_type')

        def __is_polymer(self, input_entity_schemaClass):
            return str(input_entity_schemaClass) in self.type_dic.get('polymer_type')

        def __is_genomeEncodedEntity(self, input_entity_schemaClass):
            return str(input_entity_schemaClass) in self.type_dic.get('genomeEncodedEntity_type')

        def __is_entitySet(self, input_entity_schemaClass):
            return str(input_entity_schemaClass) in self.type_dic.get('entitySet_type')

        def __is_simpleEntity(self, input_entity_schemaClass):
            return str(input_entity_schemaClass) in self.type_dic.get('simpleEntity_type')

        def __is_otherEntity(self, input_entity_schemaClass):
            return str(input_entity_schemaClass) in self.type_dic.get('otherEntity_type')

        def __is_drug(self, input_entity_schemaClass):
            return str(input_entity_schemaClass) in self.type_dic.get('drug_type')

        def get_type_of_physical_entity(self, input_entity_schemaClass):
            if (self.__is_complex(input_entity_schemaClass)
                    or self.__is_polymer(input_entity_schemaClass)
                    or self.__is_simpleEntity(input_entity_schemaClass)
                    or self.__is_otherEntity(input_entity_schemaClass)):
                # 'Complex' 'Polymer' 'SimpleEntity' 'OtherEntity'
                return input_entity_schemaClass
            elif (self.__is_genomeEncodedEntity(input_entity_schemaClass)):
                return 'GenomeEncodedEntity'
            elif (self.__is_entitySet(input_entity_schemaClass)):
                return 'EntitySet'
            elif (self.__is_drug(input_entity_schemaClass)):
                return 'Drug'
        # end of the inner class -> TypeDetectorOfPhysicalEntity

    # the __init__ for the outer class "PhysicalEntityProcessor"
    def __init__(self, graph: Graph):
        self.__graph = graph
        self.__type_detector = self.TypeDetectorOfPhysicalEntity()
        self.__funcs_for_get_components = {'Complex': self.__get_components_of_complex,
                                           'Polymer': self.__get_components_of_polymer,
                                           'GenomeEncodedEntity': self.__get_components_of_GenomeEncodedEntity,
                                           'EntitySet': self.__get_components_of_EntitySet,
                                           'SimpleEntity': self.__get_components_of_SimpleEntity,
                                           'OtherEntity': self.__get_components_of_OtherEntity,
                                           'Drug': self.__get_components_of_Drug}
        self.id_index = 0
        self.name_index = 1

    def __get_components_of_complex(self, physical_entity_id: str):
        # MATCH (n:Complex)-[r:hasComponent]->(m) WHERE n.stId = 'R-HSA-917704' AND n.speciesName='Homo sapiens' RETURN m.stId, m.displayName
        __instruction__ = "MATCH (n:Complex)-[r:hasComponent]->(m) WHERE n.stId = '" + str(
            physical_entity_id) + "' RETURN m.stId"
        __components__ = self.__graph.run(__instruction__).to_ndarray().flatten(order='C').tolist()

        # if Complex has no components, we define itself as a component.
        if (len(__components__) == 0):
            # MATCH (n:Complex) WHERE n.stId = 'R-HSA-917704' AND n.speciesName='Homo sapiens' RETURN n.stId, n.displayName
            __instruction__ = "MATCH (n:Complex) WHERE n.stId = '" + str(
                physical_entity_id) + "' RETURN n.stId"
            __components__ = self.__graph.run(__instruction__).to_ndarray().flatten(order='C').tolist()

        return __components__

    def __get_components_of_polymer(self, physical_entity_id: str):
        # MATCH (n:Polymer)-[r:repeatedUnit]->(m) WHERE n.stId = 'R-HSA-9626247' AND n.speciesName='Homo sapiens' RETURN m.stId, m.displayName
        __instruction__ = "MATCH (n:Polymer)-[r:repeatedUnit]->(m) WHERE n.stId = '" + str(
            physical_entity_id) + "' RETURN m.stId"
        __components__ = self.__graph.run(__instruction__).to_ndarray().flatten(order='C').tolist()

        # if Polymer has no components, we define itself as a component.
        if (len(__components__) == 0):
            # MATCH (n:Polymer) WHERE n.stId = 'R-HSA-2214302' AND n.speciesName='Homo sapiens' RETURN n.stId, n.displayName
            __instruction__ = "MATCH (n:Polymer) WHERE n.stId = '" + str(
                physical_entity_id) + "' RETURN n.stId"
            __components__ = self.__graph.run(__instruction__).to_ndarray().flatten(order='C').tolist()

        return __components__

    # GenomeEncodedEntity has no components, so define itself as an attribute
    def __get_components_of_GenomeEncodedEntity(self, physical_entity_id: str):
        # MATCH (n:GenomeEncodedEntity) WHERE n.stId = 'R-HSA-2029007' AND n.speciesName='Homo sapiens' RETURN n.stId, n.displayName
        __instruction__ = "MATCH (n:GenomeEncodedEntity) WHERE n.stId = '" + str(
            physical_entity_id) + "' RETURN n.stId"
        __components__ = self.__graph.run(__instruction__).to_ndarray().flatten(order='C').tolist()
        return __components__

    def __get_components_of_EntitySet(self, physical_entity_id: str):
        # MATCH (n:EntitySet)-[r:hasMember]->(m) WHERE n.stId = 'R-HSA-170079' AND n.speciesName='Homo sapiens' RETURN m.stId, m.displayName
        __instruction__ = "MATCH (n:EntitySet)-[r:hasMember]->(m) WHERE n.stId = '" + str(
            physical_entity_id) + "' RETURN m.stId"
        __components__ = self.__graph.run(__instruction__).to_ndarray().flatten(order='C').tolist()

        # if EntitySet has no members(component), we define itself as a member(component).
        if (len(__components__) == 0):
            # MATCH (n:EntitySet) WHERE n.stId = 'R-HSA-170079' AND n.speciesName='Homo sapiens' RETURN n.stId, n.displayName
            __instruction__ = "MATCH (n:EntitySet) WHERE n.stId = '" + str(
                physical_entity_id) + "' RETURN n.stId"
            __components__ = self.__graph.run(__instruction__).to_ndarray().flatten(order='C').tolist()

        return __components__

    # SimpleEntity has no components, so define itself as an attribute
    # A kind reminder that the SimpleEntity has no n.speciesName attribute
    def __get_components_of_SimpleEntity(self, physical_entity_id: str):
        # MATCH (n:SimpleEntity) WHERE n.stId = 'R-ALL-29438' RETURN n.stId, n.displayName, n.speciesName
        __instruction__ = "MATCH (n:SimpleEntity) WHERE n.stId = '" + str(
            physical_entity_id) + "' RETURN n.stId"
        __components__ = self.__graph.run(__instruction__).to_ndarray().flatten(order='C').tolist()
        return __components__

    # OtherEntity has no components, so define itself as an attribute
    # A kind reminder that the OtherEntity has no n.speciesName attribute
    def __get_components_of_OtherEntity(self, physical_entity_id: str):
        # MATCH (n:OtherEntity) WHERE n.stId = 'R-ALL-422139' RETURN n.stId, n.displayName
        __instruction__ = "MATCH (n:OtherEntity) WHERE n.stId = '" + str(
            physical_entity_id) + "' RETURN n.stId"
        __components__ = self.__graph.run(__instruction__).to_ndarray().flatten(order='C').tolist()
        return __components__

    # Drug has no components, so define itself as an attribute
    # A kind reminder that the Drug has no n.speciesName attribute
    def __get_components_of_Drug(self, physical_entity_id: str):
        # MATCH (n:Drug) WHERE n.stId = 'R-ALL-9674322' RETURN n.stId, n.displayName
        __instruction__ = "MATCH (n:Drug) WHERE n.stId = '" + str(
            physical_entity_id) + "' RETURN n.stId"
        __components__ = self.__graph.run(__instruction__).to_ndarray().flatten(order='C').tolist()
        return __components__

    # get schemaClass of physical entity
    def __get_schemaClass_of_physical_entity(self, physical_entity_id: str):
        __instruction__ = "MATCH (n:PhysicalEntity) WHERE n.stId = '" + str(
            physical_entity_id) + "' AND n.speciesName='Homo sapiens' RETURN n.schemaClass"
        __physical_entity_schemaClass__ndarray = self.__graph.run(__instruction__).to_ndarray()
        # if is None, it's possibly because that the node is An Simple Entity/OtherEntity/Drug which doesn't have n.speciesName
        if __physical_entity_schemaClass__ndarray is None or __physical_entity_schemaClass__ndarray.size == 0:
            __instruction__ = "MATCH (n:PhysicalEntity) WHERE n.stId = '" + str(
                physical_entity_id) + "' RETURN n.schemaClass"
            __physical_entity_schemaClass__ndarray = self.__graph.run(__instruction__).to_ndarray()

        __physical_entity_schemaClass = __physical_entity_schemaClass__ndarray[0, 0]

        return __physical_entity_schemaClass

    def get_components_of_single_physical_entity(self, physical_entity_id: str):
        __physical_entity_schemaClass__ = self.__get_schemaClass_of_physical_entity(physical_entity_id)
        type_of_physical_entity = self.__type_detector.get_type_of_physical_entity(__physical_entity_schemaClass__)
        # func_for_get_components_selected = self.__funcs_for_get_components.get(type_of_physical_entity)
        func_for_get_components_selected = self.__funcs_for_get_components[type_of_physical_entity]
        components = []
        if (func_for_get_components_selected is not None):
            components = func_for_get_components_selected(physical_entity_id)

        # list of physical_entity_id
        return components

    '''
        __get_unique_componets_from_physical_entities(self, physical_entities):
        input: list of physical_entity_id
        output: list of physical_entity_id(components), dict:physical_entity_id(node) -> set(physical_entity_id(component), physical_entity_id(component)....physical_entity_id(component)
        '''

    def get_component_name_by_component_id(self, component_id) -> str:
        __instruction_output = "MATCH (n:PhysicalEntity) WHERE n.stId = '" + str(
            component_id) + "' AND n.speciesName='Homo sapiens' RETURN n.displayName"

        component_name = self.__graph.run(__instruction_output).to_ndarray().flatten(order='C').tolist()
        if component_name is None or len(component_name) == 0:
            __instruction = "MATCH (n:PhysicalEntity) WHERE n.stId = '" + str(
                component_id) + "' RETURN n.displayName"
            component_name = self.__graph.run(__instruction).to_ndarray().flatten(order='C').tolist()
            if component_name is None or len(component_name) == 0:
                return ""

        component_name = component_name[0]
        component_name = re.sub(u"\\[.*?\\]", "", component_name)
        component_name = component_name.strip()
        return component_name

    # tuple[list[str], dict[str, list[str]]
    def get_unique_components_and_components_dict_from_list_of_physical_entities(self, physical_entities: list[str]) -> \
            tuple[
                list[str], dict[str, list[str]]]:
        component_ids_set = set()

        components_dict = {}

        for physical_entity_id in physical_entities:
            component_ids = self.get_components_of_single_physical_entity(
                physical_entity_id)

            if len(component_ids) == 0:
                print("error! for finding no components -> physical_entity_stId:" + str(physical_entity_id))

            for component_id in component_ids:
                component_ids_set.add(component_id)

            components_dict[physical_entity_id] = list(set(component_ids))

        component_ids_unique = list(component_ids_set)

        return component_ids_unique, components_dict

    def get_unique_components_without_duplicate_names_and_mapping_components_names_list_and_physical_entity_id_to_list_of_component_ids_dict_from_list_of_physical_entities(
            self, physical_entities) -> tuple[list[str], list[str], dict[str, list[str]]]:
        original_component_ids_unique, components_dict = self.get_unique_components_and_components_dict_from_list_of_physical_entities(
            physical_entities)

        components_names_set = set()

        # There will be multiple ids mapping to one name
        component_id_to_component_name_dict = dict()

        # This dictionary will drop some ids, ensuring id and name is a one-to-one correspondence
        component_name_to_component_id_dict_one_to_one = dict()

        original_component_ids_unique.sort()
        for component_id in original_component_ids_unique:
            component_name = self.get_component_name_by_component_id(component_id)
            components_names_set.add(component_name)
            component_id_to_component_name_dict[component_id] = component_name
            component_name_to_component_id_dict_one_to_one[component_name] = component_id

        components_names_list = list(components_names_set)

        # sort the components_names_list
        components_names_list = sorted(components_names_list)

        # return value
        unique_components_without_duplicate_names = list()

        for component_name in components_names_list:
            component_id = component_name_to_component_id_dict_one_to_one[component_name]
            unique_components_without_duplicate_names.append(component_id)

        original_component_id_to_component_id_without_duplicate_names_dict = dict()

        for original_component_id in original_component_ids_unique:
            component_name = component_id_to_component_name_dict[original_component_id]
            component_id = component_name_to_component_id_dict_one_to_one[component_name]
            original_component_id_to_component_id_without_duplicate_names_dict[original_component_id] = component_id

        # return value
        physical_entity_id_to_component_id_dict = dict()
        for physical_entity_id, list_of_component_id in components_dict.items():
            list_of_component_id_without_duplicate_names = list()
            for original_component_id in list_of_component_id:
                component_id = original_component_id_to_component_id_without_duplicate_names_dict[original_component_id]
                list_of_component_id_without_duplicate_names.append(component_id)
            physical_entity_id_to_component_id_dict[physical_entity_id] = list_of_component_id_without_duplicate_names

        return unique_components_without_duplicate_names, components_names_list, physical_entity_id_to_component_id_dict


# The processor which deal with Reactome DataBase
class ReactomeProcessor:
    def __init__(self, user_name, password):
        self.__link = "bolt://localhost:7687"

        # user_name = 'neo4j'
        self.__user_name = user_name

        # password = '123456'
        self.__password = password

        self.__graph = self.__login(self.__link, self.__user_name, self.__password)

        self.__pathway_processor = PathWayProcessor(self.__graph)

        self.__reaction_processor = ReactionProcessor(self.__graph)

        # PhysicalEntityProcessor
        self.__physical_entity_processor = PhysicalEntityProcessor(self.__graph)

    @staticmethod
    def __login(link, user_name, password):
        graph = Graph(link, auth=(user_name, password))
        return graph

    '''
    get_all_top_pathways(self):
    output: a list of pathway_id
    '''

    def get_all_top_pathways(self):
        toplevel_pathways = self.__pathway_processor.get_all_top_level_pathways_ids()
        print(toplevel_pathways)
        return toplevel_pathways

    def get_pathway_name_by_id(self, pathway_stId):
        __instruction__ = "MATCH (n:Pathway) WHERE n.stId = '" + str(pathway_stId) + "' RETURN n.displayName"
        pathways = self.__graph.run(__instruction__).to_ndarray()
        pathways = pathways.flatten(order='C')
        if (pathways.size == 1):
            return pathways[0]
        else:
            print("sorry, we can't find pathway with stId = '" + str(pathway_stId) + "'")
            return ''

    def get_reaction_name_by_id(self, reaction_stId):
        __instruction__ = "MATCH (n:Reaction) WHERE n.stId = '" + str(reaction_stId) + "' RETURN n.displayName"
        reactions = self.__graph.run(__instruction__).to_ndarray()
        reactions = reactions.flatten(order='C')
        if (reactions.size == 1):
            return reactions[0]
        else:
            print("sorry, we can't find reaction with stId = '" + str(reaction_stId) + "'")
            return ''

    def get_physical_entity_name_by_id(self, physical_entity_stId):
        __instruction__ = "MATCH (n:PhysicalEntity) WHERE n.stId = '" + str(
            physical_entity_stId) + "' RETURN n.displayName"
        physical_entities = self.__graph.run(__instruction__).to_ndarray()
        physical_entities = physical_entities.flatten(order='C')
        if (physical_entities.size == 1):
            return physical_entities[0]
        else:
            print("sorry, we can't find physical entity with stId = '" + str(physical_entity_stId) + "'")
            return ''

    def get_all_relationships_for_single_pathway(self, pathway_id):
        reactions = self.__reaction_processor.get_reactions_ids_from_single_pathway(pathway_id)
        unique_edges_for_single_pathway = self.__reaction_processor.get_all_unique_relationships_of_set_of_reactions(
            reactions)
        return unique_edges_for_single_pathway

    def generate_relationships_without_duplicate_names_based_on_original_relationships_and_dic(self,
                                                                                               original_relationships:
                                                                                               list[list[str]],
                                                                                               original_entity_id_to_no_duplicate_name_entity_id_dic:
                                                                                               dict[str, str]):

        relationship_without_duplicate_name = list()
        for original_relationship in original_relationships:
            entity_id_reaction_id_direction_list = list()

            entity_id = original_relationship[self.__reaction_processor.entity_id_index]
            entity_id_without_duplicate_name = original_entity_id_to_no_duplicate_name_entity_id_dic[entity_id]
            reaction_id = original_relationship[self.__reaction_processor.reaction_id_index]
            direction = original_relationship[self.__reaction_processor.direction_index]

            entity_id_reaction_id_direction_list.append(entity_id_without_duplicate_name)
            entity_id_reaction_id_direction_list.append(reaction_id)
            entity_id_reaction_id_direction_list.append(direction)

            relationship_without_duplicate_name.append(entity_id_reaction_id_direction_list)

        return relationship_without_duplicate_name

    def extract_edges_ids_names_and_nodes_ids_names_and_relationships_and_all_component_ids_names_and_list_of_components_of_all_entities_for_one_pathway_without_duplicate_name(
            self, pathway_stId) -> tuple[
        list[str], list[str], list[str], list[str], list[str], list[str], list[str], list[str]]:
        if pathway_stId != -1:
            # normal pathway
            print("\n")
            print("\n")
            print(
                "\033[1;36m" + "************" + self.get_pathway_name_by_id(pathway_stId) + "************" + "\033[0m")
            reaction_ids = self.__reaction_processor.get_reactions_ids_from_single_pathway(pathway_stId)
            reaction_id_list, reaction_name_list = self.__reaction_processor.get_list_of_reaction_ids_without_duplicate_name_and_mapping_list_of_reaction_name_by_list_of_reaction_ids(
                reaction_ids)

            # build a dictionary that mapping: reaction_id -> line_index
            reactions_index_dic = {reaction_id: index for index, reaction_id in enumerate(reaction_id_list)}

            # get unique physical entities for one pathway
            original_physical_entity_list, physical_entity_ids, physical_entity_names, original_entity_id_to_no_duplicate_name_entity_id_dic = self.__reaction_processor.get_original_physical_entity_list_and_list_of_unique_physical_entities_id_without_duplicate_name_and_list_of_mapping_names_and_original_physical_entity_id_to_no_duplicate_name_physical_entity_id_dic_from_list_of_reactions_ids(
                reaction_id_list)

            # build a dictionary that mapping: entity_id -> line_index
            entities_index_dic = {entity_id: index for index, entity_id in
                                  enumerate(physical_entity_ids)}

            # PhysicalEntity_id, Reaction_id, 0/1
            relationships_between_entities_and_reactions = self.__reaction_processor.get_all_unique_relationships_of_set_of_reactions(
                reaction_id_list)

            relationship_without_duplicate_name = self.generate_relationships_without_duplicate_names_based_on_original_relationships_and_dic(
                relationships_between_entities_and_reactions, original_entity_id_to_no_duplicate_name_entity_id_dic)

            # PhysicalEntity_id, Reaction_id, 0/1    -a list
            relationship_without_duplicate_name = list(set(tuple(edge) for edge in relationship_without_duplicate_name))

            # every edge from tuple to list
            relationship_without_duplicate_name = list(list(edge) for edge in relationship_without_duplicate_name)

        else:
            # we'll calculate on the whole reactome

            # The latter instruction is the old version that we will never use,
            # as there will be a small amount(100+) of reactions that belongs to no pathways
            # We just simply quit these datas

            # reactions = self.__reaction_processor.get_all_reactions_of_homo_sapiens_in_Reactome()

            top_pathways = self.__pathway_processor.get_all_top_level_pathways_ids()

            reaction_ids = self.__reaction_processor.get_all_reactions_of_homo_sapiens_based_on_all_top_pathways_in_Reactome(
                top_pathways)

            reaction_id_list, reaction_name_list = self.__reaction_processor.get_list_of_reaction_ids_without_duplicate_name_and_mapping_list_of_reaction_name_by_list_of_reaction_ids(
                reaction_ids)

            # build a dictionary that mapping: reaction_id -> reaction_index
            reactions_index_dic = {reaction_id: index for index, reaction_id in enumerate(reaction_id_list)}

            # get unique physical entities for one pathway
            original_physical_entity_list, physical_entity_ids, physical_entity_names, original_entity_id_to_no_duplicate_name_entity_id_dic = self.__reaction_processor.get_original_physical_entity_list_and_list_of_unique_physical_entities_id_without_duplicate_name_and_list_of_mapping_names_and_original_physical_entity_id_to_no_duplicate_name_physical_entity_id_dic_from_list_of_reactions_ids(
                reaction_id_list)

            # build a dictionary that mapping: entity_id -> entity_index
            entities_index_dic = {entity_id: index for index, entity_id in
                                  enumerate(physical_entity_ids)}

            # PhysicalEntity_id, Reaction_id, 0/1
            relationships_between_entities_and_reactions = self.__reaction_processor.get_all_unique_relationships_of_set_of_reactions(
                reaction_id_list)

            relationship_without_duplicate_name = self.generate_relationships_without_duplicate_names_based_on_original_relationships_and_dic(
                relationships_between_entities_and_reactions, original_entity_id_to_no_duplicate_name_entity_id_dic)

            # PhysicalEntity_id, Reaction_id, 0/1    -a list
            relationship_without_duplicate_name = list(set(tuple(edge) for edge in relationship_without_duplicate_name))

            # every edge from tuple to list
            relationship_without_duplicate_name = list(list(edge) for edge in relationship_without_duplicate_name)

        # out of the if / else.
        relationships_between_nodes_and_edges_with_index_style = list()
        # relationship: node_id,reaction_id,direction(-1 or 1)
        for relationship in relationship_without_duplicate_name:
            # node_index,reaction_index,direction(-1 or 1)
            line_message = ""
            entity_id = relationship[self.__reaction_processor.entity_id_index]
            entity_index = entities_index_dic[entity_id]

            reaction_id = relationship[self.__reaction_processor.reaction_id_index]
            reaction_index = reactions_index_dic[reaction_id]

            direction = relationship[self.__reaction_processor.direction_index]

            line_message = line_message + str(entity_index) + "," + str(reaction_index) + "," + str(direction)

            relationships_between_nodes_and_edges_with_index_style.append(line_message)

        relationships_between_nodes_and_edges_with_index_style.sort(key=lambda l: (int(re.findall('\d+', l)[1]), int(re.findall('\d+', l)[0]), int(re.findall('-?\d+', l)[2])))

        unique_components_without_duplicate_names, component_names_list, physical_entity_id_to_list_of_component_ids_dict = self.__physical_entity_processor.get_unique_components_without_duplicate_names_and_mapping_components_names_list_and_physical_entity_id_to_list_of_component_ids_dict_from_list_of_physical_entities(
            physical_entity_ids)

        # build a dictionary that mapping: component_id -> line_index
        components_index_dic = {component_id: index for index, component_id in
                                enumerate(unique_components_without_duplicate_names)}

        entity_index_to_components_indices_mapping_list = list()

        for node_id, set_of_components in physical_entity_id_to_list_of_component_ids_dict.items():
            node_id_index = entities_index_dic[node_id]
            # component_msg = str(node_id_index) + ":"
            # we won't store the index of entity, as it's in the same sequence with the data in nodes.txt
            component_msg = ""
            list_of_components = []
            for component_id in set_of_components:
                component_id_index = components_index_dic[component_id]
                list_of_components.append(component_id_index)

            list_of_components = sorted(list_of_components)

            for component_id_index in list_of_components:
                component_msg = component_msg + str(component_id_index) + ","

            # remove the comma in the end
            component_msg = component_msg[:-1]
            entity_index_to_components_indices_mapping_list.append(component_msg)

        num_of_edges = str(len(reaction_name_list))
        num_of_nodes = str(len(physical_entity_names))
        dimensionality = str(len(component_names_list))

        print("reactions(hyper edges): " + num_of_edges)
        print("physical entities(nodes): " + num_of_nodes)
        print("physical entities dimensionality(attributes): " + dimensionality)
        print("\n")

        return reaction_id_list, reaction_name_list, physical_entity_ids, physical_entity_names, relationships_between_nodes_and_edges_with_index_style, unique_components_without_duplicate_names, component_names_list, entity_index_to_components_indices_mapping_list

    def extract_edges_nodes_relationships_all_components_and_dic_of_entity_components_for_one_pathway(self,
                                                                                                      pathway_stId) -> \
            tuple[list, list, list, list, list]:

        """
        extract
        input: pathway_stId: the id of a pathway
        output: reaction_ids of the pathway, node_ids of the pathway, component_ids of all the nodes of the pathway, dictionary of node_id to a set of its component_ids

        """

        if pathway_stId != -1:
            # normal pathway
            print("\n")
            print("\n")
            print(
                "\033[1;36m" + "************" + self.get_pathway_name_by_id(pathway_stId) + "************" + "\033[0m")

            reactions = self.__reaction_processor.get_reactions_ids_from_single_pathway(pathway_stId)

            # build a dictionary that mapping: reaction_id -> line_index
            reactions_index_dic = {reaction_id: index for index, reaction_id in enumerate(reactions)}

            # get unique physical entities for one pathway
            physical_entity_ids_from_reactions_for_one_pathway = self.__reaction_processor.get_unique_physical_entities_ids_from_list_of_reactions_ids(
                reactions)

            # build a dictionary that mapping: entity_id -> line_index
            entities_index_dic = {entity_id: index for index, entity_id in
                                  enumerate(physical_entity_ids_from_reactions_for_one_pathway)}

            relationships_between_nodes_edges = self.get_all_relationships_for_single_pathway(pathway_stId)

        else:
            # we'll calculate on the whole reactome

            # The latter instruction is the old version that we will never use,
            # as there will be a small amount(100+) of reactions that belongs to no pathways
            # We just simply quit these datas

            # reactions = self.__reaction_processor.get_all_reactions_of_homo_sapiens_in_Reactome()

            top_pathways = self.__pathway_processor.get_all_top_level_pathways_ids()

            reactions = self.__reaction_processor.get_all_reactions_of_homo_sapiens_based_on_all_top_pathways_in_Reactome(
                top_pathways)

            # build a dictionary that mapping: reaction_id -> reaction_index
            reactions_index_dic = {reaction_id: index for index, reaction_id in enumerate(reactions)}

            physical_entity_ids_from_reactions_for_one_pathway = self.__reaction_processor.get_unique_physical_entities_ids_from_list_of_reactions_ids(
                reactions)

            # build a dictionary that mapping: entity_id -> entity_index
            entities_index_dic = {entity_id: index for index, entity_id in
                                  enumerate(physical_entity_ids_from_reactions_for_one_pathway)}

            relationships_between_nodes_edges = self.__reaction_processor.get_all_unique_relationships_of_set_of_reactions(
                reactions)

        relationships_between_nodes_and_edges_with_index_style = []

        # relationship: node_id,reaction_id,direction(-1 or 1)
        for relationship in relationships_between_nodes_edges:
            # node_index,reaction_index,direction(-1 or 1)
            line_message = ""
            entity_id = relationship[self.__reaction_processor.entity_id_index]
            entity_index = entities_index_dic[entity_id]

            reaction_id = relationship[self.__reaction_processor.reaction_id_index]
            reaction_index = reactions_index_dic[reaction_id]

            direction = relationship[self.__reaction_processor.direction_index]

            line_message = line_message + str(entity_index) + "," + str(reaction_index) + "," + str(direction)

            relationships_between_nodes_and_edges_with_index_style.append(line_message)

        relationships_between_nodes_and_edges_with_index_style.sort(key=lambda l: (int(re.findall('\d+', l)[1]), int(re.findall('\d+', l)[0]), int(re.findall('-?\d+', l)[2])))

        # remove the duplicate components
        component_ids_unique_for_one_pathway, components_dic = self.__physical_entity_processor.get_unique_components_and_components_dict_from_list_of_physical_entities(
            physical_entity_ids_from_reactions_for_one_pathway)

        # build a dictionary that mapping: component_id -> line_index
        components_index_dic = {component_id: index for index, component_id in
                                enumerate(component_ids_unique_for_one_pathway)}

        # dela with the components message like -
        # a list of ["node_id:component_id,component_id,component_id..", "node_id:component_id,component_id,component_id.."]
        entity_index_to_components_indices_mapping_list = []
        for node_id, set_of_components in components_dic.items():
            node_id_index = entities_index_dic[node_id]
            # component_msg = str(node_id_index) + ":"
            # we won't store the index of entity, as it's in the same sequence with the data in nodes.txt
            component_msg = ""
            list_of_components = []
            for component_id in set_of_components:
                component_id_index = components_index_dic[component_id]
                list_of_components.append(component_id_index)

            list_of_components = sorted(list_of_components)

            for component_id_index in list_of_components:
                component_msg = component_msg + str(component_id_index) + ","

            # remove the comma in the end
            component_msg = component_msg[:-1]
            entity_index_to_components_indices_mapping_list.append(component_msg)

        num_of_edges = str(len(reactions))
        num_of_nodes = str(len(physical_entity_ids_from_reactions_for_one_pathway))
        dimensionality = str(len(component_ids_unique_for_one_pathway))

        print("reactions(hyper edges): " + num_of_edges)
        print("physical entities(nodes): " + num_of_nodes)
        print("physical entities dimensionality(attributes): " + dimensionality)
        print("\n")

        return reactions, physical_entity_ids_from_reactions_for_one_pathway, relationships_between_nodes_and_edges_with_index_style, component_ids_unique_for_one_pathway, entity_index_to_components_indices_mapping_list

    def get_reactions_index_to_list_of_relationships_dic_based_on_relationships(self, relationships: list[str]) -> \
            tuple[dict[str, list], dict[str, list], dict[str, list], dict[str, list]]:
        # reactions: a list of reaction_id
        # relationships: a list of {entity_index, reaction_index, direction}
        reaction_index_to_list_of_relationships_dic: dict[str, list] = {}
        reaction_index_to_list_of_input_relationships_dic: dict[str, list] = {}
        reaction_index_to_list_of_output_relationships_dic: dict[str, list] = {}
        reaction_index_to_list_of_regulation_relationship_dic: dict[str, list] = {}
        for relationship in relationships:
            line_elements = relationship.split(",")
            entity_index = line_elements[self.__reaction_processor.entity_id_index]
            reaction_index = line_elements[self.__reaction_processor.reaction_id_index]
            direction = line_elements[self.__reaction_processor.direction_index]
            if reaction_index in reaction_index_to_list_of_relationships_dic.keys():
                list_of_relationships = reaction_index_to_list_of_relationships_dic[reaction_index]
                list_of_relationships.append(relationship)
            else:
                reaction_index_to_list_of_relationships_dic[reaction_index] = list()
                reaction_index_to_list_of_relationships_dic[reaction_index].append(relationship)

            if int(eval(direction)) < 0:
                if reaction_index in reaction_index_to_list_of_input_relationships_dic.keys():
                    list_of_input_relationships = reaction_index_to_list_of_input_relationships_dic[reaction_index]
                    list_of_input_relationships.append(relationship)
                else:
                    reaction_index_to_list_of_input_relationships_dic[reaction_index] = list()
                    reaction_index_to_list_of_input_relationships_dic[reaction_index].append(relationship)

            elif int(eval(direction)) > 0:
                if reaction_index in reaction_index_to_list_of_output_relationships_dic.keys():
                    list_of_output_relationships = reaction_index_to_list_of_output_relationships_dic[reaction_index]
                    list_of_output_relationships.append(relationship)
                else:
                    reaction_index_to_list_of_output_relationships_dic[reaction_index] = list()
                    reaction_index_to_list_of_output_relationships_dic[reaction_index].append(relationship)

            # == 0
            else:
                if reaction_index in reaction_index_to_list_of_regulation_relationship_dic.keys():
                    list_of_regulation_relationships = reaction_index_to_list_of_regulation_relationship_dic[
                        reaction_index]
                    list_of_regulation_relationships.append(relationship)
                else:
                    reaction_index_to_list_of_regulation_relationship_dic[reaction_index] = list()
                    reaction_index_to_list_of_regulation_relationship_dic[reaction_index].append(relationship)

        return reaction_index_to_list_of_relationships_dic, reaction_index_to_list_of_input_relationships_dic, reaction_index_to_list_of_output_relationships_dic, reaction_index_to_list_of_regulation_relationship_dic

    def get_reaction_status_dic(self, reaction_index_to_list_of_relationships_dic) -> {str: int}:
        reaction_to_relationship_status_dic: {str: int} = {"total_num_of_reactions": 0,
                                                           "num_of_reactions_with_one_relationship": 0,
                                                           "num_of_reactions_with_two_relationships": 0,
                                                           "num_of_reactions_with_three_relationships": 0,
                                                           "num_of_reactions_with_four_relationships": 0,
                                                           "num_of_reactions_with_five_relationships": 0,
                                                           "num_of_reactions_with_six_relationships": 0,
                                                           "num_of_reactions_with_seven_relationships": 0,
                                                           "num_of_reactions_with_eight_relationships": 0,
                                                           "num_of_reactions_with_more_than_eight_relationships": 0}

        dic_key_name: {int: str} = {1: "num_of_reactions_with_one_relationship",
                                    2: "num_of_reactions_with_two_relationships",
                                    3: "num_of_reactions_with_three_relationships",
                                    4: "num_of_reactions_with_four_relationships",
                                    5: "num_of_reactions_with_five_relationships",
                                    6: "num_of_reactions_with_six_relationships",
                                    7: "num_of_reactions_with_seven_relationships",
                                    8: "num_of_reactions_with_eight_relationships"}

        reaction_to_relationship_status_dic["total_num_of_reactions"] = len(reaction_index_to_list_of_relationships_dic)

        for reaction_index, list_of_relationships in reaction_index_to_list_of_relationships_dic.items():
            num_of_relationships = len(list_of_relationships)
            if num_of_relationships in dic_key_name.keys():
                key_name = dic_key_name.get(num_of_relationships)
                temp_val = reaction_to_relationship_status_dic.get(key_name)
                reaction_to_relationship_status_dic[dic_key_name.get(len(list_of_relationships))] = temp_val + 1
            else:
                temp_val = reaction_to_relationship_status_dic.get(
                    "num_of_reactions_with_more_than_eight_relationships")
                reaction_to_relationship_status_dic[
                    "num_of_reactions_with_more_than_eight_relationships"] = temp_val + 1

        return reaction_to_relationship_status_dic

    def print_reaction_status_dic(self, reaction_to_relationship_status_dic: {str: int}, mode: str = ""):
        """

        :param reaction_to_relationship_status_dic:
        :param mode:
        :return:
        """
        if "input" == mode:
            mode_message = "input" + " "
        elif "output" == mode:
            mode_message = "output" + " "
        elif "regulation" == mode:
            mode_message = "regulation" + " "
        else:
            mode_message = ""

        total_num = reaction_to_relationship_status_dic.get("total_num_of_reactions")
        reaction_num_with_one_rela = reaction_to_relationship_status_dic.get("num_of_reactions_with_one_relationship")
        reaction_num_with_two_rela = reaction_to_relationship_status_dic.get("num_of_reactions_with_two_relationships")
        reaction_num_with_three_rela = reaction_to_relationship_status_dic.get(
            "num_of_reactions_with_three_relationships")
        reaction_num_with_four_rela = reaction_to_relationship_status_dic.get(
            "num_of_reactions_with_four_relationships")
        reaction_num_with_five_rela = reaction_to_relationship_status_dic.get(
            "num_of_reactions_with_five_relationships")
        reaction_num_with_six_rela = reaction_to_relationship_status_dic.get("num_of_reactions_with_six_relationships")
        reaction_num_with_seven_rela = reaction_to_relationship_status_dic.get(
            "num_of_reactions_with_seven_relationships")
        reaction_num_with_eight_rela = reaction_to_relationship_status_dic.get(
            "num_of_reactions_with_eight_relationships")
        reaction_num_with_more_than_eight_rela = reaction_to_relationship_status_dic.get(
            "num_of_reactions_with_more_than_eight_relationships")

        if total_num == 0:
            print("total num of reactions with " + mode_message + "relationship: " + str(total_num))
        else:
            print("total num of reactions with " + mode_message + "relationship: " + str(total_num))
            print("num of reactions with one " + mode_message + "node: " + str(
                reaction_num_with_one_rela) + " ( {:.2%}".format(
                float(reaction_num_with_one_rela) / float(total_num)) + ")")
            print("num of reactions with two " + mode_message + "nodes: " + str(
                reaction_num_with_two_rela) + " ( {:.2%}".format(
                float(reaction_num_with_two_rela) / float(total_num)) + ")")
            print("num of reactions with three " + mode_message + "nodes: " + str(
                reaction_num_with_three_rela) + " ( {:.2%}".format(
                float(reaction_num_with_three_rela) / float(total_num)) + ")")
            print("num of reactions with four " + mode_message + "nodes: " + str(
                reaction_num_with_four_rela) + " ( {:.2%}".format(
                float(reaction_num_with_four_rela) / float(total_num)) + ")")
            print("num of reactions with five " + mode_message + "nodes: " + str(
                reaction_num_with_five_rela) + " ( {:.2%}".format(
                float(reaction_num_with_five_rela) / float(total_num)) + ")")
            print("num of reactions with six " + mode_message + "nodes: " + str(
                reaction_num_with_six_rela) + " ( {:.2%}".format(
                float(reaction_num_with_six_rela) / float(total_num)) + ")")
            print("num of reactions with seven " + mode_message + "nodes: " + str(
                reaction_num_with_seven_rela) + " ( {:.2%}".format(
                float(reaction_num_with_seven_rela) / float(total_num)) + ")")
            print("num of reactions with eight " + mode_message + "nodes: " + str(
                reaction_num_with_eight_rela) + " ( {:.2%}".format(
                float(reaction_num_with_eight_rela) / float(total_num)) + ")")
            print("num of reactions with more than eight " + mode_message + "nodes: " + str(
                reaction_num_with_more_than_eight_rela) + " ( {:.2%}".format(
                float(reaction_num_with_more_than_eight_rela) / float(total_num)) + ")")

    def get_entity_index_to_list_of_relationships_dic_based_on_relationships(self, relationships: list[str]) -> dict[
        str, list]:
        entity_index_to_list_of_relationships_dic: dict[str, list] = {}

        for relationship in relationships:
            line_elements = relationship.split(",")
            entity_index = line_elements[self.__reaction_processor.entity_id_index]
            reaction_index = line_elements[self.__reaction_processor.reaction_id_index]
            direction = line_elements[self.__reaction_processor.direction_index]
            if entity_index in entity_index_to_list_of_relationships_dic.keys():
                list_of_relationships = entity_index_to_list_of_relationships_dic[entity_index]
                list_of_relationships.append(relationship)
            else:
                entity_index_to_list_of_relationships_dic[entity_index] = list()
                entity_index_to_list_of_relationships_dic[entity_index].append(relationship)

        return entity_index_to_list_of_relationships_dic

    def get_entity_status_dic(self, entity_index_to_list_of_relationships_dic) -> {str: int}:
        entity_to_relationship_status_dic: {str: int} = {"total_num_of_entities": 0,
                                                         "num_of_entities_with_one_relationship": 0,
                                                         "num_of_entities_with_two_relationships": 0,
                                                         "num_of_entities_with_three_relationships": 0,
                                                         "num_of_entities_with_four_relationships": 0,
                                                         "num_of_entities_with_five_relationships": 0,
                                                         "num_of_entities_with_six_relationships": 0,
                                                         "num_of_entities_with_seven_relationships": 0,
                                                         "num_of_entities_with_eight_relationships": 0,
                                                         "num_of_entities_with_more_than_eight_relationships": 0}

        dic_key_name: {int: str} = {1: "num_of_entities_with_one_relationship",
                                    2: "num_of_entities_with_two_relationships",
                                    3: "num_of_entities_with_three_relationships",
                                    4: "num_of_entities_with_four_relationships",
                                    5: "num_of_entities_with_five_relationships",
                                    6: "num_of_entities_with_six_relationships",
                                    7: "num_of_entities_with_seven_relationships",
                                    8: "num_of_entities_with_eight_relationships"}

        entity_to_relationship_status_dic["total_num_of_entities"] = len(entity_index_to_list_of_relationships_dic)

        for entity_index, list_of_relationships in entity_index_to_list_of_relationships_dic.items():
            num_of_relationships = len(list_of_relationships)
            if num_of_relationships in dic_key_name.keys():
                key_name = dic_key_name.get(num_of_relationships)
                temp_val = entity_to_relationship_status_dic.get(key_name)
                entity_to_relationship_status_dic[dic_key_name.get(len(list_of_relationships))] = temp_val + 1
            else:
                temp_val = entity_to_relationship_status_dic.get(
                    "num_of_entities_with_more_than_eight_relationships")
                entity_to_relationship_status_dic[
                    "num_of_entities_with_more_than_eight_relationships"] = temp_val + 1

        return entity_to_relationship_status_dic

    def print_entity_status_dic(self, entity_status_dic):
        total_num = entity_status_dic.get("total_num_of_entities")
        entity_num_with_one_rela = entity_status_dic.get("num_of_entities_with_one_relationship")
        entity_num_with_two_rela = entity_status_dic.get("num_of_entities_with_two_relationships")
        entity_num_with_three_rela = entity_status_dic.get(
            "num_of_entities_with_three_relationships")
        entity_num_with_four_rela = entity_status_dic.get(
            "num_of_entities_with_four_relationships")
        entity_num_with_five_rela = entity_status_dic.get(
            "num_of_entities_with_five_relationships")
        entity_num_with_six_rela = entity_status_dic.get("num_of_entities_with_six_relationships")
        entity_num_with_seven_rela = entity_status_dic.get(
            "num_of_entities_with_seven_relationships")
        entity_num_with_eight_rela = entity_status_dic.get(
            "num_of_entities_with_eight_relationships")
        entity_num_with_more_than_eight_rela = entity_status_dic.get(
            "num_of_entities_with_more_than_eight_relationships")

        print("total num of entities: " + str(total_num))
        print("num of entities with one reaction: " + str(
            entity_num_with_one_rela) + " ( {:.2%}".format(
            float(entity_num_with_one_rela) / float(total_num)) + ")")
        print("num of entities with two reactions: " + str(
            entity_num_with_two_rela) + " ( {:.2%}".format(
            float(entity_num_with_two_rela) / float(total_num)) + ")")
        print("num of entities with three reactions: " + str(
            entity_num_with_three_rela) + " ( {:.2%}".format(
            float(entity_num_with_three_rela) / float(total_num)) + ")")
        print("num of entities with four reactions: " + str(
            entity_num_with_four_rela) + " ( {:.2%}".format(
            float(entity_num_with_four_rela) / float(total_num)) + ")")
        print("num of entities with five reactions: " + str(
            entity_num_with_five_rela) + " ( {:.2%}".format(
            float(entity_num_with_five_rela) / float(total_num)) + ")")
        print("num of entities with six reactions: " + str(
            entity_num_with_six_rela) + " ( {:.2%}".format(
            float(entity_num_with_six_rela) / float(total_num)) + ")")
        print("num of entities with seven reactions: " + str(
            entity_num_with_seven_rela) + " ( {:.2%}".format(
            float(entity_num_with_seven_rela) / float(total_num)) + ")")
        print("num of entities with eight reactions: " + str(
            entity_num_with_eight_rela) + " ( {:.2%}".format(
            float(entity_num_with_eight_rela) / float(total_num)) + ")")
        print("num of entities with more than eight reactions: " + str(
            entity_num_with_more_than_eight_rela) + " ( {:.2%}".format(
            float(entity_num_with_more_than_eight_rela) / float(total_num)) + ")")

    def get_entity_index_to_list_of_components_dic_based_on_entity_component_mapping_list(self,
                                                                                          entity_component_mapping_list:
                                                                                          list[str]):
        entity_index_to_list_of_components_dic: dict[int, list[int]] = {}
        entity_index = 0
        for line_element in entity_component_mapping_list:
            components_str_indexes = line_element.split(",")
            components_indexes = []

            for component_index in components_str_indexes:
                component_index = int(component_index)
                components_indexes.append(component_index)
            entity_index_to_list_of_components_dic[entity_index] = components_indexes

            entity_index = entity_index + 1

        return entity_index_to_list_of_components_dic

    def get_entity_components_status_dic_based_on_entity_index_to_list_of_components_dic(self,
                                                                                         entity_index_to_list_of_components_dic):
        entity_components_status_dic: {str: int} = {"total_num_of_entities": 0,
                                                    "num_of_entities_with_one_component": 0,
                                                    "num_of_entities_with_two_components": 0,
                                                    "num_of_entities_with_three_components": 0,
                                                    "num_of_entities_with_four_components": 0,
                                                    "num_of_entities_with_five_components": 0,
                                                    "num_of_entities_with_six_components": 0,
                                                    "num_of_entities_with_seven_components": 0,
                                                    "num_of_entities_with_eight_components": 0,
                                                    "num_of_entities_with_more_than_eight_components": 0}

        dic_key_name: {int: str} = {1: "num_of_entities_with_one_component",
                                    2: "num_of_entities_with_two_components",
                                    3: "num_of_entities_with_three_components",
                                    4: "num_of_entities_with_four_components",
                                    5: "num_of_entities_with_five_components",
                                    6: "num_of_entities_with_six_components",
                                    7: "num_of_entities_with_seven_components",
                                    8: "num_of_entities_with_eight_components"}

        entity_components_status_dic["total_num_of_entities"] = len(entity_index_to_list_of_components_dic)

        for entity_index, list_of_components in entity_index_to_list_of_components_dic.items():
            num_of_components = len(list_of_components)
            if num_of_components in dic_key_name.keys():
                key_name = dic_key_name.get(num_of_components)
                temp_val = entity_components_status_dic.get(key_name)
                entity_components_status_dic[dic_key_name.get(len(list_of_components))] = temp_val + 1
            else:
                temp_val = entity_components_status_dic.get(
                    "num_of_entities_with_more_than_eight_components")
                entity_components_status_dic[
                    "num_of_entities_with_more_than_eight_components"] = temp_val + 1

        return entity_components_status_dic

    def print_entity_components_distribution_dic(self, entity_components_status_dic):
        total_num = entity_components_status_dic.get("total_num_of_entities")
        entity_num_with_one_component = entity_components_status_dic.get("num_of_entities_with_one_component")
        entity_num_with_two_components = entity_components_status_dic.get("num_of_entities_with_two_components")
        entity_num_with_three_components = entity_components_status_dic.get(
            "num_of_entities_with_three_components")
        entity_num_with_four_components = entity_components_status_dic.get(
            "num_of_entities_with_four_components")
        entity_num_with_five_components = entity_components_status_dic.get(
            "num_of_entities_with_five_components")
        entity_num_with_six_components = entity_components_status_dic.get("num_of_entities_with_six_components")
        entity_num_with_seven_components = entity_components_status_dic.get(
            "num_of_entities_with_seven_components")
        entity_num_with_eight_components = entity_components_status_dic.get(
            "num_of_entities_with_eight_components")
        entity_num_with_more_than_eight_components = entity_components_status_dic.get(
            "num_of_entities_with_more_than_eight_components")

        print("total num of entities: " + str(total_num))
        print("num of entities with one component: " + str(
            entity_num_with_one_component) + " ( {:.2%}".format(
            float(entity_num_with_one_component) / float(total_num)) + ")")
        print("num of entities with two components: " + str(
            entity_num_with_two_components) + " ( {:.2%}".format(
            float(entity_num_with_two_components) / float(total_num)) + ")")
        print("num of entities with three components: " + str(
            entity_num_with_three_components) + " ( {:.2%}".format(
            float(entity_num_with_three_components) / float(total_num)) + ")")
        print("num of entities with four components: " + str(
            entity_num_with_four_components) + " ( {:.2%}".format(
            float(entity_num_with_four_components) / float(total_num)) + ")")
        print("num of entities with five components: " + str(
            entity_num_with_five_components) + " ( {:.2%}".format(
            float(entity_num_with_five_components) / float(total_num)) + ")")
        print("num of entities with six components: " + str(
            entity_num_with_six_components) + " ( {:.2%}".format(
            float(entity_num_with_six_components) / float(total_num)) + ")")
        print("num of entities with seven components: " + str(
            entity_num_with_seven_components) + " ( {:.2%}".format(
            float(entity_num_with_seven_components) / float(total_num)) + ")")
        print("num of entities with eight components: " + str(
            entity_num_with_eight_components) + " ( {:.2%}".format(
            float(entity_num_with_eight_components) / float(total_num)) + ")")
        print("num of entities with more than eight components: " + str(
            entity_num_with_more_than_eight_components) + " ( {:.2%}".format(
            float(entity_num_with_more_than_eight_components) / float(total_num)) + ")")

    def execution_on_single_pathways(self, pathway_stId):
        pathway_name = self.get_pathway_name_by_id(pathway_stId)

        reactions, physical_entity_ids, relationships_between_nodes_edges, component_ids, entity_component_mapping_list = self.extract_edges_nodes_relationships_all_components_and_dic_of_entity_components_for_one_pathway(
            pathway_stId)

        # calculate the data distribution
        reaction_index_to_list_of_relationships_dic, reaction_index_to_list_of_input_relationships_dic, reaction_index_to_list_of_output_relationships_dic, reaction_index_to_list_of_regulation_relationship_dic = self.get_reactions_index_to_list_of_relationships_dic_based_on_relationships(
            relationships_between_nodes_edges)

        reaction_to_relationship_status_dic = self.get_reaction_status_dic(reaction_index_to_list_of_relationships_dic)
        reaction_to_input_relationship_status_dic = self.get_reaction_status_dic(
            reaction_index_to_list_of_input_relationships_dic)
        reaction_to_output_relationship_status_dic = self.get_reaction_status_dic(
            reaction_index_to_list_of_output_relationships_dic)
        reaction_to_regulation_relationship_status_dic = self.get_reaction_status_dic(
            reaction_index_to_list_of_regulation_relationship_dic)

        print("\033[1;32m" + "For all the relationships:" + "\033[0m")
        self.print_reaction_status_dic(reaction_to_relationship_status_dic)
        print("\n")

        print("\033[1;32m" + "For input relationship:" + "\033[0m")
        self.print_reaction_status_dic(reaction_to_input_relationship_status_dic, mode="input")
        print("\n")

        print("\033[1;32m" + "For output relationship:" + "\033[0m")
        self.print_reaction_status_dic(reaction_to_output_relationship_status_dic, mode="output")
        print("\n")

        print("\033[1;32m" + "For regulation relationship:" + "\033[0m")
        self.print_reaction_status_dic(reaction_to_regulation_relationship_status_dic, mode="regulation")
        print("\n")

        entity_index_to_list_of_relationships_dic = self.get_entity_index_to_list_of_relationships_dic_based_on_relationships(
            relationships_between_nodes_edges)
        entity_to_relationship_status_dic = self.get_entity_status_dic(entity_index_to_list_of_relationships_dic)

        print("\033[1;32m" + "For entities:" + "\033[0m")
        self.print_entity_status_dic(entity_to_relationship_status_dic)
        print("\n")

        entity_index_to_list_of_components_dic = self.get_entity_index_to_list_of_components_dic_based_on_entity_component_mapping_list(
            entity_component_mapping_list)
        entity_components_status_dic = self.get_entity_components_status_dic_based_on_entity_index_to_list_of_components_dic(
            entity_index_to_list_of_components_dic)

        print("\033[1;32m" + "For entities and components:" + "\033[0m")
        self.print_entity_components_distribution_dic(entity_components_status_dic)
        print("\n")

        # store data into a txt file
        file_processor = FileProcessor()
        file_processor.execute_for_single_pathway(pathway_name, reactions, physical_entity_ids,
                                                  relationships_between_nodes_edges, component_ids,
                                                  entity_component_mapping_list)

        # draw the histogram
        # drawer = Drawer(len(reactions), len(physical_entity_ids), len(component_ids), pathway_name)
        # drawer.generate_histogram()

    def execution_on_single_pathways_enhanced(self, pathway_stId):
        pathway_name = self.get_pathway_name_by_id(pathway_stId)

        reaction_ids, reaction_names, physical_entity_ids, physical_entity_names, relationships_between_nodes_edges, component_ids, component_names, entity_component_mapping_list = self.extract_edges_ids_names_and_nodes_ids_names_and_relationships_and_all_component_ids_names_and_list_of_components_of_all_entities_for_one_pathway_without_duplicate_name(
            pathway_stId)

        # calculate the data distribution
        reaction_index_to_list_of_relationships_dic, reaction_index_to_list_of_input_relationships_dic, reaction_index_to_list_of_output_relationships_dic, reaction_index_to_list_of_regulation_relationship_dic = self.get_reactions_index_to_list_of_relationships_dic_based_on_relationships(
            relationships_between_nodes_edges)

        reaction_to_relationship_status_dic = self.get_reaction_status_dic(reaction_index_to_list_of_relationships_dic)
        reaction_to_input_relationship_status_dic = self.get_reaction_status_dic(
            reaction_index_to_list_of_input_relationships_dic)
        reaction_to_output_relationship_status_dic = self.get_reaction_status_dic(
            reaction_index_to_list_of_output_relationships_dic)
        reaction_to_regulation_relationship_status_dic = self.get_reaction_status_dic(
            reaction_index_to_list_of_regulation_relationship_dic)

        print("\033[1;32m" + "For all the relationships:" + "\033[0m")
        self.print_reaction_status_dic(reaction_to_relationship_status_dic)
        print("\n")

        print("\033[1;32m" + "For input relationship:" + "\033[0m")
        self.print_reaction_status_dic(reaction_to_input_relationship_status_dic, mode="input")
        print("\n")

        print("\033[1;32m" + "For output relationship:" + "\033[0m")
        self.print_reaction_status_dic(reaction_to_output_relationship_status_dic, mode="output")
        print("\n")

        print("\033[1;32m" + "For regulation relationship:" + "\033[0m")
        self.print_reaction_status_dic(reaction_to_regulation_relationship_status_dic, mode="regulation")
        print("\n")

        entity_index_to_list_of_relationships_dic = self.get_entity_index_to_list_of_relationships_dic_based_on_relationships(
            relationships_between_nodes_edges)
        entity_to_relationship_status_dic = self.get_entity_status_dic(entity_index_to_list_of_relationships_dic)

        print("\033[1;32m" + "For entities:" + "\033[0m")
        self.print_entity_status_dic(entity_to_relationship_status_dic)
        print("\n")

        entity_index_to_list_of_components_dic = self.get_entity_index_to_list_of_components_dic_based_on_entity_component_mapping_list(
            entity_component_mapping_list)
        entity_components_status_dic = self.get_entity_components_status_dic_based_on_entity_index_to_list_of_components_dic(
            entity_index_to_list_of_components_dic)

        print("\033[1;32m" + "For entities and components:" + "\033[0m")
        self.print_entity_components_distribution_dic(entity_components_status_dic)
        print("\n")

        # store data into a txt file
        file_processor = FileProcessor()
        file_processor.execute_for_single_pathway_with_name_files(pathway_name, reaction_ids, reaction_names,
                                                                  physical_entity_ids, physical_entity_names,
                                                                  relationships_between_nodes_edges, component_ids,
                                                                  component_names,
                                                                  entity_component_mapping_list)

        # draw the histogram
        # drawer = Drawer(len(reaction_names), len(physical_entity_names), len(component_names), pathway_name)
        # drawer.generate_histogram()


    def execution_on_single_pathway_via_name_enhanced(self, pathway_name: str):
        pathway_id = self.__pathway_processor.get_top_level_pathway_id_by_name(pathway_name)
        self.execution_on_single_pathways_enhanced(pathway_id)

    def execution_on_all_pathways(self):
        top_pathways = self.get_all_top_pathways()
        for top_pathway_id in top_pathways:
            self.execution_on_single_pathways(top_pathway_id)

    def execution_on_all_pathways_enhanced(self):
        top_pathways = self.get_all_top_pathways()
        for top_pathway_id in top_pathways:
            self.execution_on_single_pathways_enhanced(top_pathway_id)

    def execution_on_reactome(self):

        reactions, physical_entities, relationships_with_index_style, components, entity_index_to_components_indices_mapping_list = self.extract_edges_nodes_relationships_all_components_and_dic_of_entity_components_for_one_pathway(
            -1)

        num_of_edges = str(len(reactions))
        num_of_nodes = str(len(physical_entities))
        dimensionality = str(len(components))

        print("************Reactome************")

        print("reactions(hyper edges): " + num_of_edges)
        print("physical entities(nodes): " + num_of_nodes)
        print("physical entities dimensionality(attributes): " + dimensionality)
        print("\n")

        # calculate the data distribution
        reaction_index_to_list_of_relationships_dic, reaction_index_to_list_of_input_relationships_dic, reaction_index_to_list_of_output_relationships_dic, reaction_index_to_list_of_regulation_relationship_dic = self.get_reactions_index_to_list_of_relationships_dic_based_on_relationships(
            relationships_with_index_style)

        reaction_to_relationship_status_dic = self.get_reaction_status_dic(reaction_index_to_list_of_relationships_dic)
        reaction_to_input_relationship_status_dic = self.get_reaction_status_dic(
            reaction_index_to_list_of_input_relationships_dic)
        reaction_to_output_relationship_status_dic = self.get_reaction_status_dic(
            reaction_index_to_list_of_output_relationships_dic)
        reaction_to_regulation_relationship_status_dic = self.get_reaction_status_dic(
            reaction_index_to_list_of_regulation_relationship_dic)

        print("\033[1;32m" + "For all the relationships:" + "\033[0m")
        self.print_reaction_status_dic(reaction_to_relationship_status_dic)
        print("\n")

        print("\033[1;32m" + "For input relationship:" + "\033[0m")
        self.print_reaction_status_dic(reaction_to_input_relationship_status_dic, mode="input")
        print("\n")

        print("\033[1;32m" + "For output relationship:" + "\033[0m")
        self.print_reaction_status_dic(reaction_to_output_relationship_status_dic, mode="output")
        print("\n")

        print("\033[1;32m" + "For regulation relationship:" + "\033[0m")
        self.print_reaction_status_dic(reaction_to_regulation_relationship_status_dic, mode="regulation")
        print("\n")

        entity_index_to_list_of_relationships_dic = self.get_entity_index_to_list_of_relationships_dic_based_on_relationships(
            relationships_with_index_style)
        entity_to_relationship_status_dic = self.get_entity_status_dic(entity_index_to_list_of_relationships_dic)

        print("\033[1;32m" + "For entities:" + "\033[0m")
        self.print_entity_status_dic(entity_to_relationship_status_dic)
        print("\n")

        entity_index_to_list_of_components_dic = self.get_entity_index_to_list_of_components_dic_based_on_entity_component_mapping_list(
            entity_index_to_components_indices_mapping_list)
        entity_components_status_dic = self.get_entity_components_status_dic_based_on_entity_index_to_list_of_components_dic(
            entity_index_to_list_of_components_dic)

        print("\033[1;32m" + "For entities and components:" + "\033[0m")
        self.print_entity_components_distribution_dic(entity_components_status_dic)
        print("\n")

        if not os.path.exists("./data/All_data_in_Reactome"):
            os.makedirs("./data/All_data_in_Reactome")

        # store data into a txt file
        file_processor = FileProcessor()
        file_processor.execute_for_single_pathway("All_data_in_Reactome", reactions, physical_entities,
                                                  relationships_with_index_style, components,
                                                  entity_index_to_components_indices_mapping_list)

        # draw the histogram
        # drawer = Drawer(num_of_edges, num_of_nodes, dimensionality, "All_data_in_Reactome")
        # drawer.generate_histogram()

    def execution_on_reactome_enhanced(self):

        reaction_ids, reaction_names, physical_entity_ids, physical_entity_names, relationships_between_nodes_edges, component_ids, component_names, entity_component_mapping_list = self.extract_edges_ids_names_and_nodes_ids_names_and_relationships_and_all_component_ids_names_and_list_of_components_of_all_entities_for_one_pathway_without_duplicate_name(
            -1)

        # calculate the data distribution
        reaction_index_to_list_of_relationships_dic, reaction_index_to_list_of_input_relationships_dic, reaction_index_to_list_of_output_relationships_dic, reaction_index_to_list_of_regulation_relationship_dic = self.get_reactions_index_to_list_of_relationships_dic_based_on_relationships(
            relationships_between_nodes_edges)

        reaction_to_relationship_status_dic = self.get_reaction_status_dic(reaction_index_to_list_of_relationships_dic)
        reaction_to_input_relationship_status_dic = self.get_reaction_status_dic(
            reaction_index_to_list_of_input_relationships_dic)
        reaction_to_output_relationship_status_dic = self.get_reaction_status_dic(
            reaction_index_to_list_of_output_relationships_dic)
        reaction_to_regulation_relationship_status_dic = self.get_reaction_status_dic(
            reaction_index_to_list_of_regulation_relationship_dic)

        print("\033[1;32m" + "For all the relationships:" + "\033[0m")
        self.print_reaction_status_dic(reaction_to_relationship_status_dic)
        print("\n")

        print("\033[1;32m" + "For input relationship:" + "\033[0m")
        self.print_reaction_status_dic(reaction_to_input_relationship_status_dic, mode="input")
        print("\n")

        print("\033[1;32m" + "For output relationship:" + "\033[0m")
        self.print_reaction_status_dic(reaction_to_output_relationship_status_dic, mode="output")
        print("\n")

        print("\033[1;32m" + "For regulation relationship:" + "\033[0m")
        self.print_reaction_status_dic(reaction_to_regulation_relationship_status_dic, mode="regulation")
        print("\n")

        entity_index_to_list_of_relationships_dic = self.get_entity_index_to_list_of_relationships_dic_based_on_relationships(
            relationships_between_nodes_edges)
        entity_to_relationship_status_dic = self.get_entity_status_dic(entity_index_to_list_of_relationships_dic)

        print("\033[1;32m" + "For entities:" + "\033[0m")
        self.print_entity_status_dic(entity_to_relationship_status_dic)
        print("\n")

        entity_index_to_list_of_components_dic = self.get_entity_index_to_list_of_components_dic_based_on_entity_component_mapping_list(
            entity_component_mapping_list)
        entity_components_status_dic = self.get_entity_components_status_dic_based_on_entity_index_to_list_of_components_dic(
            entity_index_to_list_of_components_dic)

        print("\033[1;32m" + "For entities and components:" + "\033[0m")
        self.print_entity_components_distribution_dic(entity_components_status_dic)
        print("\n")

        if not os.path.exists("./data/All_data_in_Reactome"):
            os.makedirs("./data/All_data_in_Reactome")

        # store data into a txt file
        file_processor = FileProcessor()
        file_processor.execute_for_single_pathway_with_name_files("All_data_in_Reactome", reaction_ids, reaction_names,
                                                                  physical_entity_ids, physical_entity_names,
                                                                  relationships_between_nodes_edges, component_ids,
                                                                  component_names,
                                                                  entity_component_mapping_list)

        # draw the histogram
        # drawer = Drawer(len(reaction_names), len(physical_entity_names), len(component_names), "All_data_in_Reactome")
        # drawer.generate_histogram()


# one jump to n jump
# "MATCH (n:Pathway)-[r:hasEvent*1..]->(m:Reaction) WHERE n.stId = 'R-HSA-9612973' AND n.speciesName='Homo sapiens' RETURN m"

# "MATCH (n:Reaction)-[r:input*1..]->(m:PhysicalEntity) WHERE n.stId = 'R-HSA-9626034' AND n.speciesName='Homo sapiens' RETURN m.displayName, m.stId, n.displayName, n.stId"


class Drawer:
    def __init__(self, num_of_hyper_edges, num_of_nodes, dimensionality, pathway_name):
        self.num_of_hyper_edges = num_of_hyper_edges
        self.num_of_nodes = num_of_nodes
        self.dimensionality = dimensionality
        self.pathway_name = pathway_name
        cur_path = os.path.abspath(os.path.dirname(__file__))
        self.root_path = cur_path[:cur_path.find("extract_data_from_reactome\\") + len("extract_data_from_reactome\\")]
        # self.root_path = cur_path[:cur_path.find("PathwayGNN\\") + len("PathwayGNN\\")]

        self.path = os.path.join(self.root_path, "data", pathway_name)

    def generate_histogram(self):
        x1 = [self.pathway_name]
        y1 = [self.num_of_hyper_edges]
        y2 = [self.num_of_nodes]
        y3 = [self.dimensionality]
        name_of_file = self.pathway_name + ".html"
        path = os.path.join(self.path, name_of_file)
        url = os.path.join(path, name_of_file)

        if os.path.exists(url):
            print("file exists, we'll delete the original file \"" + name_of_file + "\", then create a new one")
            os.remove(url)

        bar = (
            Bar()
                .add_xaxis(x1)
                .add_yaxis("Hyper Edges(Reactions)", y1)
                .add_yaxis("Nodes(Physical Entity)", y2)
                .add_yaxis("Dimensionality(All components of nodes", y3)
                .set_global_opts(title_opts=opts.TitleOpts(title=self.pathway_name)))

        bar.render(path)


class FileProcessor:
    def __init__(self):
        self.filename_reactions = "edges.txt"
        self.filename_reactions_names = "edges-names.txt"
        self.filename_physical_entities = "nodes.txt"
        self.filename_physical_entities_names = "nodes-names.txt"
        self.filename_relationships = "relationship.txt"
        self.filename_components_mapping = "components-mapping.txt"
        self.filename_components_all = "components-all.txt"
        self.filename_components_all_names = "components-all-names.txt"
        # PathwayGNN
        # self.root_path = cur_path[:cur_path.find("PathwayGNN\\") + len("PathwayGNN\\")]

    # data/All_data_in_Reactome/components-all.txt
    # create the txt file to store the data
    def createFile(self, path, file_name):
        url = os.path.join("..", "..", path, file_name)
        if not os.path.exists(os.path.join("..", "..", path)):
            os.makedirs(path)
        if os.path.exists(url):
            print("file exists, we'll delete the original file \"" + file_name + "\", then create a new one")
            os.remove(url)
        file = open(url, 'w', encoding='utf-8')

    def delete_file(self, path, file_name) -> None:
        url = os.path.join(path, file_name)
        if os.path.exists(url):
            os.remove(url)

    # write message to txt file
    def writeMessageToFile(self, path, file_name, message: list[str]):
        url = os.path.join("..", "..", path, file_name)
        if not os.path.exists(url):
            print("error! the file \"" + file_name + "\" doesn't exist!")

        message = np.array(message)
        # np.savetxt(url, message, delimiter=',', fmt='%s', encoding='utf-8')

        file = open(url, "w", encoding="UTF-8")
        for index, line in enumerate(message):
            if index == (len(message) - 1):
                file.write(line)
            else:
                file.write(line+"\n")
        file.close()

    def create_and_write_message_to_file(self, path, file_name, message: list):
        self.createFile(path, file_name)
        self.writeMessageToFile(path, file_name, message)


    def execute_for_single_pathway(self, pathway_name, reaction_ids, physical_entity_ids,
                                   relationships_between_nodes_edges, component_ids, entity_component_mapping_list):

        path = os.path.join("data", pathway_name)

        # write message to the file
        file_professor = FileProcessor()

        file_professor.createFile(path, self.filename_reactions)
        file_professor.createFile(path, self.filename_physical_entities)
        file_professor.createFile(path, self.filename_relationships)
        file_professor.createFile(path, self.filename_components_all)
        file_professor.createFile(path, self.filename_components_mapping)

        file_professor.writeMessageToFile(path, self.filename_reactions, reaction_ids)
        file_professor.writeMessageToFile(path, self.filename_physical_entities, physical_entity_ids)
        file_professor.writeMessageToFile(path, self.filename_relationships, relationships_between_nodes_edges)
        file_professor.writeMessageToFile(path, self.filename_components_all, component_ids)
        file_professor.writeMessageToFile(path, self.filename_components_mapping, entity_component_mapping_list)

    def execute_for_single_pathway_with_name_files(self, pathway_name, reaction_ids, reaction_names,
                                                   physical_entity_ids,
                                                   physical_entity_names,
                                                   relationships_between_nodes_edges, component_ids, component_names,
                                                   entity_component_mapping_list):

        path = os.path.join("data", pathway_name)

        # write message to the file
        file_professor = FileProcessor()

        file_professor.createFile(path, self.filename_reactions)
        file_professor.createFile(path, self.filename_reactions_names)
        file_professor.createFile(path, self.filename_physical_entities)
        file_professor.createFile(path, self.filename_physical_entities_names)
        file_professor.createFile(path, self.filename_relationships)
        file_professor.createFile(path, self.filename_components_all)
        file_professor.createFile(path, self.filename_components_all_names)
        file_professor.createFile(path, self.filename_components_mapping)

        file_professor.writeMessageToFile(path, self.filename_reactions, reaction_ids)
        file_professor.writeMessageToFile(path, self.filename_reactions_names, reaction_names)
        file_professor.writeMessageToFile(path, self.filename_physical_entities, physical_entity_ids)
        file_professor.writeMessageToFile(path, self.filename_physical_entities_names, physical_entity_names)

        file_professor.writeMessageToFile(path, self.filename_relationships, relationships_between_nodes_edges)
        file_professor.writeMessageToFile(path, self.filename_components_all, component_ids)
        file_professor.writeMessageToFile(path, self.filename_components_all_names, component_names)

        file_professor.writeMessageToFile(path, self.filename_components_mapping, entity_component_mapping_list)

    def read_file_via_lines(self, path, file_name):
        url = os.path.join("..", "..", path, file_name)
        res_list = []

        try:
            file_handler = open(url, "r")
            while True:
                # Get next line from file
                line = file_handler.readline()
                line = line.replace('\r', '').replace('\n', '').replace('\t', '')

                # If the line is empty then the end of file reached
                if not line:
                    break
                res_list.append(line)
        except Exception as e:
            print(e)
            print("we can't find the " + url + ", please make sure that the file exists")
        finally:
            return res_list


if __name__ == '__main__':
    # graph = Graph("bolt://localhost:7687", auth=('neo4j', '123456'))
    #
    # processor = PhysicalEntityProcessor(graph)
    #
    # components = processor.get_components_of_physical_entity('R-HSA-170079')

    time_start = time.time()  # record the start time

    reactome_processor = ReactomeProcessor('neo4j', '123456')

    # reactome_processor.execution_on_single_pathway_via_name_enhanced("Disease")

    # reactome_processor.execution_on_single_pathway_via_name_enhanced("Immune System")

    # reactome_processor.execution_on_single_pathway_via_name_enhanced("Metabolism")

    # reactome_processor.execution_on_single_pathway_via_name_enhanced("Signal Transduction")




    time_end = time.time()  # record the ending time

    time_sum = time_end - time_start  # The difference is the execution time of the program in seconds

    print("success! it takes " + str(time_sum) + " seconds to extract the data from Reactome")
