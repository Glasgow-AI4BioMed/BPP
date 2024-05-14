from app.bean import all_toplevel_pathways, ToplevelPathway
from typing import List, Optional

from app.bean.bean_collection import ToplevelPathwaySelector


class ToplevelPathwayService:
    def __init__(self):
        self.__toplevel_pathway_selector = ToplevelPathwaySelector(all_toplevel_pathways)

    def convert_toplevel_name_url_format_to_toplevel_name(self, toplevel_name_url: str) -> str:
        toplevel_name_url = toplevel_name_url.replace('_', '\x20')
        return ' '.join(word[0].upper() + word[1:] for word in toplevel_name_url.split())

    def get_all_toplevel_pathways(self) -> List[ToplevelPathway]:
        return all_toplevel_pathways

    def get_toplevel_pathway_based_on_name(self, name: str) -> Optional[ToplevelPathway]:
        return self.__toplevel_pathway_selector.select_toplevel_pathway_based_on_name(name)

    def get_toplevel_pathway_based_on_name_url_format(self, name_url_format: str) -> Optional[ToplevelPathway]:
        return self.__toplevel_pathway_selector.select_toplevel_pathway_based_on_name_url_format(name_url_format)


toplevel_pathway_service_obj = ToplevelPathwayService()
