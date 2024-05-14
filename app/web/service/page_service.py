import copy
from typing import Tuple

from app.bean.bean_collection import Edge, Node, Attribute

class PageInf:
    def __init__(self, current_page, total_pages_num):
        self.is_first_page = False
        self.is_last_page = False
        self.current_page = current_page
        self.total_pages_num = total_pages_num
        self.previous_page = 0
        self.next_page = 0
        self.page_range = None
        self.total_page_range = [x for x in range(1, self.total_pages_num + 1)]

        self.initialize()

    def initialize(self):
        self.previous_page = self.current_page - 1
        self.next_page = self.current_page + 1
        start_page = self.current_page - 3
        end_page = self.current_page + 3
        if self.current_page == 1:
            self.is_first_page = True
            self.previous_page = 1
            start_page = 1

        if self.current_page == self.total_pages_num:
            self.is_last_page = True
            self.next_page = self.total_pages_num
            end_page = self.total_pages_num

        if start_page <= 0:
            start_page = 1

        if end_page > self.total_pages_num:
            end_page = self.total_pages_num

        self.page_range = [x for x in range(start_page, end_page+1)]

    def update_current_page(self, new_current_page: int):
        self.current_page = new_current_page
        self.initialize()




class PageNavigation:
    def __init__(self, element_list, page_size: int):
        self.__element_list = element_list
        self.__page_size = page_size

        self.__total_pages = int(len(self.__element_list) / self.__page_size) + 1

        self.page_info = PageInf(0, self.__total_pages)

    def page_navigate(self, current_page: int) -> Tuple[list, PageInf]:
        if current_page < 1:
            current_page = 1
        if current_page > self.__total_pages:
            current_page = self.__total_pages
        start_index: int = (current_page - 1) * self.__page_size
        if current_page < self.__total_pages:
            end_index: int = start_index + self.__page_size
        else:
            # elif page == self.__total_pages
            end_index: int = len(self.__element_list)

        self.page_info.update_current_page(current_page)

        # return copy.deepcopy(self.__element_list[start_index:end_index])
        return self.__element_list[start_index:end_index], self.page_info
