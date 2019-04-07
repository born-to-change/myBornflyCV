# import math
#
# class Pizza(object):
#     def __init__(self, radius, height):
#         self.radius = radius
#         self.height = height
#
#     @staticmethod
#     def compute_area(radius):
#         return math.pi * (radius ** 2)
#
#     @classmethod
#     def compute_volume(cls, height, radius):
#         return height * cls.compute_area(radius)
#
#     def get_volume(self):
#         return self.compute_volume(self.height, self.radius)
#
#
#
# print(Pizza(2,3).get_volume())  #37.69911184307752

import abc


class BasePizza(object):
    __metaclass__ = abc.ABCMeta

    default_ingredients = 'cheese'
    @staticmethod
    def change_ingredients():
        return 'steak'

    @classmethod
    #@abc.abstractmethod                #类方法,可以使用类中静态方法
    def get_ingredients(cls):
        """Returns the ingredient list."""
        return cls.default_ingredients + cls.change_ingredients()


class DietPizza(BasePizza):
    def get_ingredients(self):
        #return self.get_ingredients()
        return 'egg' + super(DietPizza, self).get_ingredients()


print(DietPizza().get_ingredients())
#  你构建的每个pizza都通过继承BasePizza的方式，你不得不覆盖get_ingredients方法，但是能够使用默认机制通过super()来获取ingredient列表。