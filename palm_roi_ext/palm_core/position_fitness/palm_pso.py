# -*- coding: utf-8 -*-
# @文件：palm_pso.py
# @作者：Huterox
# @邮箱：3139541502@qq.com
# -------------------------------
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
import numpy as np
from palm_roi_ext.palm_core.position_fitness.pso import PSOInstance


class PsoPositionPalm():
    """
    基于PSO算法实现的中心点自适应算法
    """

    def __init__(self,init_center=None,random_radius=None,padding_step=None,
                 population_number=None,iter_number=None,
                 bound_box=None,base_radius=None, binary_image=None,
                 init_population = False
                 ):
        """
        :param init_center: 初始化中心点
        :param random_radius: 初始化种群的半径
        :param padding_step: fitness函数当中扩展半径的步数
        :param population_number: 初始化种群的个数
        :param iter_number: 迭代次数
        :param bound_box: 边界（这里直接设置到图像宽高就行了，当然也可以剪枝）
        :param base_radius: fitness函数当中扩充初始圆的基础半径
        :param binary_imag: 二值化的手部图像，用于判断圆心
        :param init_population: 是否初始化种群
        """
        self.init_center = init_center
        self.padding_step = padding_step
        self.population_number = population_number
        self.iter_number = iter_number
        self.bound_box = bound_box
        self.random_radius = random_radius
        self.binary_image = binary_image
        self.base_radius = base_radius
        self.init_population = init_population


    def generator_pso_instance(self):
        self.pso_instance = PSOInstance(
            num_particles=self.population_number,
            center=self.init_center,
            radius=self.random_radius,
            max_iterations=self.iter_number,
            bounds=self.bound_box,
            fitness_function=self.fitness,
            init_population=self.init_population
        )
        return self.pso_instance

    def fitness(self, center):
        """
        在白色区域中找到最大内接圆。
        :param binary_image: 二值化图像，黑色背景，白色手掌区域
        :param center: 圆心坐标 (x, y)
        :return: 最大内接圆的半径(由于PSO目标是优化最小化问题，因此这里的返回值是负数)
        """
        height, width = self.binary_image.shape
        max_radius = self.base_radius
        # 将中心点修正为int
        center[0],center[1] = int(center[0]),int(center[1])
        # 从圆心出发，逐渐扩大半径，直到圆接触到白色区域的边界
        for radius in range(self.base_radius, max(height, width), self.padding_step):
            # 检查圆是否超出边界
            if center[0] - radius < 0 or center[0] + radius >= width or center[1] - radius < 0 or center[1] + radius >= height:
                break
            # 检查圆的边界上是否有黑色点
            for angle in range(0, 360, 10):  # 基准10度验证即可
                x = int(center[0] + radius * np.cos(np.deg2rad(angle)))
                y = int(center[1] + radius * np.sin(np.deg2rad(angle)))
                if x < 0 or x >= width or y < 0 or y >= height:
                    continue
                if self.binary_image[y, x] == 0:
                    break
            else:
                max_radius = radius
                continue
            break
        return -max_radius


    def fit_area_center(self, **kwargs):
        self.pso_instance.optimize()
        best_position, best_fitness = self.pso_instance.get_best_solution()
        return  best_position, abs(best_fitness)
