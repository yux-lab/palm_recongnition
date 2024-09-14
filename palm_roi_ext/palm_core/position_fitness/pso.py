# -*- coding: utf-8 -*-
# @文件：pso.py
# @作者：Huterox
# @邮箱：3139541502@qq.com
# -------------------------------
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
import numpy as np

class PSOInstance(object):
    def __init__(self, num_particles, center, radius, max_iterations, bounds, fitness_function,
                 w_start=0.9, w_end=0.4, c1=1.496, c2=1.496,init_population=True):
        """
        初始化PSO类。
        :param num_particles: 粒子数量
        :param center: 初始化粒子群的中心点
        :param radius: 初始化粒子与中心点的最大距离
        :param max_iterations: 最大迭代次数
        :param bounds: 搜索空间的边界，二维数组，每一行对应一个维度的上下限
        :param fitness_function: 目标函数，最小化此函数
        :param w_start: 初始惯性权重
        :param w_end: 最终惯性权重
        :param c1: 认知因子
        :param c2: 社会因子
        """
        self.num_particles = num_particles
        self.center = np.array(center)
        self.radius = radius
        self.max_iterations = max_iterations
        self.bounds = np.array(bounds)
        self.fitness_function = fitness_function

        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2

        # 初始化粒子群,与全局最优解，最优位置
        if init_population:
            self.population = self.initialize_population()



    def initialize_population(self):
        """
        初始化粒子群的位置和速度。

        :return: 包含所有粒子信息的列表
        """
        dim = len(self.bounds)  # 维度
        population = []
        for _ in range(self.num_particles):
            # 随机初始化粒子的位置
            pix = (np.random.rand(dim) * 2 - 1) * self.radius
            position = self.center + pix
            # 初始化粒子的速度
            velocity = np.zeros(dim)
            # 添加到粒子群
            population.append({
                'position': position,
                'velocity': velocity,
                'best_position': position.copy(),
                'best_fitness': None
            })
        # 通过初始化得到的粒子位置，计算初始化的最优解和位置
        self.global_best_fitness = float('inf')
        self.global_best_position = None
        for particle in population:
            fitness = self.fitness_function(particle['position'])
            particle['best_fitness'] = fitness
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = particle['position']
        return population

    def update_velocity(self, particle, iteration):
        """
        更新粒子的速度。

        :param particle: 当前粒子的信息字典
        :param iteration: 当前迭代次数
        """
        # 计算线性递减的惯性权重
        w = self.w_start - (self.w_start - self.w_end) * (iteration / self.max_iterations)
        # 计算认知部分
        cognitive = self.c1 * np.random.rand() * (particle['best_position'] - particle['position'])
        # 计算社会部分
        social = self.c2 * np.random.rand() * (self.global_best_position - particle['position'])
        # 更新速度
        particle['velocity'] = w * particle['velocity'] + cognitive + social

    def update_position(self, particle):
        """
        更新粒子的位置，并确保其位于边界内。

        :param particle: 当前粒子的信息字典
        """
        particle['position'] += particle['velocity']
        # 限制位置在边界范围内
        particle['position'] = np.clip(particle['position'], self.bounds[:, 0], self.bounds[:, 1])

    def evaluate_fitness(self):
        """
        评估粒子群的适应度，并更新个体最优和全局最优。
        """
        for particle in self.population:
            # 计算适应度值
            fitness = self.fitness_function(particle['position'])
            # 如果当前粒子的适应度优于历史最优，则更新
            if particle['best_fitness'] is None or fitness < particle['best_fitness']:
                particle['best_fitness'] = fitness
                particle['best_position'] = particle['position'].copy()

    def optimize(self):
        """
        执行优化过程。
        """
        for iteration in range(self.max_iterations):
            for particle in self.population:
                self.update_velocity(particle, iteration)
                self.update_position(particle)

            self.evaluate_fitness()
            # 更新全局最优解
            for particle in self.population:
                if particle['best_fitness'] < self.global_best_fitness:
                    self.global_best_fitness = particle['best_fitness']
                    self.global_best_position = particle['best_position'].copy()

    def get_best_solution(self):
        """
        获取优化过程中找到的最佳解。
        :return: 最佳解的位置和适应度值
        """
        return self.global_best_position, self.global_best_fitness

if __name__ == '__main__':

    bounds = np.array([[-5, 5], [-5, 5]])  # 定义边界
    center = np.array([0, 0])  # 中心点
    radius = 2  # 圆心范围
    num_particles = 50  # 粒子数量
    max_iterations = 100  # 迭代次数

    # 定义适应函数（最小化）
    def fitness_function(x):
        return x[0] ** 2 + x[1] ** 2  # 测试：最小化目标函数 f(x) = x^2 + y^2


    # 创建并运行PSO实例
    pso_instance = PSOInstance(num_particles, center, radius, max_iterations, bounds, fitness_function)
    pso_instance.optimize()

    # 获取最佳解决方案
    best_position, best_fitness = pso_instance.get_best_solution()
    print(f"Best Position: {best_position}, Best Fitness: {best_fitness}")