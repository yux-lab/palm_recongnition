import sys
import os
import cv2
import numpy as np

# 添加路径
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
from abc import ABC, abstractmethod

from base import ShowImage, config_toml
from palm_roi_ext.hand_key_points.key_point import HandKeyPointDetect
from palm_roi_ext.hand_segment.segment import FastInstanceSegmentation
from palm_roi_ext.palm_core.position_fitness.palm_pso import PsoPositionPalm
from palm_roi_ext.palm_core.rotate import HandeRotateCommand
from palm_roi_ext.positions import TriangleTransform, DistTransform


class ROIExtract(ABC):

    @abstractmethod
    def roi_extract(self, **kwargs):
        pass

    def extract_roi(self, image_rgb, center, max_radius):
        # 绘制最大内接圆和内切正方形
        image_with_circle = cv2.circle(image_rgb.copy(), center, max_radius, (0, 255, 0), 2)
        square_side = int(max_radius * np.sqrt(2))
        square_top_left = (int(center[0] - square_side / 2), int(center[1] - square_side / 2))
        square_bottom_right = (int(center[0] + square_side / 2), int(center[1] + square_side / 2))
        image_with_circle_and_square = cv2.rectangle(image_with_circle, square_top_left, square_bottom_right,
                                                     (0, 0, 255), 2)

        # 裁剪内切正方形区域
        cropped_square = image_rgb[square_top_left[1]:square_bottom_right[1], square_top_left[0]:square_bottom_right[0]]

        # 创建与原图同样大小的黑色掩膜
        mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        cv2.circle(mask, center, max_radius, (255, 255, 255), -1)
        cropped_circle = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
        non_zero_y, non_zero_x = np.nonzero(mask)
        cropped_circle = cropped_circle[non_zero_y.min():non_zero_y.max(), non_zero_x.min():non_zero_x.max()]

        # 调整大小到224x224
        cropped_square_resized = cv2.resize(cropped_square, (224, 224))
        cropped_circle_resized = cv2.resize(cropped_circle, (224, 224))

        return image_with_circle_and_square, cropped_square_resized, cropped_circle_resized


class AutoRotateRoIExtract(ROIExtract, ShowImage):

    def __init__(self):
        self.key_points_instance = HandKeyPointDetect()
        self.rotate_instance = HandeRotateCommand()
        self.hand_segment = FastInstanceSegmentation()
        self.index_center_instance = TriangleTransform()
        self.index_center_instance_ = DistTransform()
        self.pso_roi_instance = PsoPositionPalm()

    def roi_extract(self, img):
        x_up, y_up = img.shape[:2][::-1]
        key_points = self.key_points_instance.get_hand_key_point(img)
        img, angle, key_points = self.rotate_instance.rotate_angle_img(key_points, img)
        hand_segment = self.hand_segment.segment(img)
        hand_only_image, hand_binary_image = self.hand_segment.keep_only_hand_in_image(img, hand_segment)
        center, base_radius = self.index_center_instance.get_init_center(key_points)
        center, base_radius = self.pso_optimize(center, hand_binary_image, x_up, y_up, base_radius)
        draw_img, roi_square, roi_circle = self.extract_roi(hand_only_image, center, base_radius)
        return draw_img, roi_square, roi_circle

    def pso_optimize(self, center_point, binary_image, x_up, y_up, base_radius=5):
        self.pso_roi_instance.random_radius = config_toml["PSO"]["random_radius"]
        self.pso_roi_instance.init_center = center_point
        self.pso_roi_instance.population_number = config_toml["PSO"]["population_number"]
        self.pso_roi_instance.iter_number = config_toml["PSO"]["iter_number"]
        self.pso_roi_instance.binary_image = binary_image
        self.pso_roi_instance.base_radius = base_radius
        self.pso_roi_instance.padding_step = config_toml["PSO"]["padding_step"]
        bound_box = [[0, x_up], [0, y_up]]
        self.pso_roi_instance.bound_box = bound_box
        self.pso_roi_instance.init_center = np.array(center_point)
        self.pso_roi_instance.binary_image = binary_image
        self.pso_roi_instance.init_population = True
        self.pso_roi_instance.generator_pso_instance()
        self.pso_roi_instance.pso_instance.optimize()
        best_position, best_fitness = self.pso_roi_instance.pso_instance.get_best_solution()
        center_point = [int(i) for i in best_position]
        base_radius = abs(int(best_fitness))
        return center_point, base_radius

    # 保存ROI区域
    def save_roi_from_directory(self, input_dir, output_dir):
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(root, file)
                    try:
                        img = cv2.imread(img_path)
                        if img is None:
                            print(f"Failed to read image: {img_path}")
                            continue

                        draw_img, roi_square, roi_circle = self.roi_extract(img)

                        # 确保输出目录存在
                        relative_path = os.path.relpath(root, input_dir)
                        save_dir = os.path.join(output_dir, relative_path)
                        os.makedirs(save_dir, exist_ok=True)

                        # 保存ROI区域
                        roi_square_path = os.path.join(save_dir, f'roi_square_{file}')
                        # roi_circle_path = os.path.join(save_dir, f'roi_circle_{file}')

                        cv2.imwrite(roi_square_path, roi_square)
                        # cv2.imwrite(roi_circle_path, roi_circle)

                        print(f"Saved ROI square image: {roi_square_path}")
                        # print(f"Saved ROI circle image: {roi_circle_path}")

                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")


# if __name__ == '__main__':
#     input_base_dir = r'D:\Yux\palm prints\datasets\WEHI & MOHI\MOHI'
#     output_base_dir = r'D:\Yux\palm prints\datasets\WEHI & MOHI\MOHI_ROI'
#
#     roi_extractor = AutoRotateRoIExtract()
#     roi_extractor.save_roi_from_directory(input_base_dir, output_base_dir)
if __name__ == '__main__':
    input_output_pairs = [
        (r'D:\Yux\palm prints\datasets\REST\REST database', r'D:\Yux\palm prints\datasets\REST\REST_ROI database'),
        (r'D:\Yux\palm prints\datasets\WEHI & MOHI\MOHI', r'D:\Yux\palm prints\datasets\WEHI & MOHI\MOHI_ROI'),
        (r'D:\Yux\palm prints\datasets\WEHI & MOHI\WEHI', r'D:\Yux\palm prints\datasets\WEHI & MOHI\WEHI_ROI'),
        (r'D:\Yux\palm prints\datasets\COEP\database', r'D:\Yux\palm prints\datasets\COEP\database_ROI'),
        (r'D:\Yux\palm prints\datasets\BMPD\archive\Birjand University Mobile Palmprint Database (BMPD)', r'D:\Yux\palm prints\datasets\BMPD\archive\BMPD_ROI'),
        (r'D:\Yux\palm prints\datasets\SMPD\archive\Sapienza University Mobile Palmprint Database(SMPD)', r'D:\Yux\palm prints\datasets\SMPD\archive\SMPD_ROI'),
        (r'D:\Yux\palm prints\datasets\CASIA\CASIA-Multi-Spectral-PalmprintV1\images', r'D:\Yux\palm prints\datasets\CASIA\CASIA-Multi-Spectral-PalmprintV1\images_ROI'),
        (r'D:\Yux\palm prints\datasets\CASIA\CASIA-PalmprintV1', r'D:\Yux\palm prints\datasets\CASIA\CASIA-PalmprintV1-ROI'),
        (r'D:\Yux\palm prints\datasets\COEP\database', r'D:\Yux\palm prints\datasets\COEP\database_ROI')
    ]

    roi_extractor = AutoRotateRoIExtract()

    for input_base_dir, output_base_dir in input_output_pairs:
        roi_extractor.save_roi_from_directory(input_base_dir, output_base_dir)
