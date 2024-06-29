# Author : WangZhen
import cv2
import numpy as np


class ImageHash:
    def __init__(self, img_path: str):
        self.img_path = img_path
        self.img = cv2.imread(img_path)
        if self.img is None:
            raise ValueError(f"无法读取图片: {img_path}")

    def ahash(self) -> str:
        """
        均值哈希算法
        :return: 哈希值
        """
        # 将原图缩放为8x8大小
        img_resized = cv2.resize(self.img, (8, 8))
        # 灰度化
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        # 计算所有像素的平均值
        avg_result = img_gray.mean()
        # 比较每个像素点和平均值
        for i in range(img_gray.shape[0]):
            for j in range(img_gray.shape[1]):
                if img_gray[i][j] >= avg_result:
                    img_gray[i][j] = 1
                else:
                    img_gray[i][j] = 0
        # 统计64位hash值
        hash_str = ""
        for i in img_gray.ravel():
            hash_str += str(i)

        return hash_str

    def dhash(self) -> str:
        """
        差异哈希算法
        :return: 哈希值
        """
        # 将原图缩放为9x8大小
        img_resized = cv2.resize(self.img, (9, 8))
        # 灰度化
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        # 创建一个哈希数组
        hash_np = np.zeros(shape=(8, 8), dtype=int)
        # 计算差异
        for i in range(img_gray.shape[0]):
            for j in range(img_gray.shape[1] - 1):
                if img_gray[i][j] > img_gray[i][j + 1]:
                    hash_np[i][j] = 1
        # 统计64位hash值
        hash_str = ''.join(hash_np.ravel().astype(str))

        return hash_str

    def phash(self) -> str:
        # todo 未实现
        pass


# 使用示例
if __name__ == "__main__":
    # 图片路径
    img_path = ""
    img_hash = ImageHash(img_path)
    print("aHash:", img_hash.ahash())
    print("dHash:", img_hash.dhash())
