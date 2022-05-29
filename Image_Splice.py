import PIL.Image as Image
import os
import re

IMAGES_PATH = 'D:\\我的课程\\毕业论文\\图片集'  # 图片集地址
IMAGES_FORMAT = ['.jpg', '.JPG', '.png']  # 图片格式
IMAGE_SIZE = 256  # 每张小图片的大小
IMAGE_ROW = 2  # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 1  # 图片间隔，也就是合并成一张图后，一共有几列
IMAGE_SAVE_PATH = 'final.jpg'  # 图片转换后的地址

# 获取图片集地址下的所有图片名称
image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]


def images(name="对抗"):
    ore = re.compile(r"^%s风格\d.png" % name)
    all_need = []
    for image in image_names:
        if ore.search(image):
            all_need.append(image)
    return all_need


# 定义图像拼接函数
def image_compose(image_names):
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            from_image = Image.open(IMAGES_PATH + '/' + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
    return to_image.save(IMAGE_SAVE_PATH)  # 保存新图


if __name__ == "__main__":
    all_need = images(name="拖延")
    print(all_need)
    image_compose(all_need)
