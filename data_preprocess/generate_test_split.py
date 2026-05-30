"""生成 VOC 2007 test 集的增量 split 文件.

读取原始 ImageSets/Main/test.txt，按 category_split.json 的类别划分，
生成 task0_test.txt 和 task1_test.txt。
"""
import argparse
import os
import json
import xml.etree.ElementTree as ET

# 加载配置文件
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_SCRIPT_DIR, 'config.json'), 'r', encoding='utf-8') as f:
    _cfg_data = json.load(f)
_PATHS = _cfg_data['paths']


def parse_voc_xml_classes(xml_path):
    """解析 VOC XML，返回图中所有类别名列表."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    classes = set()
    for obj in root.findall('object'):
        name = obj.find('name').text.strip().lower()
        classes.add(name)
    return classes


def main():
    """主入口函数。"""
    parser = argparse.ArgumentParser(
        description='Pascal VOC 2007 Incremental Dataset Preprocessing'
    )
    parser.add_argument(
        '--protocol', type=str, default='10_10',
        choices=['19_1', '15_5', '10_10'],
        help='Incremental protocol: B+I'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for category shuffling'
    )
    args = parser.parse_args()

    data_root = _PATHS['data_root']
    protocol = args.protocol
    seed = args.seed
    
    # 读取原始 test 集图片列表
    test_list_file = os.path.join(data_root, 'ImageSets', 'Main', 'test.txt')
    with open(test_list_file, 'r') as f:
        test_img_ids = [line.strip() for line in f.readlines()]
    print(f"[INFO] Total test images: {len(test_img_ids)}")
    
    # 读取类别划分
    split_dir = os.path.join(data_root, f'incremental_{protocol}', f'seed{seed}')
    with open(os.path.join(split_dir, 'category_split.json'), 'r') as f:
        split_info = json.load(f)
    
    task_classes = split_info['task_classes']
    task0_set = set(task_classes[0])
    task1_set = set(task_classes[1])
    print(f"[INFO] Task 0 classes ({len(task0_set)}): {sorted(task0_set)}")
    print(f"[INFO] Task 1 classes ({len(task1_set)}): {sorted(task1_set)}")
    
    # 对每张 test 图，解析标注，分类到对应 task
    task0_imgs = []
    task1_imgs = []
    both_imgs = []
    neither_imgs = []
    
    for img_id in test_img_ids:
        xml_path = os.path.join(data_root, 'Annotations', f'{img_id}.xml')
        if not os.path.exists(xml_path):
            print(f"[WARN] Annotation not found: {xml_path}")
            continue
        
        img_classes = parse_voc_xml_classes(xml_path)
        
        has_task0 = bool(img_classes & task0_set)
        has_task1 = bool(img_classes & task1_set)
        
        if has_task0 and has_task1:
            both_imgs.append(img_id)
        elif has_task0:
            task0_imgs.append(img_id)
        elif has_task1:
            task1_imgs.append(img_id)
        else:
            neither_imgs.append(img_id)
    
    # 写入文件
    # Task 0 test: 包含 Task 0 类别的图（包括同时有 Task 1 的）
    task0_test = task0_imgs + both_imgs
    task1_test = task1_imgs + both_imgs
    
    with open(os.path.join(split_dir, 'task0_test.txt'), 'w') as f:
        for img_id in task0_test:
            f.write(img_id + '\n')
    
    with open(os.path.join(split_dir, 'task1_test.txt'), 'w') as f:
        for img_id in task1_test:
            f.write(img_id + '\n')
    
    print(f"\n{'='*50}")
    print(f"Task 0 test: {len(task0_test)} images")
    print(f"  - Only Task 0 classes: {len(task0_imgs)}")
    print(f"  - Both tasks: {len(both_imgs)}")
    print(f"Task 1 test: {len(task1_test)} images")
    print(f"  - Only Task 1 classes: {len(task1_imgs)}")
    print(f"  - Both tasks: {len(both_imgs)}")
    print(f"Neither task: {len(neither_imgs)} images")
    print(f"{'='*50}")
    print(f"[DONE] Files saved to {split_dir}/")


if __name__ == '__main__':
    main()
