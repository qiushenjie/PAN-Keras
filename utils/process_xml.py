import os
import numpy as np
try:
   import xml.etree.cElementTree as ET
except ImportError:
   import xml.etree.ElementTree as ET

def str_to_poly(coor_str): # coor_str: xmin1 ymin1 xmax1 ymax1 xmin2 ymin2 xmax2 ymax2......
    coor_list = coor_str.split(' ')
    coor_list = [int(i) for i in coor_list]
    assert (len(coor_list) % 4 == 0), 'wrong label length'
    coor_list = np.array(coor_list).reshape(-1, 2, 2)
    return coor_list

def read_xml_list(xml_name):
    coordinate = []
    tree = ET.parse(xml_name)
    root = tree.getroot()
    filename = root.find('filename').text
    for item in root.findall('object'):
        bndbox = item.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        # coordinate.append([[xmin, ymin],[xmax, ymin],[xmax, ymax], [xmin, ymax]])
        coordinate.append([xmin, ymin, xmax, ymax])
    return np.array(coordinate)

def read_xml_str(xml_name):
    coordinate = ''
    tree = ET.parse(xml_name)
    root = tree.getroot()
    filename = root.find('filename').text
    for item in root.findall('object'):
        bndbox = item.find('bndbox')
        xmin = bndbox.find('xmin').text
        xmax = bndbox.find('xmax').text
        ymin = bndbox.find('ymin').text
        ymax = bndbox.find('ymax').text
        # coordinate += xmin + ' ' + ymin + ' ' + xmax + ' ' + ymax + ' '
        coordinate += xmin + ' ' + ymin + ' ' + xmax + ' ' + ymin + ' ' + xmax + ' ' + ymax + ' ' + xmin + ' ' + ymax + ' '
    return filename, coordinate[:-1]

if __name__ == '__main__':
    data_path = './datasets'
    train_txt_path = 'train.txt'
    val_txt_path = 'val.txt'
    train_dataset = open(train_txt_path, 'w')
    val_dataset = open(val_txt_path, 'w')
    files = os.listdir(data_path)
    train_split = 0.85
    name = []
    for item in files:
        if item.endswith('xml') and (item[:-3] + 'jpg') in files:
            name.append(item)
    for item in name[: (int(len(name) * train_split))]:
        filename, coord = read_xml_str(os.path.join(data_path, item))
        info = filename + ' ' + coord + '\n'
        train_dataset.write(info)
    for item in name[(int(len(name) * train_split)):]:
        filename, coord = read_xml_str(os.path.join(data_path, item))
        info = filename + ' ' + coord + '\n'
        val_dataset.write(info)
    print((int(len(name) * train_split)), len(name) - (int(len(name) * train_split)))

