# -*- coding:utf-8 -*-
# Author：hankcs
# Date: 2018-06-02 22:53
# 《自然语言处理入门》2.9 准确率评测
# 配套书籍：http://nlp.hankcs.com/book.php
# 讨论答疑：https://bbs.hankcs.com/
import re
from pyhanlp import *
from tests.test_utility import ensure_data


def to_region(segmentation: str) -> list:
    """
    将分词结果转换为区间
    :param segmentation: 商品 和 服务
    :return: [(0, 2), (2, 3), (3, 5)]
    """
    region = []
    start = 0
    for word in re.compile("\\s+").split(segmentation.strip()):#默认情况下，函数strip ()将删除字符串开头和结尾的空格，\s表示匹配任何空白字符，\s+匹配任意多个上述字符
        end = start + len(word)#re.split()方法中，每次调用都需要编译正则表达式，消耗了一定的时间。如果需要反复使用同一正则表达式，可以使用re.compile()方法进行优化
        region.append((start, end))
        start = end
    return region


def prf(gold: str, pred: str, dic) -> tuple:
    """
    计算P、R、F1
    :param gold: 标准答案文件，比如“商品 和 服务”
    :param pred: 分词结果文件，比如“商品 和服 务”
    :param dic: 词典
    :return: (P, R, F1, OOV_R, IV_R)
    """
    A_size, B_size, A_cap_B_size, OOV, IV, OOV_R, IV_R = 0, 0, 0, 0, 0, 0, 0
    with open(gold,encoding='utf-8') as gd, open(pred,encoding='utf-8') as pd:
        for g, p in zip(gd, pd):
            A, B = set(to_region(g)), set(to_region(p))
            A_size += len(A)
            B_size += len(B)
            A_cap_B_size += len(A & B)
            text = re.sub("\\s+", "", g)
            for (start, end) in A:
                word = text[start: end]
                if word in dic:
                    IV += 1
                else:
                    OOV += 1

            for (start, end) in A & B:
                word = text[start: end]
                if word in dic:
                    IV_R += 1
                else:
                    OOV_R += 1
    p, r = A_cap_B_size / B_size * 100, A_cap_B_size / A_size * 100
    return p, r, 2 * p * r / (p + r), OOV_R / OOV * 100, IV_R / IV * 100


if __name__ == '__main__':
    print(to_region('商品 和 服务'))
    print(prf('C:\\Users\\12194\\Desktop\\hanlp-python\\pyhanlp\\tests\\book\\ch02\\gold.txt','C:\\Users\\12194\\Desktop\\hanlp-python\\pyhanlp\\tests\\book\\ch02\\pred.txt',['和','和服','服务','务']))
