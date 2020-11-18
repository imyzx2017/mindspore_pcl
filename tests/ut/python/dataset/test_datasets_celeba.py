# Copyright 2020 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as vision
from mindspore import log as logger
from mindspore.dataset.vision import Inter

DATA_DIR = "../data/dataset/testCelebAData/"


def test_celeba_dataset_label():
    data = ds.CelebADataset(DATA_DIR, shuffle=False, decode=True)
    expect_labels = [
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1,
         0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
         0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
         0, 0, 1],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1,
         0, 0, 1]]
    count = 0
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info("----------image--------")
        logger.info(item["image"])
        logger.info("----------attr--------")
        logger.info(item["attr"])
        for index in range(len(expect_labels[count])):
            assert item["attr"][index] == expect_labels[count][index]
        count = count + 1
    assert count == 4


def test_celeba_dataset_op():
    data = ds.CelebADataset(DATA_DIR, decode=True, num_shards=1, shard_id=0)
    crop_size = (80, 80)
    resize_size = (24, 24)
    # define map operations
    data = data.repeat(2)
    center_crop = vision.CenterCrop(crop_size)
    resize_op = vision.Resize(resize_size, Inter.LINEAR)  # Bilinear mode
    data = data.map(operations=center_crop, input_columns=["image"])
    data = data.map(operations=resize_op, input_columns=["image"])

    count = 0
    for item in data.create_dict_iterator(num_epochs=1):
        logger.info("----------image--------")
        logger.info(item["image"])
        count = count + 1
    assert count == 8


def test_celeba_dataset_ext():
    ext = [".JPEG"]
    data = ds.CelebADataset(DATA_DIR, decode=True, extensions=ext)
    expect_labels = [
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1,
         0, 1, 0, 1, 0, 0, 1],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1,
         0, 1, 0, 1, 0, 0, 1]]
    count = 0
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info("----------image--------")
        logger.info(item["image"])
        logger.info("----------attr--------")
        logger.info(item["attr"])
        for index in range(len(expect_labels[count])):
            assert item["attr"][index] == expect_labels[count][index]
        count = count + 1
    assert count == 2


def test_celeba_dataset_distribute():
    data = ds.CelebADataset(DATA_DIR, decode=True, num_shards=2, shard_id=0)
    count = 0
    for item in data.create_dict_iterator(num_epochs=1):
        logger.info("----------image--------")
        logger.info(item["image"])
        logger.info("----------attr--------")
        logger.info(item["attr"])
        count = count + 1
    assert count == 2


def test_celeba_get_dataset_size():
    data = ds.CelebADataset(DATA_DIR, shuffle=False, decode=True)
    size = data.get_dataset_size()
    assert size == 4

    data = ds.CelebADataset(DATA_DIR, shuffle=False, decode=True, usage="train")
    size = data.get_dataset_size()
    assert size == 2

    data = ds.CelebADataset(DATA_DIR, shuffle=False, decode=True, usage="valid")
    size = data.get_dataset_size()
    assert size == 1

    data = ds.CelebADataset(DATA_DIR, shuffle=False, decode=True, usage="test")
    size = data.get_dataset_size()
    assert size == 1

if __name__ == '__main__':
    test_celeba_dataset_label()
    test_celeba_dataset_op()
    test_celeba_dataset_ext()
    test_celeba_dataset_distribute()
    test_celeba_get_dataset_size()
