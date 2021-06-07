from auto_augment.autoaug.experiment.experiment import AutoAugExperiment
from auto_augment.autoaug.utils.yaml_config import get_config
from hub_fitter import HubFitterClassifer
import os
import argparse
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    help="config file",
)
parser.add_argument(
    "--workspace",
    default=None,
    help="work_space",
)


def main():
    search_test()


def search_test():
    args = parser.parse_args()
    config = args.config
    config = get_config(config, show=True)
    task_config = config.task_config
    data_config = config.data_config
    resource_config = config.resource_config
    algo_config = config.algo_config
    search_space = config.get("search_space", None)

    if args.workspace is not None:
        task_config["workspace"] = args.workspace
    workspace = task_config["workspace"]

    # 算法，任务，资源，数据，搜索空间(optional)配置导入，
    exper = AutoAugExperiment.create(
        algo_config=algo_config,
        task_config=task_config,
        resource_config=resource_config,
        data_config=data_config,
        search_space=search_space,
        fitter=HubFitterClassifer)
    result = exper.search()  # 开始搜索任务
    policy = result.get_best_policy()  # 最佳策略获取， policy格式见 搜索结果应用格式
    print("policy is:{}".format(policy))
    dump_path = os.path.join(workspace, "auto_aug_config.json")
    result.dump_best_policy(path=dump_path)


if __name__ == "__main__":
    main()
