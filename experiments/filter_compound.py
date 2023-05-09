import sys
BASE_PATH = '/projectnb/multilm/yusuf/CS543-final-project'
sys.path.append(BASE_PATH)

import hydra
from omegaconf import DictConfig
from datasets import Dataset
from datasets import load_dataset
from filtering.one_pass_compound_filter import CompoundCounter, CompoundDistanceFilter, CompoundRuleFilter, tdv_distance, wasserstein_distance, jsd_distance


@hydra.main(version_base="1.2", config_path="/projectnb/multilm/yusuf/CS543-final-project/configs", config_name="base.yaml")
def main(config: DictConfig):

    if config.distance == 'jsd':
        distance = jsd_distance
    elif config.distance == 'wasserstein':
        distance = wasserstein_distance
    elif config.distance == 'tdv':
        distance = tdv_distance
    
    if config.filter == 'rule':
        filter = CompoundRuleFilter(CompoundCounter, filter_size=config.filter_size, n_counters=config.n_counters, cold_start=config.cold_start)
    elif config.filter == 'distance':
        filter = CompoundDistanceFilter(CompoundCounter, distance, filter_size=config.filter_size, n_counters=config.n_counters, cold_start=config.cold_start)

    stream = load_dataset('the_pile', split='train', streaming=True)
    stream.shuffle(seed=34)
    filter.filter(stream)

    sampled_dataset = Dataset.from_dict({'text': filter.sentences})
    sampled_dataset.save_to_disk(f'{BASE_PATH}/data/{config.filter}_{config.distance}')

if __name__ == "__main__":
    main()