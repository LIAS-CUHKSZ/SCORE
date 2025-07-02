import argparse
from common.utils.anno import split_views_by_coverage, find_similar_views_by_coverage
from common.utils.utils import load_yaml_munch, read_txt_list
from common.scene_release import ScannetppScene_Release

from pathlib import Path
from tqdm import tqdm
import hydra
import json

def main(args):
    cfg = load_yaml_munch(args.config_file)
    
    scene_ids = cfg.scene_ids
    print('Scenes in list:', len(scene_ids))
    
    rasterout_dir = Path(cfg.rasterout_dir) / cfg.image_type
    best_view_cache_dir = Path(cfg.best_view_cache_dir) / cfg.image_type
    best_view_cache_dir.mkdir(parents=True, exist_ok=True)
    
    for scene_id in tqdm(scene_ids, desc='scene'):
        print(f'Processing: {scene_id}')
        scene = ScannetppScene_Release(scene_id, data_root=Path(cfg.data_root))
        
        train_views, test_views = split_views_by_coverage(scene, rasterout_dir, cfg.image_type, cfg.subsample_factor, False, cfg.split_ratio, cfg.coverage_threshold)
        similiar_views = find_similar_views_by_coverage(scene, rasterout_dir, cfg.image_type, cfg.subsample_factor, False, cfg.num_neighbors)
        
        with open(best_view_cache_dir / f'{scene_id}_train.txt', 'w') as f:
            for view in train_views:
                f.write(f"{view}\n")

        with open(best_view_cache_dir / f'{scene_id}_test.txt', 'w') as f:
            for view in test_views:
                f.write(f"{view}\n")
            
        with open(best_view_cache_dir / f'{scene_id}_similar.json', 'w') as f:
            json.dump(similiar_views, f, indent=4)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config_file", help="Path to config file")
    args = p.parse_args()
    
    main(args)