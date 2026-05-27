from pathlib import Path

class PathResolver:
    @staticmethod
    def get_lung_cancer_data_path() -> Path:
        repo_root = Path("..")
        return repo_root.joinpath('data/lung-cancer-prediction')
