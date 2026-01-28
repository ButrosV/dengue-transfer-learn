from __future__ import annotations  # For forward refs like 'ProjectConfig'
from pydantic import BaseModel
from pathlib import Path
from typing import Dict, Any, ClassVar, List
import yaml


def _nested_dict_valuesearch(dict_in: dict[str, Any], 
    dict_out: Dict[str, Any], value_type=type[str]) -> None:
    """
    Recursively extract all values of specified type from nested dictionary.
    Adding keys with matching value types to output dict.
    :param dict_in: Input dictionary to search (may be nested)
    :param dict_out: Output dictionary to collect matching key-value pairs
    :param value_type: Type to extract (default: str)
    """
    for key, value in dict_in.items():
        if isinstance(value, dict):
           value = _nested_dict_valuesearch(value, dict_out, value_type)
        elif isinstance(value, value_type):
            dict_out[key] = value



class DataConfig(BaseModel):
    """Configuration for data directories and filenames."""
    dirs: Dict[str, Path]
    files: Dict[str, Path]
    

    class Config:
        """In case more complex data types added - eg. dataframes.
        For non-defined params, if found in config.yaml - free form, no type enforcement
        """
        arbitrary_types_allowed = True
        extra = "allow"

class PreprocesConfig(BaseModel):
    """Configuration for model preprocessing variables."""
    feature_groups: Dict[str, Any]
    outlier_perc: Dict[str, float]
    nan_threshold: float


    class Config:
       """For non-defineed params, if found in config.yaml - free form, no type enforcment."""
       extra = "allow"


class ModelConfig(BaseModel):
    """Configuration for model training hyperparameters."""
    learning_rate: float
    batch_size: int = 32
    epochs: int = 10


    class Config:
       """For non-defineed params, if found in config.yaml - free form, no type enforcment."""
       extra = "allow"


class ProjectConfig(BaseModel):
    """Top-level project configuration combining data and model settings."""
    data: DataConfig
    model: ModelConfig
    preprocess: PreprocesConfig

    CONFIG_RELATIVE: ClassVar[str] = "config.yaml"
    """Default relative path to YAML config from project root.
    To override: ProjectConfig.CONFIG_RELATIVE = "custom/config.yaml"""


    class Config:
        """For non-defined params, if found in config.yaml - free form, no type enforcement."""
        extra = "allow"

    @classmethod  # for consistency with `load_configuration`, as both work hand-in-hand
    def root_path_resolver(cls):
        """
        Resolve project root path.
        Strategy priority:
        1. PROJECT_ROOT environment variable defined by user.
        2. __file__ path from script location  
        3. Walk up current working directory to find 'src/' directory for Jupyter
        
        :return: Absolute Path object pointing to project root directory
        :raises RuntimeError: If project root cannot be automatically determined
        """
        import os
        project_root = os.getenv("PROJECT_ROOT")
        if project_root:
            return Path(project_root)

        try:
            return Path(__file__).resolve().parents[1]
        except NameError:
            cwd = Path.cwd()
            current = cwd
            for root_candidate in range(3):
                if (current / "src").exists():
                    return current
                if not current.name:  # in case global root reached
                    break
                current = current.parent
            raise RuntimeError("""Cannot find project root. Set PROJECT_ROOT env var
                                or provide 'config_path' for 
                                ProjectConfig.load_configuration().""")


    @staticmethod  # just a helper for yaml file path resolution - no need for self 
    def _process_section_values(config_dict: dict[str, Any], 
        process_condition: str, root: Path | None = None) -> None:
        """
        Process directory/file sections in config, extracting and resolving relative paths.
        Finds sections containing `process_condition` 
            (eg 'dir' or 'file') extracts string values recursively,
            and converts relative paths to absolute Paths if `root` is given value.
        :param config_dict: Configuration dictionary to process in-place
        :param process_condition: Key substring to match (eg 'dir' or 'file')
        :param root: Optional root Path for absolute path resolution
        """
        for section in config_dict.values():
            if isinstance(section, dict):
                for key, value in section.items():
                    if process_condition in key:
                        temp_section = dict()
                        _nested_dict_valuesearch(value, temp_section, value_type=str)
                        if isinstance(root, Path):
                            temp_section = {k: root / v for k,v in temp_section.items()}
                        section[key] = temp_section



    @classmethod  # Factory Method pattern, no need to instantiate clacc before use
    def load_configuration(cls, config_path: str | None = None) -> 'ProjectConfig':
        """
        Load YAML configuration file and convert directory paths to absolute Path 
        objects relative to project root. Automatically handle script and 
        Jupyter notebook environments.
        
        :param config_path: Relative or absolute path to the YAML configuration file. 
                           If None, automatically loads 'src/config.yaml' from project root.
        :return: Validated ProjectConfig instance with all paths resolved to absolute paths
        :raises FileNotFoundError: If specified config file doesn't exist
        :raises RuntimeError: If project root directory cannot be resolved
        """
        project_root = cls.root_path_resolver()
        if config_path is None:
            path = project_root / cls.CONFIG_RELATIVE
        else:
            path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found at {path}.")
        with open(path) as f:
            config_dict = yaml.safe_load(f)

        ProjectConfig._process_section_values(config_dict, process_condition="dir", root=project_root)
        ProjectConfig._process_section_values(config_dict, process_condition="file")

        return cls(**config_dict)


# Usage example:
# cnfg = ProjectConfig.load_configuration()
# print(cnfg.model.learning_rate)  # Type-safe access
