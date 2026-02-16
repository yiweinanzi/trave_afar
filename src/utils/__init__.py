"""
工具模块
"""
from .logger import (
    get_logger,
    setup_logger,
    get_logger_level_from_env,
    configure_logging_from_config,
    LogColors,
    ColoredFormatter,
    JSONFormatter,
)
from .model_downloader import download_bge_m3
from .data_alignment import (
    collect_training_poi_ids,
    summarize_dataset_embedding_alignment,
)

# 实验追踪模块
try:
    from .experiment import (
        ExperimentTracker,
        MLflowTracker,
        JSONTracker,
        NullTracker,
        ExperimentManager,
        ExperimentConfig,
        MLflowExperiment,
        ModelRegistry,
        MLflowCallback,
        compare_experiments,
        get_best_run,
        track_experiment,
        experiment_context,
        setup_mlflow_server,
        auto_detect_tracking_uri,
    )
    _experiment_available = True
except ImportError:
    _experiment_available = False

__all__ = [
    # 日志相关
    'download_bge_m3',
    'get_logger',
    'setup_logger',
    'get_logger_level_from_env',
    'configure_logging_from_config',
    'LogColors',
    'ColoredFormatter',
    'JSONFormatter',
    'collect_training_poi_ids',
    'summarize_dataset_embedding_alignment',
    # 实验追踪
    'ExperimentTracker',
    'MLflowTracker',
    'JSONTracker',
    'NullTracker',
    'ExperimentManager',
    'ExperimentConfig',
    'MLflowExperiment',
    'ModelRegistry',
    'MLflowCallback',
    'compare_experiments',
    'get_best_run',
    'track_experiment',
    'experiment_context',
    'setup_mlflow_server',
    'auto_detect_tracking_uri',
]
