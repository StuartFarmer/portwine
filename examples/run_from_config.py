#!/usr/bin/env python3
"""
Command-line entrypoint to run Portwine ExecutionBase from a config file (YAML or JSON).

Usage:
  python examples/run_from_config.py --config path/to/config.yaml
"""

import argparse
import importlib
import os
import sys
import logging

# JSON and YAML support
import json
try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. Install via 'pip install pyyaml'", file=sys.stderr)
    sys.exit(1)

from portwine.execution import ExecutionBase
from portwine.scheduler import daily_schedule

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def load_class(path: str):
    """Dynamically import a class from its fully-qualified name."""
    module_name, class_name = path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def main():
    parser = argparse.ArgumentParser(
        description="Run Portwine ExecutionBase from a YAML/JSON config file"
    )
    parser.add_argument(
        '-c', '--config', required=True,
        help='Path to YAML or JSON config file'
    )
    args = parser.parse_args()

    config_path = os.path.expanduser(args.config)
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    ext = os.path.splitext(config_path)[1].lower()
    if ext in ('.yaml', '.yml'):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    elif ext == '.json':
        with open(config_path) as f:
            cfg = json.load(f)
    else:
        logger.error("Unsupported config file extension: %s", ext)
        sys.exit(1)

    # 1) Strategy
    strat_spec = cfg.get('strategy', {})
    StrategyCls = load_class(strat_spec['class'])
    strat_params = strat_spec.get('params', {})
    strategy = StrategyCls(**strat_params)
    logger.info(f"Loaded strategy %s", strat_spec['class'])

    # 2) Data Loader
    dl_spec = cfg.get('data_loader', {})
    DataLoaderCls = load_class(dl_spec['class'])
    dl_params = dl_spec.get('params', {})
    data_loader = DataLoaderCls(**dl_params)
    logger.info(f"Loaded data loader %s", dl_spec['class'])

    # 3) Broker
    broker_spec = cfg.get('broker', {})
    BrokerCls = load_class(broker_spec['class'])
    broker_params = broker_spec.get('params', {})
    broker = BrokerCls(**broker_params)
    logger.info(f"Loaded broker %s", broker_spec['class'])

    # 4) Alternative Data Loader (optional)
    alt_loader = None
    for alt_spec in cfg.get('alternative_data', []) or []:
        AltCls = load_class(alt_spec['class'])
        alt_params = alt_spec.get('params', {})
        # Special-case base_loader to allow referencing primary loader
        if 'base_loader' in alt_params and isinstance(alt_params['base_loader'], str):
            bl = alt_params['base_loader']
            if bl == dl_spec['class']:
                alt_params['base_loader'] = data_loader
            else:
                BaseLoaderCls = load_class(bl)
                alt_params['base_loader'] = BaseLoaderCls()
        alt_loader = AltCls(**alt_params)
        logger.info(f"Loaded alternative data loader %s", alt_spec['class'])
        break  # only support the first for now

    # 5) Execution parameters
    exec_cfg = cfg.get('execution', {})
    exec_kwargs = {
        'alternative_data_loader': alt_loader,
        'min_change_pct': exec_cfg.get('min_change_pct', 0.01),
        'min_order_value': exec_cfg.get('min_order_value', 1.0),
    }
    executor = ExecutionBase(strategy, data_loader, broker, **exec_kwargs)
    logger.info("Executor initialized")

    # 6) Schedule
    sched_cfg = cfg.get('schedule', {})
    preset = sched_cfg.get('preset')
    if preset:
        logger.error("Preset schedules are not yet implemented: %s", preset)
        sys.exit(1)

    schedule = daily_schedule(
        after_open_minutes=sched_cfg.get('after_open_minutes'),
        before_close_minutes=sched_cfg.get('before_close_minutes'),
        calendar_name=sched_cfg.get('calendar_name', 'NYSE'),
        start_date=sched_cfg.get('start_date'),
        end_date=sched_cfg.get('end_date'),
        interval_seconds=sched_cfg.get('interval_seconds'),
        inclusive=sched_cfg.get('inclusive', False),
    )
    logger.info("Schedule constructed")

    # 7) Run
    logger.info("Starting execution run...")
    executor.run(schedule)
    logger.info("Execution run complete")


if __name__ == '__main__':
    main() 