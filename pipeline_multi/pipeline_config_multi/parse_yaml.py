#!/usr/bin/env python3
"""
YAML Config Parser for Multi-Robot Shell Scripts

Usage:
    # Parse entire config to shell variables
    eval "$(python3 pipeline_multi/pipeline_config_multi/parse_yaml.py detection_config.yaml)"

    # Get specific value
    python3 pipeline_multi/pipeline_config_multi/parse_yaml.py detection_config.yaml detection_timeout

    # Parse multi_robot_config.yaml (extracts robot IDs)
    eval "$(python3 pipeline_multi/pipeline_config_multi/parse_yaml.py multi_robot_config.yaml)"
"""

import sys
import yaml
from pathlib import Path


def flatten_dict(d, parent_key='', sep='_'):
    """Flatten nested dict to single level with underscore-separated keys.

    Returns: dict with values as tuples (value, is_list)
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Check if it's a list of dicts (like robots config)
            if v and isinstance(v[0], dict):
                # Special handling for robot list - extract robot_ids
                if 'robot_id' in v[0]:
                    robot_ids = [str(item['robot_id']) for item in v if item.get('enabled', True)]
                    items.append((f"{new_key}_ids", (' '.join(robot_ids), True)))
                # Also flatten each robot's config
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(flatten_dict(item, f"{new_key}_{i}", sep=sep).items())
            else:
                # Simple list - convert to space-separated string for bash array
                str_items = [f'"{item}"' if isinstance(item, str) else str(item) for item in v]
                items.append((new_key, (' '.join(str_items), True)))
        else:
            items.append((new_key, (v, False)))
    return dict(items)


def to_shell_var(key, value, is_list=False):
    """Convert key-value pair to shell variable assignment."""
    key = key.upper()
    if isinstance(value, bool):
        return f'{key}={"true" if value else "false"}'
    elif isinstance(value, (int, float)):
        return f'{key}={value}'
    elif isinstance(value, str):
        # If this was originally a list, output as bash array
        if is_list:
            return f'{key}=({value})'
        return f'{key}="{value}"'
    else:
        return f'{key}="{value}"'


def extract_robot_ids(config):
    """Extract enabled robot IDs from config."""
    robot_ids = []

    # Check robots list at top level
    if 'robots' in config and isinstance(config['robots'], list):
        for robot in config['robots']:
            if isinstance(robot, dict) and robot.get('enabled', True):
                if 'robot_id' in robot:
                    robot_ids.append(robot['robot_id'])

    return robot_ids


def main():
    if len(sys.argv) < 2:
        print("Usage: parse_yaml.py <config.yaml> [key]", file=sys.stderr)
        sys.exit(1)

    config_path = Path(sys.argv[1])

    # Support both absolute and relative paths
    if not config_path.exists():
        # Try relative to script directory
        script_dir = Path(__file__).parent
        config_path = script_dir / config_path.name
        if not config_path.exists():
            print(f"Error: Config file not found: {sys.argv[1]}", file=sys.stderr)
            sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    # Flatten nested config
    flat_config = flatten_dict(config)

    # Add special handling for robot_ids if in multi_robot_config
    robot_ids = extract_robot_ids(config)
    if robot_ids:
        flat_config['robot_ids'] = (' '.join(str(rid) for rid in robot_ids), True)

    # If specific key requested, return just that value
    if len(sys.argv) >= 3:
        key = sys.argv[2].lower()
        if key in flat_config:
            value, is_list = flat_config[key]
            if isinstance(value, bool):
                print("true" if value else "false")
            else:
                print(value)
        else:
            print(f"Error: Key '{key}' not found in config", file=sys.stderr)
            sys.exit(1)
    else:
        # Output all variables for eval
        for key, (value, is_list) in flat_config.items():
            print(to_shell_var(key, value, is_list))


if __name__ == "__main__":
    main()
