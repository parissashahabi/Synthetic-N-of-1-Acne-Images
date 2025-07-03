"""
Generic config parser to add all dataclass fields as command line arguments.
"""
import argparse
from dataclasses import fields
from typing import get_type_hints, get_origin, get_args


def add_config_args(parser: argparse.ArgumentParser, config_class, prefix: str = ""):
    """
    Add all dataclass fields as command line arguments.
    
    Args:
        parser: ArgumentParser instance
        config_class: Dataclass configuration class
        prefix: Prefix for argument names (e.g., "model-" or "train-")
    """
    # Get type hints for proper type conversion
    type_hints = get_type_hints(config_class)
    
    for field in fields(config_class):
        field_name = field.name
        field_type = type_hints.get(field_name, field.type)
        default_value = field.default
        
        # Create argument name with prefix
        arg_name = f"--{prefix}{field_name.replace('_', '-')}" if prefix else f"--{field_name.replace('_', '-')}"
        
        # Handle different types
        if field_type == bool:
            # For boolean fields, use store_true/store_false
            if default_value:
                parser.add_argument(arg_name, action='store_false', 
                                  help=f"Disable {field_name} (default: {default_value})")
            else:
                parser.add_argument(arg_name, action='store_true',
                                  help=f"Enable {field_name} (default: {default_value})")
        
        elif field_type in (int, float, str):
            parser.add_argument(arg_name, type=field_type, default=default_value,
                              help=f"{field_name} (default: {default_value})")
        
        elif get_origin(field_type) == tuple:
            # Handle tuples like Tuple[int, ...]
            inner_type = get_args(field_type)[0] if get_args(field_type) else int
            parser.add_argument(arg_name, type=inner_type, nargs='+', default=default_value,
                              help=f"{field_name} (default: {default_value})")
        
        elif hasattr(field_type, '__origin__') and field_type.__origin__ == list:
            # Handle lists
            inner_type = get_args(field_type)[0] if get_args(field_type) else str
            parser.add_argument(arg_name, type=inner_type, nargs='+', default=default_value,
                              help=f"{field_name} (default: {default_value})")
        
        else:
            # For other types, treat as string and let the config handle conversion
            parser.add_argument(arg_name, type=str, default=default_value,
                              help=f"{field_name} (default: {default_value})")


def update_config_from_args(config_instance, args, prefix: str = ""):
    """
    Update config instance with values from parsed arguments.
    
    Args:
        config_instance: Instance of dataclass config
        args: Parsed arguments from argparse
        prefix: Prefix used when adding arguments
    """
    for field in fields(config_instance):
        field_name = field.name
        arg_name = f"{prefix}{field_name}" if prefix else field_name
        
        if hasattr(args, arg_name):
            arg_value = getattr(args, arg_name)
            if arg_value is not None:
                # Convert tuples if needed
                if hasattr(field.type, '__origin__') and field.type.__origin__ == tuple:
                    if isinstance(arg_value, list):
                        arg_value = tuple(arg_value)
                
                setattr(config_instance, field_name, arg_value)
    
    return config_instance