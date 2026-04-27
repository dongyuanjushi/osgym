import os
import logging
from typing import Union

from ..schema import evaluator

logger = logging.getLogger("desktopenv.getters.info")


@evaluator(
    role="getter",
)
def get_vm_screen_size(env, config: dict) -> dict:
    return env.controller.get_vm_screen_size()


@evaluator(
    role="getter",
    config={
        "app_class_name": "str: X11/window-manager class name of the target application whose window size to query",
    },
)
def get_vm_window_size(env, config: dict) -> dict:
    return env.controller.get_vm_window_size(app_class_name=config["app_class_name"])


@evaluator(
    role="getter",
    config={
        "dest": "str: destination file name under the local cache dir to save the fetched wallpaper bytes",
    },
)
def get_vm_wallpaper(env, config: dict) -> Union[str, bytes]:
    _path = os.path.join(env.cache_dir, config["dest"])

    content = env.controller.get_vm_wallpaper()
    
    # Check if content is None or empty
    if content is None:
        logger.error("Failed to get VM wallpaper: controller returned None")
        # Create an empty file to prevent downstream errors
        with open(_path, "wb") as f:
            f.write(b"")
        return _path
    
    if not isinstance(content, bytes):
        logger.error(f"Invalid wallpaper content type: {type(content)}, expected bytes")
        # Create an empty file to prevent downstream errors
        with open(_path, "wb") as f:
            f.write(b"")
        return _path
    
    if len(content) == 0:
        logger.warning("VM wallpaper content is empty")
        # Create an empty file to prevent downstream errors
        with open(_path, "wb") as f:
            f.write(b"")
        return _path

    with open(_path, "wb") as f:
        f.write(content)

    return _path


@evaluator(
    role="getter",
    config={
        "path": "str: absolute directory path on the VM whose directory tree to list",
    },
)
def get_list_directory(env, config: dict) -> dict:
    return env.controller.get_vm_directory_tree(config["path"])
