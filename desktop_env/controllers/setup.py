import json
import logging
import os
import os.path
import platform
import shutil
import sqlite3
import tempfile
import time
import traceback
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Any, Union, Optional
from typing import Dict, List

import requests
from playwright.async_api import async_playwright, TimeoutError
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive, GoogleDriveFile, GoogleDriveFileList
from requests_toolbelt.multipart.encoder import MultipartEncoder

from desktop_env.controllers.python import PythonController
from desktop_env.evaluators.metrics.utils import compare_urls
from desktop_env.evaluators.schema import SETUP_SCHEMA_ATTR, setup_action

logger = logging.getLogger("desktopenv.setup")

FILE_PATH = os.path.dirname(os.path.abspath(__file__))

MAX_RETRIES = 20


class SetupController:
    def __init__(self, vm_ip: str, server_port: int = 5000, chromium_port: int = 9222, vlc_port: int = 8080, cache_dir: str = "cache", client_password: str = "", screen_width: int = 1920, screen_height: int = 1080):
        self.vm_ip: str = vm_ip
        self.server_port: int = server_port
        self.chromium_port: int = chromium_port
        self.vlc_port: int = vlc_port
        self.http_server: str = f"http://{vm_ip}:{server_port}"
        self.http_server_setup_root: str = f"http://{vm_ip}:{server_port}/setup"
        self.cache_dir: str = cache_dir
        self.use_proxy: bool = False
        self.client_password: str = client_password
        self.screen_width: int = screen_width
        self.screen_height: int = screen_height

    def reset_cache_dir(self, cache_dir: str):
        self.cache_dir = cache_dir

    # async def setup(self, config: List[Dict[str, Any]]):
    #     """
    #     Args:
    #         config (List[Dict[str, Any]]): list of dict like {str: Any}. each
    #           config dict has the structure like
    #             {
    #                 "type": str, corresponding to the `_{:}_setup` methods of
    #                   this class
    #                 "parameters": dick like {str, Any} providing the keyword
    #                   parameters
    #             }
    #     """

    #     for cfg in config:
    #         config_type: str = cfg["type"]
    #         parameters: Dict[str, Any] = cfg["parameters"]

    #         # Assumes all the setup the functions should follow this name
    #         # protocol
    #         setup_function: str = "_{:}_setup".format(config_type)
    #         assert hasattr(self, setup_function), f'Setup controller cannot find init function {setup_function}'
    #         if config_type in ["chrome_open_tabs", "chrome_close_tabs", "login"]:
    #             await getattr(self, setup_function)(**parameters)
    #         else:
    #             getattr(self, setup_function)(**parameters)

    #         logger.info("SETUP: %s(%s)", setup_function, str(parameters))
    
    async def setup(self, config: List[str], use_proxy: bool = False) -> bool:
        """
        Execute refactored config commands that are Python function call strings.

        Args:
            config (List[str]): list of Python function call strings like
              "_launch_setup(command=['google-chrome', '--remote-debugging-port=1337'])"
              Each string should be a direct function call that can be executed with eval().
            use_proxy (bool): whether to use proxy for browser operations

        Returns:
            bool: True if all commands executed successfully, False otherwise
        """
        self.use_proxy = use_proxy
        
        MAX_RETRIES = 10
        # make sure connection can be established
        logger.info(f"try to connect {self.http_server}")
        retry = 0
        while retry < MAX_RETRIES:
            try:
                _ = requests.get(self.http_server + "/terminal")
                break
            except:
                time.sleep(5)
                retry += 1
                logger.info(f"retry: {retry}/{MAX_RETRIES}")

            if retry == MAX_RETRIES:
                return False

        # Auto-discover decorated setup actions. Every method tagged with
        # ``@setup_action`` is registered as a callable for refactored config
        # strings; adding a new helper only needs the decorator.
        local_namespace: Dict[str, Any] = {
            name: getattr(self, name)
            for name in dir(self)
            if name.startswith("_") and name.endswith("_setup")
            and getattr(getattr(type(self), name, None), SETUP_SCHEMA_ATTR, None) is not None
        }

        for i, cmd in enumerate(config):
            try:
                logger.info(f"Executing refactored setup step {i+1}/{len(config)}: {cmd}")
                # INSERT_YOUR_CODE
                func_name = cmd.split('(')[0].strip()
                func_obj = local_namespace.get(func_name)
                # Check if function is a coroutine and needs to be awaited
                if func_obj and asyncio.iscoroutinefunction(func_obj):
                    # Await using running loop; assumes 'setup' is an async function
                    await eval(cmd, {"__builtins__": {}}, local_namespace)
                else:
                    eval(cmd, {"__builtins__": {}}, local_namespace)
                logger.info(f"SETUP COMPLETED: {cmd}")
            except Exception as e:
                logger.error(f"SETUP FAILED at step {i+1}/{len(config)}: {cmd}")
                logger.error(f"Error details: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise Exception(f"Setup step {i+1} failed: {cmd} - {e}") from e

        return True

    @setup_action(
        summary="Fetch remote URLs to the host and push them to literal VM paths.",
        args={
            "files": "list of {'url': str, 'path': str}; each url is downloaded then uploaded to path on the VM",
        },
    )
    async def _download_setup(self, files: List[Dict[str, str]]):
        """
        Args:
            files (List[Dict[str, str]]): files to download. lisf of dict like
              {
                "url": str, the url to download
                "path": str, the path on the VM to store the downloaded file
              }
        """

        # if not config:
        # return
        # if not 'download' in config:
        # return
        # for url, path in config['download']:
        for f in files:
            url: str = f["url"]
            path: str = f["path"]
            cache_path: str = os.path.join(self.cache_dir, "{:}_{:}".format(
                uuid.uuid5(uuid.NAMESPACE_URL, url),
                os.path.basename(path)))
            if not url or not path:
                raise Exception(f"Setup Download - Invalid URL ({url}) or path ({path}).")

            if not os.path.exists(cache_path):
                max_retries = 3
                downloaded = False
                e = None
                for i in range(max_retries):
                    try:
                        response = requests.get(url, stream=True)
                        response.raise_for_status()

                        with open(cache_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        logger.info("File downloaded successfully")
                        downloaded = True
                        break

                    except requests.RequestException as e:
                        logger.error(
                            f"Failed to download {url} caused by {e}. Retrying... ({max_retries - i - 1} attempts left)")
                if not downloaded:
                    raise requests.RequestException(f"Failed to download {url}. No retries left. Error: {e}")

            form = MultipartEncoder({
                "file_path": path,
                "file_data": (os.path.basename(path), open(cache_path, "rb"))
            })
            headers = {"Content-Type": form.content_type}
            logger.debug(form.content_type)

            # send request to server to upload file
            try:
                logger.debug("REQUEST ADDRESS: %s", self.http_server + "/setup" + "/upload")
                response = requests.post(self.http_server + "/setup" + "/upload", headers=headers, data=form)
                if response.status_code == 200:
                    logger.info("Command executed successfully: %s", response.text)
                else:
                    logger.error("Failed to upload file. Status code: %s", response.text)
            except requests.exceptions.RequestException as e:
                logger.error("An error occurred while trying to send the request: %s", e)

    @setup_action(
        summary="Upload host-local files to literal VM paths.",
        args={
            "files": "list of {'local_path': str, 'path': str}; each local_path on the host is copied to path on the VM",
        },
    )
    async def _upload_file_setup(self, files: List[Dict[str, str]]):
        """
        Args:
            files (List[Dict[str, str]]): files to download. lisf of dict like
              {
                "local_path": str, the local path to the file to upload
                "path": str, the path on the VM to store the downloaded file
              }
        """
        for f in files:
            local_path: str = f["local_path"]
            path: str = f["path"]

            if not os.path.exists(local_path):
                logger.error(f"Setup Upload - Invalid local path ({local_path}).")
                return

            form = MultipartEncoder({
                "file_path": path,
                "file_data": (os.path.basename(path), open(local_path, "rb"))
            })
            headers = {"Content-Type": form.content_type}
            logger.debug(form.content_type)

            # send request to server to upload file
            try:
                logger.debug("REQUEST ADDRESS: %s", self.http_server + "/setup" + "/upload")
                response = requests.post(self.http_server + "/setup" + "/upload", headers=headers, data=form)
                if response.status_code == 200:
                    logger.info("Command executed successfully: %s", response.text)
                else:
                    logger.error("Failed to upload file. Status code: %s", response.text)
            except requests.exceptions.RequestException as e:
                logger.error("An error occurred while trying to send the request: %s", e)

    @setup_action(
        summary="Set the desktop wallpaper to a file already present on the VM.",
        args={
            "path": "absolute path on the VM to an image file already uploaded/installed",
        },
    )
    async def _change_wallpaper_setup(self, path: str):
        # if not config:
        # return
        # if not 'wallpaper' in config:
        # return

        # path = config['wallpaper']
        if not path:
            raise Exception(f"Setup Wallpaper - Invalid path ({path}).")

        payload = json.dumps({"path": path})
        headers = {
            'Content-Type': 'application/json'
        }

        # send request to server to change wallpaper
        try:
            response = requests.post(self.http_server + "/setup" + "/change_wallpaper", headers=headers, data=payload)
            if response.status_code == 200:
                logger.info("Command executed successfully: %s", response.text)
            else:
                logger.error("Failed to change wallpaper. Status code: %s", response.text)
        except requests.exceptions.RequestException as e:
            logger.error("An error occurred while trying to send the request: %s", e)

    async def _tidy_desktop_setup(self, **config):
        raise NotImplementedError()

    @setup_action(
        summary="Open a file on the VM with its default desktop application (xdg-open).",
        args={
            "path": "absolute path on the VM to a file that already exists",
        },
    )
    async def _open_setup(self, path: str):
        # if not config:
        # return
        # if not 'open' in config:
        # return
        # for path in config['open']:
        if not path:
            raise Exception(f"Setup Open - Invalid path ({path}).")

        payload = json.dumps({"path": path})
        headers = {
            'Content-Type': 'application/json'
        }

        # send request to server to open file
        try:
            response = requests.post(self.http_server + "/setup" + "/open_file", headers=headers, data=payload)
            if response.status_code == 200:
                logger.info("Command executed successfully: %s", response.text)
            else:
                logger.error("Failed to open file. Status code: %s", response.text)
        except requests.exceptions.RequestException as e:
            logger.error("An error occurred while trying to send the request: %s", e)

    @setup_action(
        summary="Launch a process on the VM (non-blocking; e.g. an application).",
        args={
            "command": "argv list (preferred) or shell string when shell=True",
            "shell?": "interpret command via shell instead of argv (default False)",
        },
    )
    async def _launch_setup(self, command: Union[str, List[str]], shell: bool = False):
        if not command:
            raise Exception("Empty command to launch.")

        if not shell and isinstance(command, str) and len(command.split()) > 1:
            logger.warning("Command should be a list of strings. Now it is a string. Will split it by space.")
            command = command.split()

        payload = json.dumps({"command": command, "shell": shell})
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(self.http_server + "/setup" + "/launch", headers=headers, data=payload)
            if response.status_code == 200:
                logger.info("Command executed successfully: %s", response.text)
            else:
                logger.error("Failed to launch application. Status code: %s", response.text)
        except requests.exceptions.RequestException as e:
            logger.error("An error occurred while trying to send the request: %s", e)

    @setup_action(
        summary="Run a command on the VM and wait for it to finish, optionally polling until a condition.",
        args={
            "command": "argv list (preferred) or shell string when shell=True; supports {CLIENT_PASSWORD}/{SCREEN_*} placeholders",
            "stdout?": "filename inside cache_dir to capture stdout to (empty = discard)",
            "stderr?": "filename inside cache_dir to capture stderr to (empty = discard)",
            "shell?": "interpret command via shell instead of argv (default False)",
            "until?": "{'returncode'|'stdout'|'stderr': value}; retry until condition matches or 5 failures",
        },
    )
    async def _execute_setup(
            self,
            command: List[str],
            stdout: str = "",
            stderr: str = "",
            shell: bool = False,
            until: Optional[Dict[str, Any]] = None
    ):
        if not command:
            raise Exception("Empty command to launch.")

        until: Dict[str, Any] = until or {}
        terminates: bool = False
        nb_failings = 0

        def replace_screen_env_in_command(cmd):
            password = self.client_password
            width = self.screen_width
            height = self.screen_height
            width_half = str(width // 2)
            height_half = str(height // 2)
            if isinstance(cmd, str):
                new_cmd = cmd.replace("{CLIENT_PASSWORD}", password)
                new_cmd = new_cmd.replace("{SCREEN_WIDTH_HALF}", width_half)
                new_cmd = new_cmd.replace("{SCREEN_HEIGHT_HALF}", height_half)
                new_cmd = new_cmd.replace("{SCREEN_WIDTH}", str(width))
                new_cmd = new_cmd.replace("{SCREEN_HEIGHT}", str(height))
                return new_cmd
            else:
                new_cmd_list = []
                for item in cmd:
                    item = item.replace("{CLIENT_PASSWORD}", password)
                    item = item.replace("{SCREEN_WIDTH_HALF}", width_half)
                    item = item.replace("{SCREEN_HEIGHT_HALF}", height_half)
                    item = item.replace("{SCREEN_WIDTH}", str(width))
                    item = item.replace("{SCREEN_HEIGHT}", str(height))
                    new_cmd_list.append(item)
                return new_cmd_list

        command = replace_screen_env_in_command(command)
        payload = json.dumps({"command": command, "shell": shell})
        headers = {"Content-Type": "application/json"}

        while not terminates:
            try:
                response = requests.post(self.http_server + "/setup" + "/execute", headers=headers, data=payload)
                if response.status_code == 200:
                    results: Dict[str, str] = response.json()
                    if stdout:
                        with open(os.path.join(self.cache_dir, stdout), "w") as f:
                            f.write(results["output"])
                    if stderr:
                        with open(os.path.join(self.cache_dir, stderr), "w") as f:
                            f.write(results["error"])
                    logger.info("Command executed successfully: %s -> %s"
                                , " ".join(command) if isinstance(command, list) else command
                                , response.text
                                )
                else:
                    logger.error("Failed to launch application. Status code: %s", response.text)
                    results = None
                    nb_failings += 1
            except requests.exceptions.RequestException as e:
                logger.error("An error occurred while trying to send the request: %s", e)
                traceback.print_exc()

                results = None
                nb_failings += 1

            if len(until) == 0:
                terminates = True
            elif results is not None:
                terminates = "returncode" in until and results["returncode"] == until["returncode"] \
                             or "stdout" in until and until["stdout"] in results["output"] \
                             or "stderr" in until and until["stderr"] in results["error"]
            terminates = terminates or nb_failings >= 5
            if not terminates:
                time.sleep(0.3)

    @setup_action(
        summary="Alias of _execute_setup; runs a command on the VM and waits for it.",
        args={
            "command": "argv list (preferred) or shell string when shell=True",
            "kwargs?": "forwarded to _execute_setup (stdout/stderr/shell/until)",
        },
    )
    async def _command_setup(self, command: List[str], **kwargs):
        await self._execute_setup(command, **kwargs)

    @setup_action(
        summary="Run a command on the VM and verify a post-condition (window exists / probe command succeeds).",
        args={
            "command": "argv list (preferred) or shell string when shell=True",
            "verification?": "{'window_exists': str} or {'command_success': List[str]} to assert post-state",
            "max_wait_time?": "max seconds to wait for verification (default 10)",
            "check_interval?": "seconds between verification polls (default 1.0)",
            "shell?": "interpret command via shell instead of argv (default False)",
        },
    )
    async def _execute_with_verification_setup(
            self,
            command: List[str],
            verification: Dict[str, Any] = None,
            max_wait_time: int = 10,
            check_interval: float = 1.0,
            shell: bool = False
    ):
        """Execute command with verification of results

        Args:
            command: Command to execute
            verification: Dict with verification criteria:
                - window_exists: Check if window with this name exists
                - command_success: Execute this command and check if it succeeds
            max_wait_time: Maximum time to wait for verification
            check_interval: Time between verification checks
            shell: Whether to use shell
        """
        if not command:
            raise Exception("Empty command to launch.")

        verification = verification or {}

        payload = json.dumps({
            "command": command,
            "shell": shell,
            "verification": verification,
            "max_wait_time": max_wait_time,
            "check_interval": check_interval
        })
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(self.http_server + "/setup" + "/execute_with_verification",
                                   headers=headers, data=payload, timeout=max_wait_time + 10)
            if response.status_code == 200:
                result = response.json()
                logger.info("Command executed and verified successfully: %s -> %s"
                            , " ".join(command) if isinstance(command, list) else command
                            , response.text
                            )
                return result
            else:
                logger.error("Failed to execute with verification. Status code: %s", response.text)
                raise Exception(f"Command verification failed: {response.text}")
        except requests.exceptions.RequestException as e:
            logger.error("An error occurred while trying to send the request: %s", e)
            traceback.print_exc()
            raise Exception(f"Request failed: {e}")

    @setup_action(
        summary="Pause the setup pipeline for a fixed duration (useful while a launched app finishes loading).",
        args={
            "seconds": "non-negative number of seconds to sleep on the host driver",
        },
    )
    async def _sleep_setup(self, seconds: float):
        await asyncio.sleep(seconds)

    async def _act_setup(self, action_seq: List[Union[Dict[str, Any], str]]):
        # TODO
        raise NotImplementedError()

    async def _replay_setup(self, trajectory: str):
        """
        Args:
            trajectory (str): path to the replay trajectory file
        """
        # TODO
        raise NotImplementedError()

    @setup_action(
        summary="Bring an existing X11 window to the foreground on the VM.",
        args={
            "window_name": "title (or class when by_class=True) of the window to activate",
            "strict?": "if True, match the title exactly; otherwise substring match (default False)",
            "by_class?": "match WM_CLASS instead of WM_NAME (default False)",
        },
    )
    async def _activate_window_setup(self, window_name: str, strict: bool = False, by_class: bool = False):
        if not window_name:
            raise Exception(f"Setup Open - Invalid path ({window_name}).")

        payload = json.dumps({"window_name": window_name, "strict": strict, "by_class": by_class})
        headers = {
            'Content-Type': 'application/json'
        }

        # send request to server to open file
        try:
            response = requests.post(self.http_server + "/setup" + "/activate_window", headers=headers, data=payload)
            if response.status_code == 200:
                logger.info("Command executed successfully: %s", response.text)
            else:
                logger.error(f"Failed to activate window {window_name}. Status code: %s", response.text)
        except requests.exceptions.RequestException as e:
            logger.error("An error occurred while trying to send the request: %s", e)

    @setup_action(
        summary="Close an X11 window on the VM by name/class.",
        args={
            "window_name": "title (or class when by_class=True) of the window to close",
            "strict?": "if True, match the title exactly; otherwise substring match (default False)",
            "by_class?": "match WM_CLASS instead of WM_NAME (default False)",
        },
    )
    async def _close_window_setup(self, window_name: str, strict: bool = False, by_class: bool = False):
        if not window_name:
            raise Exception(f"Setup Open - Invalid path ({window_name}).")

        payload = json.dumps({"window_name": window_name, "strict": strict, "by_class": by_class})
        headers = {
            'Content-Type': 'application/json'
        }

        # send request to server to open file
        try:
            response = requests.post(self.http_server + "/setup" + "/close_window", headers=headers, data=payload)
            if response.status_code == 200:
                logger.info("Command executed successfully: %s", response.text)
            else:
                logger.error(f"Failed to close window {window_name}. Status code: %s", response.text)
        except requests.exceptions.RequestException as e:
            logger.error("An error occurred while trying to send the request: %s", e)

    @setup_action(
        summary="Install and start tinyproxy on the VM so the browser can route through 127.0.0.1:18888.",
        args={
            "client_password?": "sudo password for the VM user; falls back to controller's client_password",
        },
    )
    async def _proxy_setup(self, client_password: str = ""):
        """Setup system-wide proxy configuration

        Args:
            client_password (str): Password for sudo operations, defaults to ""
        """
        retry = 0
        while retry < MAX_RETRIES:
            try:
                _ = requests.get(self.http_server + "/terminal")
                break
            except:
                time.sleep(5)
                retry += 1
                logger.info(f"retry: {retry}/{MAX_RETRIES}")

            if retry == MAX_RETRIES:
                return False

        # Use client_password from instance if not provided
        password = client_password if client_password else self.client_password

        # Configure proxy environment variables
        proxy_commands = [
            f"echo '{password}' | sudo -S bash -c \"apt-get update\"",
            f"echo '{password}' | sudo -S bash -c \"apt-get install -y tinyproxy\"",
            f"echo '{password}' | sudo -S bash -c \"echo 'Port 18888' > /tmp/tinyproxy.conf\"",
            f"echo '{password}' | sudo -S bash -c \"echo 'Allow 127.0.0.1' >> /tmp/tinyproxy.conf\"",
        ]

        # Execute all proxy configuration commands
        for cmd in proxy_commands:
            try:
                await self._execute_setup([cmd], shell=True)
            except Exception as e:
                logger.error(f"Failed to execute proxy setup command: {e}")
                raise

        await self._launch_setup(["tinyproxy -c /tmp/tinyproxy.conf -d"], shell=True)
        logger.info("Proxy setup completed successfully")

    # Chrome setup
    @setup_action(
        summary="Open URLs as new tabs in the running Chrome on the VM via the CDP debugging port.",
        args={
            "urls_to_open": "list of fully-qualified URL strings",
        },
    )
    async def _chrome_open_tabs_setup(self, urls_to_open: List[str]):  # Declare as async function
        host = self.vm_ip
        port = self.chromium_port
        remote_debugging_url = f"http://{host}:{port}"
        logger.info("Connect to Chrome @: %s", remote_debugging_url)
        logger.debug("PLAYWRIGHT ENV: %s", repr(os.environ))
        for attempt in range(15):
            if attempt > 0:
                await asyncio.sleep(5)  # Convert sync sleep to async
            browser = None
            async with async_playwright() as p:  # Use async context manager
                try:
                    # Async connection method with await
                    browser = await p.chromium.connect_over_cdp(remote_debugging_url)
                except Exception as e:
                    if attempt < 14:
                        logger.error(f"Attempt {attempt + 1}: Failed to connect, retrying. Error: {e}")
                        continue
                    else:
                        logger.error(f"Failed to connect after multiple attempts: {e}")
                        raise e
                if not browser:
                    return
                logger.info("Opening %s...", urls_to_open)
                context = None
                for i, url in enumerate(urls_to_open):
                    if i == 0:
                        # Ensure context exists (async operations may delay initialization)
                        context = browser.contexts[0] if browser.contexts else await browser.new_context()
                    # Create new page async with await
                    page = await context.new_page()
                    try:
                        # Navigate to URL async with await
                        await page.goto(url, timeout=60000)
                    except Exception as e:
                        logger.warning(f"Opening {url} exceeds time limit: {e}")
                    logger.info(f"Opened tab {i + 1}: {url}")
                    if i == 0:
                        # Close default tab async with await
                        default_page = context.pages[0]
                        await default_page.close()
                # Ensure no sync operations before returning
                return browser, context

    @setup_action(
        summary="Close any Chrome tabs whose URL matches one in the list (via the CDP debugging port).",
        args={
            "urls_to_close": "list of fully-qualified URL strings; matched with compare_urls",
        },
    )
    async def _chrome_close_tabs_setup(self, urls_to_close: List[str]):
        # Changed to async sleep
        await asyncio.sleep(5)  # Wait for Chrome to finish launching asynchronously
        host = self.vm_ip
        port = self.chromium_port
        remote_debugging_url = f"http://{host}:{port}"
        
        # Using async context manager
        async with async_playwright() as p:
            browser = None
            for attempt in range(15):
                try:
                    # Async connection with await
                    browser = await p.chromium.connect_over_cdp(remote_debugging_url)
                    break
                except Exception as e:
                    if attempt < 14:
                        logger.error(f"Attempt {attempt + 1}: Failed to connect, retrying. Error: {e}")
                        # Async sleep between attempts
                        await asyncio.sleep(5)
                    else:
                        logger.error(f"Failed to connect after multiple attempts: {e}")
                        raise e
            if not browser:
                return
            context = None
            for i, url in enumerate(urls_to_close):
                if i == 0:
                    # Get first context asynchronously
                    context = browser.contexts[0] if browser.contexts else await browser.new_context()
                # Process pages in async context
                for page in context.pages:
                    if compare_urls(page.url, url):
                        # Async page closing
                        await page.close()
                        logger.info(f"Closed tab {i + 1}: {url}")
                        break
            return browser, context

    # google drive setup
    @setup_action(
        summary="Run delete/mkdirs/upload operations against a Google Drive account via PyDrive.",
        args={
            "settings_file?": "path to the PyDrive settings.yml (default evaluation_examples/settings/googledrive/settings.yml)",
            "operation": "list[str], each one of {'delete','mkdirs','upload'}",
            "args": "list[dict] with one entry per operation; per-op keys: delete=>{query?, trash?}, mkdirs=>{path}, upload=>{url, path}",
        },
    )
    async def _googledrive_setup(self, **config):
        """ Clean google drive space (eliminate the impact of previous experiments to reset the environment)
        @args:
            config(Dict[str, Any]): contain keys
                settings_file(str): path to google drive settings file, which will be loaded by pydrive.auth.GoogleAuth()
                operation(List[str]): each operation is chosen from ['delete', 'upload']
                args(List[Dict[str, Any]]): parameters for each operation
            different args dict for different operations:
                for delete:
                    query(str): query pattern string to search files or folder in google drive to delete, please refer to
                        https://developers.google.com/drive/api/guides/search-files?hl=en about how to write query string.
                    trash(bool): whether to delete files permanently or move to trash. By default, trash=false, completely delete it.
                for mkdirs:
                    path(List[str]): the path in the google drive to create folder
                for upload:
                    path(str): remote url to download file
                    dest(List[str]): the path in the google drive to store the downloaded file
        """
        settings_file = config.get('settings_file', 'evaluation_examples/settings/googledrive/settings.yml')
        gauth = GoogleAuth(settings_file=settings_file)
        drive = GoogleDrive(gauth)

        def mkdir_in_googledrive(paths: List[str]):
            paths = [paths] if type(paths) != list else paths
            parent_id = 'root'
            for p in paths:
                q = f'"{parent_id}" in parents and title = "{p}" and mimeType = "application/vnd.google-apps.folder" and trashed = false'
                folder = drive.ListFile({'q': q}).GetList()
                if len(folder) == 0:  # not exists, create it
                    parents = {} if parent_id == 'root' else {'parents': [{'id': parent_id}]}
                    file = drive.CreateFile({'title': p, 'mimeType': 'application/vnd.google-apps.folder', **parents})
                    file.Upload()
                    parent_id = file['id']
                else:
                    parent_id = folder[0]['id']
            return parent_id

        for oid, operation in enumerate(config['operation']):
            if operation == 'delete':  # delete a specific file
                # query pattern string, by default, remove all files/folders not in the trash to the trash
                params = config['args'][oid]
                q = params.get('query', '')
                trash = params.get('trash', False)
                q_file = f"( {q} ) and mimeType != 'application/vnd.google-apps.folder'" if q.strip() else "mimeType != 'application/vnd.google-apps.folder'"
                filelist: GoogleDriveFileList = drive.ListFile({'q': q_file}).GetList()
                q_folder = f"( {q} ) and mimeType = 'application/vnd.google-apps.folder'" if q.strip() else "mimeType = 'application/vnd.google-apps.folder'"
                folderlist: GoogleDriveFileList = drive.ListFile({'q': q_folder}).GetList()
                for file in filelist:  # first delete file, then folder
                    file: GoogleDriveFile
                    if trash:
                        file.Trash()
                    else:
                        file.Delete()
                for folder in folderlist:
                    folder: GoogleDriveFile
                    # note that, if a folder is trashed/deleted, all files and folders in it will be trashed/deleted
                    if trash:
                        folder.Trash()
                    else:
                        folder.Delete()
            elif operation == 'mkdirs':
                params = config['args'][oid]
                mkdir_in_googledrive(params['path'])
            elif operation == 'upload':
                params = config['args'][oid]
                url = params['url']
                with tempfile.NamedTemporaryFile(mode='wb', delete=False) as tmpf:
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            tmpf.write(chunk)
                    tmpf.close()
                    paths = [params['path']] if params['path'] != list else params['path']
                    parent_id = mkdir_in_googledrive(paths[:-1])
                    parents = {} if parent_id == 'root' else {'parents': [{'id': parent_id}]}
                    file = drive.CreateFile({'title': paths[-1], **parents})
                    file.SetContentFile(tmpf.name)
                    file.Upload()
                return
            else:
                raise ValueError('[ERROR]: not implemented clean type!')

    @setup_action(
        summary="Drive a Playwright login flow against a supported platform (currently 'googledrive').",
        args={
            "platform": "name of the login flow; supported: 'googledrive'",
            "settings_file": "path to a JSON file with {'email','password'} for the platform",
        },
    )
    async def _login_setup(self, **config):
        """Login to a website using async Playwright API."""
        host = self.vm_ip
        port = self.chromium_port
        remote_debugging_url = f"http://{host}:{port}"

        # Use async context manager for playwright instance
        async with async_playwright() as p:
            browser = None
            # Async connection retry logic
            for attempt in range(15):
                try:
                    # Await browser connection
                    browser = await p.chromium.connect_over_cdp(remote_debugging_url)
                    break
                except Exception as e:
                    if attempt < 14:
                        logger.error(f"Attempt {attempt + 1}: Failed to connect, retrying. Error: {e}")
                        # Use async sleep between attempts
                        await asyncio.sleep(5)
                    else:
                        logger.error(f"Failed to connect after multiple attempts: {e}")
                        raise e

            if not browser:
                return None

            # Get first browser context
            context = browser.contexts[0]
            platform = config['platform']

            if platform == 'googledrive':
                url = 'https://drive.google.com/drive/my-drive'
                # Create new page async
                page = await context.new_page()
                
                try:
                    # Async page navigation with timeout
                    await page.goto(url, timeout=60000)
                except Exception:
                    logger.warning(f"Opening {url} exceeded time limit")
                logger.info(f"Opened new page: {url}")

                # Load credentials
                with open(config['settings_file']) as f:
                    settings = json.load(f)
                email, password = settings['email'], settings['password']

                try:
                    # Async wait for email input
                    await page.wait_for_selector('input[type="email"]', state="visible", timeout=3000)
                    await page.fill('input[type="email"]', email)
                    await page.click('#identifierNext > div > button')
                    
                    # Async wait for password input
                    await page.wait_for_selector('input[type="password"]', state="visible", timeout=5000)
                    await page.fill('input[type="password"]', password)
                    await page.click('#passwordNext > div > button')
                    
                    # Wait for final page load
                    await page.wait_for_load_state('load', timeout=5000)
                except TimeoutError:
                    logger.error("Timeout during Google Drive login sequence")
                    return None
            else:
                raise NotImplementedError(f"Platform {platform} not supported")

            return browser, context

    @setup_action(
        summary="Inject fake entries into Chrome's browsing-history SQLite DB on the VM.",
        args={
            "history": "list of {'url': str, 'title': str, 'visit_time_from_now_in_seconds': int} entries to insert",
        },
    )
    async def _update_browse_history_setup(self, **config):
        cache_path = os.path.join(self.cache_dir, "history_new.sqlite")
        db_url = "https://drive.usercontent.google.com/u/0/uc?id=1Lv74QkJYDWVX0RIgg0Co-DUcoYpVL0oX&export=download" # google drive
        if not os.path.exists(cache_path):
                max_retries = 3
                downloaded = False
                e = None
                for i in range(max_retries):
                    try:
                        response = requests.get(db_url, stream=True)
                        response.raise_for_status()

                        with open(cache_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        logger.info("File downloaded successfully")
                        downloaded = True
                        break

                    except requests.RequestException as e:
                        logger.error(
                            f"Failed to download {db_url} caused by {e}. Retrying... ({max_retries - i - 1} attempts left)")
                if not downloaded:
                    raise requests.RequestException(f"Failed to download {db_url}. No retries left. Error: {e}")
        else:
            logger.info("File already exists in cache directory")
        # copy a new history file in the tmp folder
        db_path = cache_path

        history = config['history']

        for history_item in history:
            url = history_item['url']
            title = history_item['title']
            visit_time = datetime.now() - timedelta(seconds=history_item['visit_time_from_now_in_seconds'])

            # Chrome use ms from 1601-01-01 as timestamp
            epoch_start = datetime(1601, 1, 1)
            chrome_timestamp = int((visit_time - epoch_start).total_seconds() * 1000000)

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute('''
                   INSERT INTO urls (url, title, visit_count, typed_count, last_visit_time, hidden)
                   VALUES (?, ?, ?, ?, ?, ?)
               ''', (url, title, 1, 0, chrome_timestamp, 0))

            url_id = cursor.lastrowid

            cursor.execute('''
                   INSERT INTO visits (url, visit_time, from_visit, transition, segment_id, visit_duration)
                   VALUES (?, ?, ?, ?, ?, ?)
               ''', (url_id, chrome_timestamp, 0, 805306368, 0, 0))

            conn.commit()
            conn.close()

        logger.info('Fake browsing history added successfully.')

        controller = PythonController(self.vm_ip, self.server_port)

        # get the path of the history file according to the platform
        os_type = controller.get_vm_platform()

        if os_type == 'Windows':
            chrome_history_path = controller.execute_python_command(
                """import os; print(os.path.join(os.getenv('USERPROFILE'), "AppData", "Local", "Google", "Chrome", "User Data", "Default", "History"))""")[
                'output'].strip()
        elif os_type == 'Darwin':
            chrome_history_path = controller.execute_python_command(
                """import os; print(os.path.join(os.getenv('HOME'), "Library", "Application Support", "Google", "Chrome", "Default", "History"))""")[
                'output'].strip()
        elif os_type == 'Linux':
            if "arm" in platform.machine():
                chrome_history_path = controller.execute_python_command(
                    "import os; print(os.path.join(os.getenv('HOME'), 'snap', 'chromium', 'common', 'chromium', 'Default', 'History'))")[
                    'output'].strip()
            else:
                chrome_history_path = controller.execute_python_command(
                    "import os; print(os.path.join(os.getenv('HOME'), '.config', 'google-chrome', 'Default', 'History'))")[
                    'output'].strip()
        else:
            raise Exception('Unsupported operating system')

        form = MultipartEncoder({
            "file_path": chrome_history_path,
            "file_data": (os.path.basename(chrome_history_path), open(db_path, "rb"))
        })
        headers = {"Content-Type": form.content_type}
        logger.debug(form.content_type)

        # send request to server to upload file
        try:
            logger.debug("REQUEST ADDRESS: %s", self.http_server + "/setup" + "/upload")
            response = requests.post(self.http_server + "/setup" + "/upload", headers=headers, data=form)
            if response.status_code == 200:
                logger.info("Command executed successfully: %s", response.text)
            else:
                logger.error("Failed to upload file. Status code: %s", response.text)
        except requests.exceptions.RequestException as e:
            logger.error("An error occurred while trying to send the request: %s", e)

        self._execute_setup(["sudo chown -R user:user /home/user/.config/google-chrome/Default/History"], shell=True)

    def _upload_local_file_to_vm(self, local_path: str, remote_path: str):
        """Upload a host-local file to ``remote_path`` on the VM via /setup/upload."""
        with open(local_path, "rb") as fh:
            form = MultipartEncoder({
                "file_path": remote_path,
                "file_data": (os.path.basename(remote_path), fh),
            })
            headers = {"Content-Type": form.content_type}
            try:
                response = requests.post(self.http_server + "/setup" + "/upload", headers=headers, data=form)
                if response.status_code == 200:
                    logger.info("Uploaded %s -> %s: %s", local_path, remote_path, response.text)
                else:
                    logger.error("Failed to upload %s. Status: %s", remote_path, response.text)
                    raise Exception(f"Upload failed: {response.text}")
            except requests.exceptions.RequestException as e:
                logger.error("Upload request error: %s", e)
                raise

    @setup_action(
        summary="Create a fresh .xlsx on the VM, optionally pre-filled with literal values.",
        args={
            "path": "absolute path on the VM, e.g. '/home/user/Desktop/budget.xlsx'",
            "data?": "2D list of cell values; row 0 is written to row 1 starting at A1",
            "sheet_name?": "title of the active sheet",
        },
    )
    async def _create_calc_file_setup(
            self,
            path: str,
            data: Optional[List[List[Any]]] = None,
            sheet_name: Optional[str] = None,
    ):
        """Create an .xlsx spreadsheet on the VM, optionally pre-filled with values.

        Args:
            path: absolute path on the VM, e.g. "/home/user/Desktop/budget.xlsx".
            data: 2D list of cell values. Row 0 is written to row 1 starting at A1.
            sheet_name: optional sheet title.
        """
        from openpyxl import Workbook

        if not path:
            raise Exception("Setup CreateCalc - empty path.")

        wb = Workbook()
        ws = wb.active
        if sheet_name:
            ws.title = sheet_name
        if data:
            for row in data:
                ws.append(list(row))

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            local_path = tmp.name
        try:
            wb.save(local_path)
            self._upload_local_file_to_vm(local_path, path)
        finally:
            if os.path.exists(local_path):
                os.unlink(local_path)

    @setup_action(
        summary="Create a fresh .docx on the VM, optionally pre-populated with paragraphs.",
        args={
            "path": "absolute path on the VM, e.g. '/home/user/Desktop/notes.docx'",
            "paragraphs?": "list of paragraph strings; each becomes one paragraph",
            "content?": "alternative single string; split on '\\n' into paragraphs (ignored when paragraphs is set)",
        },
    )
    async def _create_writer_file_setup(
            self,
            path: str,
            paragraphs: Optional[List[str]] = None,
            content: Optional[str] = None,
    ):
        """Create a .docx document on the VM with optional initial text.

        Args:
            path: absolute path on the VM, e.g. "/home/user/Desktop/notes.docx".
            paragraphs: list of paragraphs; each becomes its own paragraph.
            content: alternative single string; split on "\\n" into paragraphs.
        """
        from docx import Document

        if not path:
            raise Exception("Setup CreateWriter - empty path.")

        if paragraphs is None:
            paragraphs = content.split("\n") if content else []

        doc = Document()
        for para in paragraphs:
            doc.add_paragraph(para)

        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            local_path = tmp.name
        try:
            doc.save(local_path)
            self._upload_local_file_to_vm(local_path, path)
        finally:
            if os.path.exists(local_path):
                os.unlink(local_path)

    @setup_action(
        summary="Create a fresh .pptx on the VM, optionally pre-populated with slides.",
        args={
            "path": "absolute path on the VM, e.g. '/home/user/Desktop/deck.pptx'",
            "slides?": "list of {'title'?: str, 'content'?: str|List[str]}; missing/empty -> single blank slide",
        },
    )
    async def _create_impress_file_setup(
            self,
            path: str,
            slides: Optional[List[Dict[str, Any]]] = None,
    ):
        """Create a .pptx presentation on the VM with optional initial slides.

        Args:
            path: absolute path on the VM, e.g. "/home/user/Desktop/deck.pptx".
            slides: list of slide dicts, each like
                {"title": str, "content": str | List[str]}. Missing fields are skipped.
                If empty/None, a single blank slide is produced.
        """
        from pptx import Presentation

        if not path:
            raise Exception("Setup CreateImpress - empty path.")

        prs = Presentation()
        slides = slides or [{}]
        for spec in slides:
            layout_idx = 1 if spec.get("title") or spec.get("content") else 6
            layout = prs.slide_layouts[layout_idx] if layout_idx < len(prs.slide_layouts) else prs.slide_layouts[0]
            slide = prs.slides.add_slide(layout)

            title = spec.get("title")
            if title and slide.shapes.title is not None:
                slide.shapes.title.text = title

            body = spec.get("content")
            if body:
                placeholder = next(
                    (ph for ph in slide.placeholders if ph.placeholder_format.idx != 0),
                    None,
                )
                if placeholder is not None:
                    lines = body if isinstance(body, list) else [body]
                    tf = placeholder.text_frame
                    tf.text = lines[0]
                    for extra in lines[1:]:
                        tf.add_paragraph().text = extra

        with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as tmp:
            local_path = tmp.name
        try:
            prs.save(local_path)
            self._upload_local_file_to_vm(local_path, path)
        finally:
            if os.path.exists(local_path):
                os.unlink(local_path)

    @setup_action(
        summary="Create a blank raster image on the VM that GIMP can open (format inferred from path's extension).",
        args={
            "path": "absolute path on the VM, e.g. '/home/user/Desktop/canvas.png'",
            "width?": "image width in pixels (default 800)",
            "height?": "image height in pixels (default 600)",
            "color?": "fill color; PIL color name, '#rrggbb', or [r,g,b]/[r,g,b,a] (default 'white')",
            "mode?": "PIL mode, e.g. 'RGB' or 'RGBA' (default 'RGB')",
        },
    )
    async def _create_gimp_image_setup(
            self,
            path: str,
            width: int = 800,
            height: int = 600,
            color: Union[str, List[int]] = "white",
            mode: str = "RGB",
    ):
        """Create a blank raster image on the VM that GIMP can open.

        Args:
            path: absolute path on the VM, e.g. "/home/user/Desktop/canvas.png".
            width, height: image dimensions in pixels.
            color: fill color — name (e.g. "white"), "#rrggbb", or [r,g,b]/[r,g,b,a].
            mode: PIL mode, e.g. "RGB" or "RGBA". The output format is inferred
                from ``path``'s extension (.png/.jpg/.bmp/.tiff/...).
        """
        from PIL import Image

        if not path:
            raise Exception("Setup CreateGimpImage - empty path.")

        fill = tuple(color) if isinstance(color, list) else color
        img = Image.new(mode, (int(width), int(height)), fill)

        ext = os.path.splitext(path)[1] or ".png"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            local_path = tmp.name
        try:
            img.save(local_path)
            self._upload_local_file_to_vm(local_path, path)
        finally:
            if os.path.exists(local_path):
                os.unlink(local_path)
