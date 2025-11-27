# ui/utils/mt5_discovery.py
"""
MetaTrader 5 installation discovery and management.

Detects MT5 installations on the system and manages connections.
"""

import os
import glob
import json
import winreg
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import configparser

logger = logging.getLogger(__name__)


@dataclass
class MT5Installation:
    """Represents a MetaTrader 5 installation."""
    name: str
    path: str
    terminal_exe: str
    data_path: str
    broker: str = ""
    server: str = ""
    login: int = 0
    is_portable: bool = False
    is_valid: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MT5Installation':
        return cls(**data)


class MT5Discovery:
    """
    Discovers and manages MetaTrader 5 installations.
    
    Scans common locations and registry for MT5 installations.
    """
    
    # Common installation paths
    COMMON_PATHS = [
        r"C:\Program Files\MetaTrader 5",
        r"C:\Program Files (x86)\MetaTrader 5",
        r"C:\Program Files\*MT5*",
        r"C:\Program Files\*MetaTrader*",
        r"C:\Program Files (x86)\*MT5*",
        r"C:\Program Files (x86)\*MetaTrader*",
        r"D:\Program Files\*MT5*",
        r"D:\Program Files\*MetaTrader*",
    ]
    
    # AppData paths for terminal data
    APPDATA_PATHS = [
        os.path.expandvars(r"%APPDATA%\MetaQuotes\Terminal"),
    ]
    
    def __init__(self, config_file: str = "config/mt5_installations.json"):
        self.config_file = Path(config_file)
        self.installations: List[MT5Installation] = []
        self._load_saved_installations()
    
    def discover_installations(self) -> List[MT5Installation]:
        """
        Discover all MT5 installations on the system.
        
        Returns:
            List of discovered MT5Installation objects
        """
        discovered = []
        
        # Search common paths
        discovered.extend(self._search_common_paths())
        
        # Search registry
        discovered.extend(self._search_registry())
        
        # Search AppData for terminals
        discovered.extend(self._search_appdata())
        
        # Remove duplicates based on terminal_exe path
        unique = {}
        for inst in discovered:
            key = inst.terminal_exe.lower()
            if key not in unique:
                unique[key] = inst
        
        self.installations = list(unique.values())
        
        # Try to get broker/server info for each
        for inst in self.installations:
            self._enrich_installation_info(inst)
        
        self._save_installations()
        
        logger.info(f"Discovered {len(self.installations)} MT5 installations")
        return self.installations
    
    def _search_common_paths(self) -> List[MT5Installation]:
        """Search common installation paths."""
        found = []
        
        for pattern in self.COMMON_PATHS:
            if '*' in pattern:
                # Use glob for wildcard patterns
                matches = glob.glob(pattern)
                for match in matches:
                    terminal = Path(match) / "terminal64.exe"
                    if terminal.exists():
                        found.append(self._create_installation(terminal))
            else:
                # Direct path
                terminal = Path(pattern) / "terminal64.exe"
                if terminal.exists():
                    found.append(self._create_installation(terminal))
                    
                # Also check for 32-bit
                terminal32 = Path(pattern) / "terminal.exe"
                if terminal32.exists():
                    found.append(self._create_installation(terminal32))
        
        return found
    
    def _search_registry(self) -> List[MT5Installation]:
        """Search Windows registry for MT5 installations."""
        found = []
        
        registry_paths = [
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
            (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
        ]
        
        for hkey, path in registry_paths:
            try:
                with winreg.OpenKey(hkey, path) as key:
                    i = 0
                    while True:
                        try:
                            subkey_name = winreg.EnumKey(key, i)
                            with winreg.OpenKey(key, subkey_name) as subkey:
                                try:
                                    display_name = winreg.QueryValueEx(subkey, "DisplayName")[0]
                                    if "MetaTrader" in display_name or "MT5" in display_name:
                                        try:
                                            install_location = winreg.QueryValueEx(subkey, "InstallLocation")[0]
                                            terminal = Path(install_location) / "terminal64.exe"
                                            if terminal.exists():
                                                inst = self._create_installation(terminal)
                                                inst.name = display_name
                                                found.append(inst)
                                        except WindowsError:
                                            pass
                                except WindowsError:
                                    pass
                            i += 1
                        except WindowsError:
                            break
            except WindowsError:
                continue
        
        return found
    
    def _search_appdata(self) -> List[MT5Installation]:
        """Search AppData for terminal data folders."""
        found = []
        
        for base_path in self.APPDATA_PATHS:
            base = Path(base_path)
            if base.exists():
                for terminal_folder in base.iterdir():
                    if terminal_folder.is_dir():
                        # Check for origin.txt which contains the path
                        origin_file = terminal_folder / "origin.txt"
                        if origin_file.exists():
                            try:
                                with open(origin_file, 'r') as f:
                                    origin_path = f.read().strip()
                                    terminal = Path(origin_path) / "terminal64.exe"
                                    if terminal.exists():
                                        inst = self._create_installation(terminal)
                                        inst.data_path = str(terminal_folder)
                                        found.append(inst)
                            except Exception as e:
                                logger.debug(f"Error reading origin.txt: {e}")
        
        return found
    
    def _create_installation(self, terminal_path: Path) -> MT5Installation:
        """Create an MT5Installation object from terminal path."""
        parent = terminal_path.parent
        name = parent.name
        
        # Try to find data path
        data_path = ""
        appdata_base = Path(os.path.expandvars(r"%APPDATA%\MetaQuotes\Terminal"))
        if appdata_base.exists():
            for folder in appdata_base.iterdir():
                origin_file = folder / "origin.txt"
                if origin_file.exists():
                    try:
                        with open(origin_file, 'r') as f:
                            if f.read().strip().lower() == str(parent).lower():
                                data_path = str(folder)
                                break
                    except Exception:
                        pass
        
        return MT5Installation(
            name=name,
            path=str(parent),
            terminal_exe=str(terminal_path),
            data_path=data_path,
            is_portable=self._check_portable(parent)
        )
    
    def _check_portable(self, install_path: Path) -> bool:
        """Check if the installation is portable mode."""
        # Portable installations have a MQL5 folder in the installation directory
        mql5_path = install_path / "MQL5"
        return mql5_path.exists() and (mql5_path / "Experts").exists()
    
    def _enrich_installation_info(self, installation: MT5Installation):
        """Try to get additional info like broker and server from config files."""
        data_path = Path(installation.data_path) if installation.data_path else None
        
        if data_path and data_path.exists():
            # Try to read from config files
            config_dir = data_path / "config"
            
            # Check for common.ini
            common_ini = config_dir / "common.ini"
            if common_ini.exists():
                try:
                    config = configparser.ConfigParser()
                    config.read(common_ini)
                    if 'Common' in config:
                        installation.login = config.getint('Common', 'Login', fallback=0)
                except Exception as e:
                    logger.debug(f"Error reading common.ini: {e}")
            
            # Try to get server from servers folder
            servers_dir = config_dir / "servers"
            if servers_dir.exists():
                for server_folder in servers_dir.iterdir():
                    if server_folder.is_dir():
                        installation.server = server_folder.name
                        # Try to get broker name
                        broker_file = server_folder / "brokerinfo.dat"
                        if broker_file.exists():
                            try:
                                # Binary file, try to extract broker name
                                with open(broker_file, 'rb') as f:
                                    data = f.read()
                                    # Look for broker name pattern
                                    # This is a simplified extraction
                                    pass
                            except Exception:
                                pass
                        break
    
    def add_installation(self, path: str, name: str = "", 
                        server: str = "", login: int = 0,
                        password: str = "") -> Optional[MT5Installation]:
        """
        Manually add an MT5 installation.
        
        Args:
            path: Path to MT5 installation directory
            name: Display name for the installation
            server: Trading server
            login: Account login
            password: Account password (not stored)
        
        Returns:
            MT5Installation if valid, None otherwise
        """
        terminal = Path(path) / "terminal64.exe"
        if not terminal.exists():
            terminal = Path(path) / "terminal.exe"
        
        if not terminal.exists():
            logger.error(f"No terminal executable found at {path}")
            return None
        
        installation = self._create_installation(terminal)
        if name:
            installation.name = name
        installation.server = server
        installation.login = login
        
        # Check if already exists
        existing = next((i for i in self.installations 
                        if i.terminal_exe.lower() == installation.terminal_exe.lower()), None)
        if existing:
            # Update existing
            existing.name = installation.name
            existing.server = server
            existing.login = login
        else:
            self.installations.append(installation)
        
        self._save_installations()
        return installation
    
    def remove_installation(self, terminal_exe: str) -> bool:
        """Remove an installation from the list."""
        original_count = len(self.installations)
        self.installations = [i for i in self.installations 
                             if i.terminal_exe.lower() != terminal_exe.lower()]
        
        if len(self.installations) < original_count:
            self._save_installations()
            return True
        return False
    
    def get_installation(self, terminal_exe: str) -> Optional[MT5Installation]:
        """Get installation by terminal path."""
        return next((i for i in self.installations 
                    if i.terminal_exe.lower() == terminal_exe.lower()), None)
    
    def _save_installations(self):
        """Save installations to config file."""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = [inst.to_dict() for inst in self.installations]
            
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved {len(self.installations)} installations to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving installations: {e}")
    
    def _load_saved_installations(self):
        """Load saved installations from config file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                
                self.installations = [MT5Installation.from_dict(d) for d in data]
                
                # Validate installations still exist
                for inst in self.installations:
                    inst.is_valid = Path(inst.terminal_exe).exists()
                
                logger.debug(f"Loaded {len(self.installations)} installations from {self.config_file}")
            except Exception as e:
                logger.error(f"Error loading installations: {e}")
                self.installations = []
    
    def get_valid_installations(self) -> List[MT5Installation]:
        """Get only valid (existing) installations."""
        return [i for i in self.installations if i.is_valid]
    
    def verify_installation(self, installation: MT5Installation) -> bool:
        """Verify an installation is valid and accessible."""
        if not Path(installation.terminal_exe).exists():
            installation.is_valid = False
            return False
        
        installation.is_valid = True
        return True
