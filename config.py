import json
import os

DEFAULT_CONFIG = {
    'image_processing': {
        'blur_kernel_width': 3,
        'blur_kernel_height': 3,
        'threshold_block_size': 15,
        'threshold_constant': 8,
        'contour_approx_epsilon': 0.02,
        'sharpen_kernel': 0.5
    },
    'ocr': {
        'language': 'eng',
        'config': '--oem 3 --psm 3 -c textord_heavy_nr=1 -c textord_min_linesize=2.5',
        'min_confidence': 40
    },
    'export': {
        'default_formats': ['txt', 'pdf'],
        'output_directory': 'exports'
    }
}

class Config:
    def __init__(self, config_file='scanner_config.json'):
        self.config_file = config_file
        self.settings = self.load_config()

    def load_config(self):
        """Load configuration from file or create default"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return DEFAULT_CONFIG.copy()
        return DEFAULT_CONFIG.copy()

    def save_config(self):
        """Save current configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(self.settings, f, indent=4)

    def update_settings(self, section, **kwargs):
        """Update specific settings in a section"""
        if section not in self.settings:
            self.settings[section] = {}
        self.settings[section].update(kwargs)
        self.save_config()

    def get_setting(self, section, key, default=None):
        """Get a specific setting value"""
        return self.settings.get(section, {}).get(key, default)