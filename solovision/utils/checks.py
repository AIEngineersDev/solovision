import pkg_resources
import logging
from pathlib import Path
import sys
import subprocess

logger = logging.getLogger(__name__)
REQUIREMENTS = Path('requirements.txt')

import subprocess
import sys
import pkg_resources
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class RequirementsChecker:
    def check_requirements(self):
        requirements = pkg_resources.parse_requirements(REQUIREMENTS.open())
        self.check_packages(requirements)

    def check_packages(self, requirements, cmds=''):
        """Test that each required package is available."""
        missing_packages = []
        for r in requirements:
            package_name = str(r).split()[0]  # Get the package name
            if package_name == "ultralytics":
                self.solo_ultralytics()
                continue  # Skip further checks for ultralytics
            try:
                pkg_resources.require(str(r))
            except Exception as e:
                logger.error(f'{e}')
                missing_packages.append(str(r))
        
        if missing_packages:
            self.install_packages(missing_packages, cmds)

    def install_packages(self, packages, cmds=''):
        try:
            logger.warning(f'\nMissing packages: {", ".join(packages)}\nAttempting installation...')
            for package in packages:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', package] + cmds.split())
            logger.info('All the missing packages were installed successfully')
        except subprocess.CalledProcessError as e:
            logger.error(f'Failed to install packages: {e}')
            sys.exit(1)

    def solo_ultralytics(self, package_name='ultralytics'):
        source= "https://github.com/AIEngineersDev/solo-ultralytics"
        github_package = f"{package_name}@git+{source}.git"

        try:
            # Check if the package is installed
            dist = pkg_resources.get_distribution(package_name)
            installed_version = dist.version

            # Check the installation source
            for line in dist._get_metadata("METADATA"):
                if source in line:
                    return     
            else:
                logger.info(f"{package_name} version: {installed_version}")
                logger.warning(f"'Unsupported {package_name}' package is installed.")   
                
            # If the package is not installed from the solo-ultralytics repo, uninstall it
            logger.info(f"Removing existing '{package_name}' package...")
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", package_name])
        
        except pkg_resources.DistributionNotFound:
            logger.info(f"'{package_name}' is not installed.")

        # Install the package from the GitHub repository
        logger.info(f"Installing '{package_name}' from the solo GitHub repository...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", f"{github_package}"])
            logger.info(f"'{package_name}' has been successfully installed from GitHub.")

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install '{package_name}' from GitHub: {e}")
            sys.exit(1)