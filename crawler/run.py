from datetime import datetime
import argparse
import subprocess
import sys
import os
import logging

logger = logging.getLogger(__name__)

def show_project_version() -> None:
    """Function to show the version of the project."""
    try:
        subprocess.run(["scrapy", "settings", "--get", "VERSION"], check=True)
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        sys.exit(1)

def show_spiders() -> None:
    """Function to get the list of available spiders."""
    try:
        result = subprocess.run(["scrapy", "list"], check=True, stdout=subprocess.PIPE, text=True)
        spiders = result.stdout.strip().split('\n')
        if len(spiders) > 0:
            print("Available spiders:")
            for i, spider in enumerate(spiders):
                print(f"{i + 1} - {spider}")
        else:
            print("No spiders found.")
    except subprocess.CalledProcessError as e:
        print("Error getting spiders:", e)

def run(spider_name: str) -> None:
    """Function to run the scrapy with the specified spider."""
    try:
        subprocess.run(["scrapy", "crawl", spider_name], check=True, env=os.environ.copy())
    except Exception as e:
        raise e
    
def main():
    """Main function to run the scrapy with the specified spider."""
    logger.info(f"Starting crawler at {datetime.now().isoformat()}")
    parser = argparse.ArgumentParser(description='Run scrapy spiders')
    parser.add_argument('--version',
                        '-v',
                        action='store_true',
                        help='Show the version of the crawler')
    
    parser.add_argument('--spider_name',
                        '-s',
                        help='Name of the spider to run')
    
    parser.add_argument('--list_spiders',
                        '-ls',
                        action='store_true',
                        help='List all available spiders')

    args = parser.parse_args()
    
    if args.version:
        show_project_version()
        sys.exit(0)
    
    if args.list_spiders:
        show_spiders()
        sys.exit(0)
        
    if args.spider_name:
        run(args.spider_name)
        sys.exit(0)
    print(f"Ending crawler at {datetime.now().isoformat()}")
        
if __name__ == "__main__":
    main()