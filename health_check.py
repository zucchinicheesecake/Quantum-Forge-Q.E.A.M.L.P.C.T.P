import psutil
import requests
import sys
import sqlite3
from datetime import datetime

class SystemHealthCheck:
    def __init__(self):
        self.results = {}
        
    def check_cpu(self):
        cpu_percent = psutil.cpu_percent(interval=1)
        self.results['cpu'] = {
            'status': 'OK' if cpu_percent < 80 else 'WARNING',
            'value': cpu_percent
        }

    def check_memory(self):
        mem = psutil.virtual_memory()
        self.results['memory'] = {
            'status': 'OK' if mem.percent < 90 else 'WARNING',
            'value': mem.percent
        }

    def check_disk(self):
        disk = psutil.disk_usage('/')
        self.results['disk'] = {
            'status': 'OK' if disk.percent < 90 else 'WARNING',
            'value': disk.percent
        }

    def check_network(self):
        try:
            response = requests.get('http://google.com', timeout=5)
            status = 'OK' if response.status_code == 200 else 'ERROR'
        except:
            status = 'ERROR'
        self.results['network'] = {'status': status}

    def check_database(self):
        try:
            conn = sqlite3.connect('app.db')
            conn.cursor()
            conn.close()
            status = 'OK'
        except:
            status = 'ERROR'
        self.results['database'] = {'status': status}

    def run_all_checks(self):
        self.check_cpu()
        self.check_memory()
        self.check_disk()
        self.check_network()
        self.check_database()
        return self.results

    def print_report(self):
        print("\nSystem Health Check Report")
        print(f"Time: {datetime.now()}\n")
        for check, result in self.results.items():
            print(f"{check.upper()}: {result['status']}")
            if 'value' in result:
                print(f"Value: {result['value']}%")
            print("-" * 30)

if __name__ == "__main__":
    checker = SystemHealthCheck()
    checker.run_all_checks()
    checker.print_report()