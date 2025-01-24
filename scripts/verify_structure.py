import os
from pathlib import Path

REQUIRED_DIRS = [
    'src/core',
    'src/adapters/phemex',
    'config/production',
    'tests/integration'
]

def validate_structure():
    missing = []
    for dir in REQUIRED_DIRS:
        if not Path(dir).exists():
            missing.append(dir)
    
    if missing:
        raise SystemExit(f'Critical directories missing: {missing}')
    
    print('Structure validation passed')

if __name__ == '__main__':
    validate_structure()
