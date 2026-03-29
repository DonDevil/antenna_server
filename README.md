# AMC Antenna Optimization Server

Server-side project for natural-language driven antenna design orchestration with ANN prediction and CST command-package generation.

## Quick Start

1. Install dependencies:

Linux (using your env folder):

source env/bin/activate
pip install -r requirements.txt

2. Generate a seed dataset:

python3 scripts/generate_synthetic_dataset.py

You can also run the typo-compatible alias:

python3 scripts/generate_sysnthetic_dataset.py

3. Validate dataset and split valid/rejected rows:

python3 scripts/validate_dataset.py

4. Train ANN model:

python3 scripts/train_ann.py

5. Run server:

python3 main.py

## API Endpoints

- `GET /api/v1/health`
- `POST /api/v1/optimize`
- `POST /api/v1/client-feedback`

## Notes

- This server emits high-level CST command packages only.
- Raw VBA generation and CST execution belong to the client-side implementation.
