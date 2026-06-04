import base64
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

try:
    from supabase import create_client
except ImportError as exc:
    raise SystemExit(
        "Missing supabase package. Install it with `pip install supabase` or add it to requirements.txt."
    ) from exc

SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_SERVICE_ROLE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY') or os.environ.get('UPABASE_SERVICE_ROLE_KEY')
MIGRATION_USER_ID = os.environ.get('MIGRATION_USER_ID')

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise SystemExit('SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in environment.')

if not MIGRATION_USER_ID:
    raise SystemExit('Set MIGRATION_USER_ID to the user UUID that should own migrated rows.')

try:
    payload_segment = SUPABASE_SERVICE_ROLE_KEY.split('.')[-2]
    payload_bytes = base64.urlsafe_b64decode(payload_segment + '==='[: (-len(payload_segment) % 4)])
    payload = json.loads(payload_bytes)
    key_role = payload.get('role')
except Exception:
    key_role = None

if key_role != 'service_role':
    raise SystemExit(
        'The configured SUPABASE_SERVICE_ROLE_KEY is not a Supabase service role key. '
        'Use the actual service role key from your Supabase project settings. '
        'Current key role: %s' % repr(key_role)
    )

ROOT = Path(__file__).resolve().parent.parent.parent
HISTORY_FILE = ROOT / 'data' / 'prediction_history.json'
if not HISTORY_FILE.exists():
    raise SystemExit(f'Missing history file: {HISTORY_FILE}')

with HISTORY_FILE.open('r', encoding='utf-8') as f:
    history = json.load(f)

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

if not isinstance(history, list):
    raise SystemExit('Expected JSON array in prediction_history.json')

records = []
for item in history:
    if not isinstance(item, dict):
        continue
    record = {
        'user_id': MIGRATION_USER_ID,
        'request': item.get('request', {}),
        'predicted_unit_price': item.get('predicted_unit_price', 0.0),
        'model': item.get('model', 'legacy-json'),
        'expert_suggestions': item.get('expertResult') if item.get('expertResult') else None,
        'created_at': item.get('timestamp'),
    }
    records.append(record)

print(f"Migrating {len(records)} rows to Supabase...")

chunk_size = 50
success_count = 0
errors = []
for idx in range(0, len(records), chunk_size):
    chunk = records[idx : idx + chunk_size]
    response = supabase.table('predictions').insert(chunk).execute()
    data = None
    error = None
    if isinstance(response, dict):
        data = response.get('data')
        error = response.get('error')
    else:
        data = getattr(response, 'data', None)
        error = getattr(response, 'error', None)

    if error:
        errors.append(error)
        print('Chunk error:', error)
    else:
        success_count += len(chunk)

print(f"Migration complete: {success_count} rows inserted.")
if errors:
    print('Errors encountered:')
    for err in errors:
        print(err)
