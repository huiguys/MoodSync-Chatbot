# download_nltk_data.py
import nltk
import os
import sys
import shutil

# Set NLTK data path explicitly
nltk_data_path = os.path.join(os.path.expanduser('~'), 'AppData', 'Roaming', 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

# Ensure UTF-8 encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Clear existing wordnet data
wordnet_path = os.path.join(nltk_data_path, 'corpora', 'wordnet')
if os.path.exists(wordnet_path):
    print(f"Removing existing wordnet data at {wordnet_path}...")
    shutil.rmtree(wordnet_path)

print("Downloading NLTK data...")
try:
    nltk.download('punkt', quiet=False, force=False)  # Use existing if up-to-date
    nltk.download('wordnet', quiet=False, force=True)  # Force redownload
    print("NLTK data downloaded successfully.")
except Exception as e:
    print(f"Error downloading NLTK data: {e}")
    sys.exit(1)

# Verify files exist and are accessible
resources = [
    ('tokenizers/punkt', 'punkt'),
    ('corpora/wordnet', 'wordnet')
]
for resource_path, resource_name in resources:
    try:
        path = nltk.data.find(resource_path)
        print(f"Verified: {resource_name} is present at {path}.")
    except LookupError:
        print(f"Error: {resource_name} is missing.")
        sys.exit(1)

# Check for key wordnet files
wordnet_files = ['noun.exc', 'verb.exc', 'adj.exc', 'adv.exc']
for file in wordnet_files:
    file_path = os.path.join(nltk_data_path, 'corpora', 'wordnet', file)
    if os.path.exists(file_path):
        print(f"Confirmed: {file} exists in wordnet directory.")
    else:
        print(f"Error: {file} is missing from wordnet directory.")
        sys.exit(1)

print("All NLTK resources verified and complete.")