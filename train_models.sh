source dmc-venv/bin/activate

python3 print("Training models...")

python3 11-rf.py
python3 12-hgb.py

python3 print("Done training all models.")

deactivate