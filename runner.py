import subprocess
import time
# Specify the path to the Python script you want to run
script_path = 'C:/Users/ariel/OneDrive/שולחן העבודה/fifa 23/Bart_trainer.py'

# Run the script using subprocess
for x in range(20):
    subprocess.run(['python', script_path])
    time.sleep(10)
