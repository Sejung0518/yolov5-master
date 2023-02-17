import os

test_data = "data/lpr_data/test"
detect_data = "runs/detect/A0A1-test/images"
test_files = os.listdir(test_data)
detect_files = os.listdir(detect_data)
for file in test_files:
    file_name = os.path.splitext(file)[0] + "_0.jpg"
    if file_name in detect_files:
        continue
    else:
        print(file + " not detected")