import sqlite3
import os

db = sqlite3.connect('/Users/prithak/Library/Developer/CoreSimulator/Devices/01C901AE-4E62-4D6A-BEB0-771E158C5CB0/data/Containers/Data/Application/BC7D0AC1-B6F9-4A9C-8FE6-84641E9828B5/Documents/auth.db')
cursor = db.cursor()

folder_path = '/Users/prithak/Desktop/Face/Images'

for name in os.listdir(folder_path):
    person_folder = os.path.join(folder_path, name)
    
    if os.path.isdir(person_folder):
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            with open(img_path, 'rb') as f:
                img_data = f.read()
            cursor.execute("INSERT INTO faces (name, image) VALUES (?, ?)", (name, img_data))

db.commit()
db.close()
