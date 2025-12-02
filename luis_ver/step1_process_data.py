import csv
import os
import json
import statistics

csv_path = 'ImageAesthetics_ECCV2016/AllinAll.csv'
output_path = 'dataset_processed.json'

# Mapping from CSV column prefix to Spec Attribute
# Note: The CSV has Answer.Attribute1, Answer.Attribute2 ... Answer.Attribute10
attribute_map = {
    'Balancing Element': 'Answer.VisualBalance',
    'Rule of Thirds': 'Answer.RuleOfThirds',
    'Symmetry': 'Answer.Symmetry',
    'Repetition': 'Answer.Repetition',
    'Object Emphasis': 'Answer.ObjectEmphasis',
    'Light': 'Answer.choiceLight',
    'Color Harmony': 'Answer.ColorHarmony',
    'Vivid Color': 'Answer.StrongColor',
    'Depth of Field': 'Answer.DoF',
    'Motion Blur': 'Answer.MotionBlur',
    'Content': 'Answer.Content',
    'Aggregate Aesthetic Score': 'Answer.overallScore'
}

def normalize_attribute(value):
    if value == 'Positive':
        return 1.0
    elif value == 'Negative':
        return 0.0
    else:
        return 0.5

def normalize_score(value):
    try:
        score = float(value)
        # Map 1-5 to 0-1
        return (score - 1.0) / 4.0
    except ValueError:
        return 0.5 # Default if error

def get_filename_from_url(url):
    # Logic from step1_generateDataset.py
    # https://farm1.staticflickr.com/450/20131843366_5df791e881_b.jpg
    # farmTag = farm1
    # folderTag = 450
    # imgTag = 20131843366_5df791e881_b.jpg
    # imgName = farm1_450_20131843366_5df791e881_b.jpg
    
    parts = url.split('/')
    if len(parts) < 5:
        return None
    
    img_tag = parts[-1]
    folder_tag = parts[-2]
    farm_domain = parts[-3] # farm1.staticflickr.com
    farm_tag = farm_domain.split('.')[0]
    
    return f"{farm_tag}_{folder_tag}_{img_tag}"

image_data = {}

print(f"Reading {csv_path}...")
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        for i in range(1, 11): # 1 to 10
            url_key = f"Input.image_url{i}"
            if url_key not in row or not row[url_key]:
                continue
                
            url = row[url_key]
            filename = get_filename_from_url(url)
            if not filename:
                continue
            
            if filename not in image_data:
                image_data[filename] = {
                    'C': [], 'L': [], 'F': [], 'O': [], 'IAS': []
                }
            
            # Extract raw attributes
            raw_attrs = {}
            for attr_name, col_prefix in attribute_map.items():
                col_name = f"{col_prefix}{i}"
                val = row.get(col_name, '')
                if attr_name == 'Aggregate Aesthetic Score':
                    raw_attrs[attr_name] = normalize_score(val)
                else:
                    raw_attrs[attr_name] = normalize_attribute(val)
            
            # Calculate Targets per worker
            # C (Composition)
            c_score = statistics.mean([
                raw_attrs['Balancing Element'],
                raw_attrs['Rule of Thirds'],
                raw_attrs['Symmetry'],
                raw_attrs['Repetition'],
                raw_attrs['Object Emphasis']
            ])
            
            # L (Light/Color)
            l_score = statistics.mean([
                raw_attrs['Light'],
                raw_attrs['Color Harmony'],
                raw_attrs['Vivid Color']
            ])
            
            # F (Focus)
            # 0.5 * DoF + 0.5 * (1 - MotionBlur)
            f_score = 0.5 * raw_attrs['Depth of Field'] + 0.5 * (1.0 - raw_attrs['Motion Blur'])
            
            # O (Originality)
            o_score = raw_attrs['Content']
            
            # IAS (Global)
            ias_score = raw_attrs['Aggregate Aesthetic Score']
            
            image_data[filename]['C'].append(c_score)
            image_data[filename]['L'].append(l_score)
            image_data[filename]['F'].append(f_score)
            image_data[filename]['O'].append(o_score)
            image_data[filename]['IAS'].append(ias_score)

# Aggregate across workers
final_dataset = []
print("Aggregating scores...")
for filename, scores in image_data.items():
    final_dataset.append({
        'img_path': os.path.join('ImageAesthetics_ECCV2016/datasetImages_warp256', filename),
        'filename': filename,
        'C': statistics.mean(scores['C']),
        'L': statistics.mean(scores['L']),
        'F': statistics.mean(scores['F']),
        'O': statistics.mean(scores['O']),
        'IAS': statistics.mean(scores['IAS'])
    })

print(f"Processed {len(final_dataset)} images.")
with open(output_path, 'w') as f:
    json.dump(final_dataset, f, indent=2)
print(f"Saved to {output_path}")
