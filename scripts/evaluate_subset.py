import os, sys
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import torch
import torchvision.transforms as transforms

sys.path.append(os.path.abspath('core'))
sys.path.append(os.path.abspath('../interns'))
from transfer_attack_core import build_attacker, compute_embedding, load_and_preprocess, resolve_image_path
from ir152 import IR_152

def get_cosine_sim(emb1, emb2):
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

def write_results(results, out_csv):
    if not results: return
    df = pd.DataFrame(results)
    if not os.path.exists(out_csv):
        df.to_csv(out_csv, index=False)
    else:
        df.to_csv(out_csv, mode='a', header=False, index=False)
    results.clear()

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-csv', default='docs/subset_input_pairs.csv')
    ap.add_argument('--dataset-root', default='../interns/dataset_extractedfaces')
    ap.add_argument('--adv-dir', default='outputs')
    ap.add_argument('--output-csv', default='results_baseline_check/subset_raw_similarities_long.csv')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    if os.path.exists(args.output_csv):
        os.remove(args.output_csv)
    df = pd.read_csv(args.input_csv)
    
    attackers = ['ArcFace', 'Facenet512', 'GhostFaceNet', 'VGG-Face']
    victims_keras = ['Facenet', 'Facenet512', 'ArcFace', 'GhostFaceNet', 'VGG-Face']
    attacks = ['PGD', 'MI_FGSM', 'TI_FGSM', 'SI_NI_FGSM', 'MI_ADMIX_DI_TI']
    
    results = []
    
    # 1. Evaluate Keras Victims
    for victim in victims_keras:
        print(f"Evaluating victim: {victim}")
        model = build_attacker(victim)
        if victim in ['ArcFace', 'GhostFaceNet']:
            input_size = (112, 112)
        elif 'Facenet' in victim:
            input_size = (160, 160)
        else:
            input_size = (224, 224)
        
        for _, rec in df.iterrows():
            row_id = rec.get('row_id', _)
            img1_path = resolve_image_path(rec['img1'], args.dataset_root)
            img2_path = resolve_image_path(rec['img2'], args.dataset_root)
            
            img1 = tf.expand_dims(load_and_preprocess(img1_path, input_size), 0)
            img2 = tf.expand_dims(load_and_preprocess(img2_path, input_size), 0)
            
            emb1 = compute_embedding(model, img1).numpy()[0]
            emb2 = compute_embedding(model, img2).numpy()[0]
            
            # Clean similarity
            for attacker in attackers:
                if attacker == 'Facenet512' and victim == 'Facenet': continue
                results.append({
                    'row_id': row_id, 'attacker_model': attacker, 'img1': rec['img1'], 'img2': rec['img2'],
                    'dataset': rec['dataset'], 'attack_type': rec['attack_type'], 'victim_model': victim,
                    'attack_method': 'clean', 'variant': 'clean', 'similarity': get_cosine_sim(emb1, emb2)
                })
            
            # Adv similarities
            for attacker in attackers:
                # Rule: skip FaceNet512 -> FaceNet
                if attacker == 'Facenet512' and victim == 'Facenet': continue
                
                csv_path = os.path.join(args.adv_dir, f"{attacker}_subset_adv_paths.csv")
                if not os.path.exists(csv_path): continue
                adv_df = pd.read_csv(csv_path)
                adv_row = adv_df[adv_df['row_id'] == row_id]
                if adv_row.empty: continue
                adv_row = adv_row.iloc[0]
                
                for attack in attacks:
                    adv_path = adv_row.get(f"{attack.lower()}_path")
                    if pd.isna(adv_path) or not os.path.exists(str(adv_path)): continue
                    
                    adv_img = tf.expand_dims(load_and_preprocess(str(adv_path), input_size), 0)
                    adv_emb = compute_embedding(model, adv_img).numpy()[0]
                    
                    results.append({
                        'row_id': row_id, 'attacker_model': attacker, 'img1': rec['img1'], 'img2': rec['img2'],
                        'dataset': rec['dataset'], 'attack_type': rec['attack_type'], 'victim_model': victim,
                        'attack_method': attack, 'variant': 'vanilla', 'similarity': get_cosine_sim(adv_emb, emb2)
                    })

        write_results(results, args.output_csv)

    # 2. Evaluate PyTorch IR152 Victim
    print("Evaluating victim: IR152")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ir152 = IR_152((112, 112))
    state_dict = torch.load('../interns/ir152.pth', map_location='cpu')
    ir152.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    ir152.to(device)
    ir152.eval()
    
    tfm = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    
    def get_ir_emb(path):
        img = Image.open(path).convert('RGB')
        tensor = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = ir152(tensor).cpu().numpy()[0]
        return emb / np.linalg.norm(emb)
        
    for _, rec in df.iterrows():
        row_id = rec.get('row_id', _)
        img1_path = resolve_image_path(rec['img1'], args.dataset_root)
        img2_path = resolve_image_path(rec['img2'], args.dataset_root)
        
        emb1 = get_ir_emb(img1_path)
        emb2 = get_ir_emb(img2_path)
        
        for attacker in attackers:
            results.append({
                'row_id': row_id, 'attacker_model': attacker, 'img1': rec['img1'], 'img2': rec['img2'],
                'dataset': rec['dataset'], 'attack_type': rec['attack_type'], 'victim_model': 'IR152',
                'attack_method': 'clean', 'variant': 'clean', 'similarity': float(np.dot(emb1, emb2))
            })
        
        for attacker in attackers:
            csv_path = os.path.join(args.adv_dir, f"{attacker}_subset_adv_paths.csv")
            if not os.path.exists(csv_path): continue
            adv_df = pd.read_csv(csv_path)
            adv_row = adv_df[adv_df['row_id'] == row_id]
            if adv_row.empty: continue
            adv_row = adv_row.iloc[0]
            
            for attack in attacks:
                adv_path = adv_row.get(f"{attack.lower()}_path")
                if pd.isna(adv_path) or not os.path.exists(str(adv_path)): continue
                
                adv_emb = get_ir_emb(str(adv_path))
                results.append({
                    'row_id': row_id, 'attacker_model': attacker, 'img1': rec['img1'], 'img2': rec['img2'],
                    'dataset': rec['dataset'], 'attack_type': rec['attack_type'], 'victim_model': 'IR152',
                    'attack_method': attack, 'variant': 'vanilla', 'similarity': float(np.dot(adv_emb, emb2))
                })

    write_results(results, args.output_csv)
    print("Evaluation Complete!")

if __name__ == '__main__':
    main()
