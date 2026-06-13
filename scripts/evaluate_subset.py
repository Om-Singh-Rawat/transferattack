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


def resolve_adv_path(raw_path, adv_dir):
    """Resolve an adversarial image path from the CSV, handling WSL paths and
    directory renames (e.g. outputs/ -> outputs_baseline/).

    Tries in order:
      1. Path as-is
      2. WSL /mnt/X/... -> X:/...
      3. Tail of path (Attacker/Attack/file.png) under adv_dir
    """
    if pd.isna(raw_path):
        return None
    p = str(raw_path).strip()

    # 1. Try as-is
    if os.path.exists(p):
        return p

    # 2. Convert WSL -> Windows
    if p.startswith('/mnt/'):
        parts = p.split('/')
        drive = parts[2].upper()
        rest = '/'.join(parts[3:])
        win = f"{drive}:/{rest}"
        if os.path.exists(win):
            return win

    # 3. Try the last 3 path components (Attacker/Attack/filename) under adv_dir
    #    e.g. /mnt/c/.../outputs/ArcFace/PGD/adv_r0_xxx.png
    #    -> outputs_baseline/ArcFace/PGD/adv_r0_xxx.png
    norm = p.replace('\\', '/')
    parts = norm.split('/')
    for depth in (3, 2, 1):
        if len(parts) >= depth:
            tail = os.path.join(*parts[-depth:])
            candidate = os.path.join(adv_dir, tail)
            if os.path.exists(candidate):
                return candidate

    return None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-csv', default='docs/subset_input_pairs.csv')
    ap.add_argument('--dataset-root', default='../interns/dataset_extractedfaces')
    ap.add_argument('--adv-dir', default='outputs_baseline')
    ap.add_argument('--output-csv', default='results_baseline_recheck/subset_raw_similarities_long.csv')
    ap.add_argument('--attacks', default='PGD,MI_FGSM,TI_FGSM,SI_NI_FGSM,MI_ADMIX_DI_TI')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    if os.path.exists(args.output_csv):
        os.remove(args.output_csv)
    df = pd.read_csv(args.input_csv)
    
    attackers = ['ArcFace', 'Facenet512', 'GhostFaceNet', 'VGG-Face']
    # 5 victims matching professor's baseline (no separate Facenet — not in thresholds)
    victims_keras = ['Facenet512', 'ArcFace', 'GhostFaceNet', 'VGG-Face']
    attacks = [x.strip() for x in args.attacks.split(',') if x.strip()]
    
    results = []
    missing_count = 0
    found_count = 0
    
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
            clean_sim = get_cosine_sim(emb1, emb2)
            
            # Clean similarity — one row per attacker
            for attacker in attackers:
                results.append({
                    'row_id': row_id, 'attacker_model': attacker, 'img1': rec['img1'], 'img2': rec['img2'],
                    'dataset': rec['dataset'], 'attack_type': rec['attack_type'], 'source_csv': 'eval',
                    'victim_model': victim, 'attack_method': 'clean', 'variant': 'clean', 
                    'similarity': clean_sim, 'source_column': f'{victim}_clean'
                })
            
            # Adversarial similarities
            for attacker in attackers:
                # Skip self-transfer
                if attacker == victim:
                    continue
                
                csv_path = os.path.join(args.adv_dir, f"{attacker}_subset_adv_paths.csv")
                if not os.path.exists(csv_path): continue
                adv_df = pd.read_csv(csv_path)
                adv_row = adv_df[adv_df['row_id'] == row_id]
                if adv_row.empty: continue
                adv_row = adv_row.iloc[0]
                
                for attack in attacks:
                    col_name = f"{attack.lower()}_path"
                    raw_adv_path = adv_row.get(col_name)
                    adv_path = resolve_adv_path(raw_adv_path, args.adv_dir)
                    
                    if adv_path is None:
                        missing_count += 1
                        continue
                    
                    found_count += 1
                    adv_img = tf.expand_dims(load_and_preprocess(adv_path, input_size), 0)
                    adv_emb = compute_embedding(model, adv_img).numpy()[0]
                    
                    results.append({
                        'row_id': row_id, 'attacker_model': attacker, 'img1': rec['img1'], 'img2': rec['img2'],
                        'dataset': rec['dataset'], 'attack_type': rec['attack_type'], 'source_csv': 'eval',
                        'victim_model': victim, 'attack_method': attack, 'variant': 'vanilla', 
                        'similarity': get_cosine_sim(adv_emb, emb2), 'source_column': f'{victim}_{attack}_adv'
                    })

        write_results(results, args.output_csv)
        print(f"  Done. Found: {found_count}, Missing: {missing_count}")

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
        clean_sim = float(np.dot(emb1, emb2))
        
        # Clean rows
        for attacker in attackers:
            results.append({
                'row_id': row_id, 'attacker_model': attacker, 'img1': rec['img1'], 'img2': rec['img2'],
                'dataset': rec['dataset'], 'attack_type': rec['attack_type'], 'source_csv': 'eval',
                'victim_model': 'IR152', 'attack_method': 'clean', 'variant': 'clean', 
                'similarity': clean_sim, 'source_column': 'IR152_clean'
            })
        
        # Adversarial rows
        for attacker in attackers:
            csv_path = os.path.join(args.adv_dir, f"{attacker}_subset_adv_paths.csv")
            if not os.path.exists(csv_path): continue
            adv_df = pd.read_csv(csv_path)
            adv_row = adv_df[adv_df['row_id'] == row_id]
            if adv_row.empty: continue
            adv_row = adv_row.iloc[0]
            
            for attack in attacks:
                col_name = f"{attack.lower()}_path"
                raw_adv_path = adv_row.get(col_name)
                adv_path = resolve_adv_path(raw_adv_path, args.adv_dir)
                
                if adv_path is None:
                    missing_count += 1
                    continue
                
                found_count += 1
                adv_emb = get_ir_emb(adv_path)
                results.append({
                    'row_id': row_id, 'attacker_model': attacker, 'img1': rec['img1'], 'img2': rec['img2'],
                    'dataset': rec['dataset'], 'attack_type': rec['attack_type'], 'source_csv': 'eval',
                    'victim_model': 'IR152', 'attack_method': attack, 'variant': 'vanilla', 
                    'similarity': float(np.dot(adv_emb, emb2)), 'source_column': f'IR152_{attack}_adv'
                })

    write_results(results, args.output_csv)
    print(f"\nEvaluation Complete!")
    print(f"Total adversarial images found: {found_count}")
    print(f"Total adversarial images missing: {missing_count}")

if __name__ == '__main__':
    main()
