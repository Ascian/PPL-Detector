import os
import json

from predictor import PPLPredictor

import pandas as pd
import torch

def main():
    config_name = "Qwen3-0.6B"  
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 读取配置
    print("\nLoad Config...")
    config = json.load(open(os.path.join('./config', f'{config_name}.json'), 'r'))
    print(config)
    data_path = config['data_path']
    result_path = config['result_path']
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    print("\nLoad Predictor...")
    predictor = PPLPredictor(
        model_name=config['model_name'],
        prompt_max_seq_len=config['prompt_max_seq_len'],
        max_seq_len=config['max_seq_len'],
        ppl_limit=config['ppl_limit'],
        ppl_threshold=config['ppl_threshold'],
        ppl_scale=config['ppl_scale'],
        device=device,
    )

    if config['need_val']:
        print("\nLoad Validation Dataset...")
        val_data = pd.read_csv(f'{data_path}/val.csv')
        val_data = val_data[['prompt', 'text', 'label']].dropna()
        val_dataset = [{
            'prompt': row['prompt'],
            'text': row['text'],
            'label': row['label']
        } for _, row in val_data.iterrows()]
        print(f"\nValidation Dataset Size: {len(val_dataset)}")

        print("\nStart Evaluate...")
        result = predictor.evaluate(val_dataset)
        print(
        """
        Training Best Results:
        AUC: {auc}
        Accuracy: {accuracy}
        F1: {f1}
        TN: {tn}
        TP: {tp}
        FN: {fn}
        FP: {fp}
        Precision: {precision}
        Recall: {recall}
        Weighted Score: {weighted_score}
        """.format(
            auc=result['auc'],
            accuracy=result['accuracy'],
            f1=result['f1'],
            tn=result['tn'],
            tp=result['tp'],
            fn=result['fn'],
            fp=result['fp'],
            precision=result['precision'],
            recall=result['recall'],
            weighted_score=result['weighted_score'],
        ))

    if config['need_test']:
        print("\nLoad Test Dataset...")
        test_data = pd.read_csv(f'{data_path}/test1.csv')
        test_data = test_data[['prompt', 'text']].dropna()
        test_dataset = [{
            'prompt': row['prompt'],
            'text': row['text'],
            'label': 0
        } for _, row in test_data.iterrows()]
        print(f"\nTest Dataset Size: {len(test_dataset)}")
        
        print("\nStart Test...")
        start_time = pd.Timestamp.now()

        all_prompts, all_scores = predictor.predict(test_dataset)

        end_time = pd.Timestamp.now()
        print("\nTest Done!")
        save_results(all_prompts, all_scores, (end_time - start_time).total_seconds(), result_path)

def save_results(all_prompts, all_scores, processing_time, result_path):
    results_data = {
        'prompt': all_prompts,
        'text_prediction': all_scores,
    }
    results = {
        "predictions_data": results_data,
        "time": processing_time
    }
    
    os.makedirs(result_path, exist_ok=True)
    writer = pd.ExcelWriter(os.path.join(result_path, 'Ascian_Yu' + '.xlsx'), engine='openpyxl')
    
    prediction_frame = pd.DataFrame(
        data = results["predictions_data"]
    )
    prediction_frame = prediction_frame.dropna()
    
    time_frame = pd.DataFrame(
        data = {
            "Data Volume": [len(prediction_frame)],
            "Time": [results["time"]],
        }
    )
    
    prediction_frame.to_excel(writer, sheet_name="predictions", index=False)
    time_frame.to_excel(writer, sheet_name="time", index=False)
    writer.close()
    
    print(f"Results saved to {os.path.join(result_path, "Ascian_Yu" + '.xlsx')}")

if __name__ == "__main__":
    main()