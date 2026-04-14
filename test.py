import time
import torch
from ultralytics import YOLO


def validate_custom_model(model_path, data_path, imgsz=1024, batch=8):

    # 1. 加载模型 (自动检测是否可以使用 GPU)
    device = 0 if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path, task='obb')

    print(f"\n" + "=" * 40)
    print(f"开始验证模型: {model_path}")
    print(f"运行设备: {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")
    print(f"输入尺寸: {imgsz} | Batch Size: {batch}")
    print("=" * 40)

    results = model.val(
        data=data_path,
        imgsz=imgsz,
        batch=batch,
        conf=0.001,
        iou=0.05,
        device=device,
        half=True,  
        workers=4,  
        rect=True,  
        plots=False,  
        save_json=False,
    )

    precision = results.results_dict.get('metrics/precision(B)', 0)
    recall = results.results_dict.get('metrics/recall(B)', 0)
    map50 = results.results_dict.get('metrics/mAP50(B)', 0)
    map50_95 = results.results_dict.get('metrics/mAP50-95(B)', 0)

    speed = results.speed
    inference_time = speed['inference']
    fps = 1000 / inference_time if inference_time > 0 else 0

    print("\n" + ">>> 验证指标结果 <<<")
    print(f"{'Metric':<15} | {'Value':<10}")
    print("-" * 30)
    print(f"{'Precision':<15} | {precision:.4f}")
    print(f"{'Recall':<15} | {recall:.4f}")
    print(f"{'mAP@.5':<15} | {map50:.4f}")
    print(f"{'mAP@.5:.95':<15} | {map50_95:.4f}")
    print("-" * 30)
    print(f"推理延迟 (ms/img): {inference_time:.2f}")
    print(f"每秒帧数 (FPS):    {fps:.2f}")
    print("=" * 40 + "\n")

    return results


if __name__ == "__main__":
    # --- 请根据你的实际路径修改 ---
    MODEL_WEIGHTS = r"C:\Users\29383\Desktop\ultralytics-main\my\sarm-2\weights\best.pt"
    DATA_CONFIG = r'C:\Users\29383\Desktop\ultralytics-main\my\data.yaml'

    validate_custom_model(
        MODEL_WEIGHTS,
        DATA_CONFIG,
        imgsz=1024,
        batch=8  # 减小 batch 可以缓解内存压力导致的系统卡顿
    )
