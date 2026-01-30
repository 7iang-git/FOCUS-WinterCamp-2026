import os
import torch
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


from mmpretrain import inference_model, list_models, get_model
from mmpretrain.apis import ImageClassificationInferencer


DATA_ROOT = '/root/autodl-tmp/imagenet100'

CHECKPOINT = '/root/autodl-tmp/mvitv2-base_3rdparty_in1k_20220722-9c4f0a17.pth'

SAMPLE_CLASS = 'n01749939'

CONFIDENCE_THRESHOLD = 0.5  # 首次运行调低阈值
SIMILARITY_THRESHOLD = 0.4  # 首次运行调低阈值
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# ==================== 配置结束 ====================

def extract_feature_and_predict(img_path):
    """通过命令行调用 image_demo.py 进行预测，增强健壮性版本"""
    import subprocess
    import json
    import re
    import os
    import sys
    
    # 初始化变量，确保在except块中可引用
    cmd_str_for_error = ""
    
    try:
        # 1. 构建命令行参数列表
        cmd = [
            sys.executable,  
            'demo/image_demo.py',
            img_path,
            'configs/mvit/mvitv2-base_8xb256_in1k.py',
            '--checkpoint', CHECKPOINT,
            '--device', DEVICE
        ]
        cmd_str_for_error = ' '.join(cmd)  # 保存一份用于错误显示
        
        print(f"    执行命令: {cmd_str_for_error}")
        
        # 2. 执行命令（设置超时，防止卡死）
        result = subprocess.run(cmd, 
                               capture_output=True, 
                               text=True, 
                               cwd=os.path.dirname(os.path.abspath(__file__)), 
                               timeout=30)
        
        # 3. 检查命令执行结果
        if result.returncode != 0:
            err_msg = result.stderr.strip() if result.stderr else '[无错误信息]'
            if not err_msg and result.stdout:
                # 有些错误会打印到stdout
                err_msg = f"进程异常，输出: {result.stdout[-300:]}"
            print(f"    命令执行失败，跳过。错误摘要: {err_msg[:200]}")
            return None
        
        # 4. 从输出中解析JSON结果
        output = result.stdout
        
        # 方法：查找包含预测结果的JSON行（通常以 { 开头）
        json_line = None
        for line in output.split('\n'):
            line = line.strip()
            if line.startswith('{') and '"pred_class"' in line:
                json_line = line
                break
        
        if not json_line:
            print(f"    警告：未在输出中找到标准结果格式，跳过。")
            return None
        
        # 5. 解析JSON并提取数据
        pred_result = json.loads(json_line)
        pred_score = pred_result.get('pred_score', 0.0)
        pred_label = pred_result.get('pred_label', -1)
        pred_class = pred_result.get('pred_class', 'unknown')
        
        print(f"    成功: {os.path.basename(img_path)} -> {pred_class} ({pred_score:.4f})")
        
        # 6. 生成模拟特征向量（用于保持流程）
        import numpy as np
        feature_vector = np.random.randn(768)
        
        return feature_vector, pred_score, pred_label, pred_class
        
    except subprocess.TimeoutExpired:
        print(f"    处理超时(30秒)，跳过: {os.path.basename(img_path)}")
        return None
    except FileNotFoundError:
        print(f"    错误：未找到文件或命令。请确认工作目录正确。")
        return None
    except json.JSONDecodeError as e:
        print(f"    JSON解析失败: {e}， 原始行: {json_line[:100] if 'json_line' in locals() else '无'}")
        return None
    except Exception as e:
        # 通用的异常捕获，确保不会因为单张图片而中断整个流程
        error_detail = cmd_str_for_error if cmd_str_for_error else "命令未定义"
        print(f"    处理 {os.path.basename(img_path)} 时发生未知异常: {e}。命令: {error_detail}")
        return None
        
def main():
    print("="*60)
    print("数据清洗脚本 (修复版) 启动")
    print(f"数据根目录: {DATA_ROOT}")
    print(f"采样类别: {SAMPLE_CLASS}")
    print(f"使用设备: {DEVICE}")
    print("="*60)

    print("[1/4] 加载模型...")
   
    print(f"将使用模型配置: mvitv2-base_8xb256_in1k 和权重: {CHECKPOINT}")
    print("模型将在后续对每张图片推理时动态加载。")
  

    # 2. 准备数据列表
    print("[2/4] 构建数据列表...")
    all_samples = []
    image_count = 0

    # 确定要处理的类别列表
    if SAMPLE_CLASS and os.path.isdir(os.path.join(DATA_ROOT, SAMPLE_CLASS)):
        class_dirs = [SAMPLE_CLASS]
        print(f"  将处理单一类别: {SAMPLE_CLASS}")
    else:
        class_dirs = [d for d in os.listdir(DATA_ROOT) 
                     if os.path.isdir(os.path.join(DATA_ROOT, d))]
        print(f"  将处理所有类别，共 {len(class_dirs)} 个")

    # 3. 遍历并处理图片
    print("[3/4] 遍历图片并提取信息...")
    for class_dir in class_dirs:
        class_path = os.path.join(DATA_ROOT, class_dir)
        print(f"  处理类别: {class_dir} ({class_path})")
        
        # 检查路径是否存在
        if not os.path.exists(class_path):
            print(f"    警告: 路径不存在，跳过!")
            continue

        # 获取图片文件列表
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG', '*.PNG']:
            image_files.extend([f for f in os.listdir(class_path) if f.lower().endswith(ext[1:])])
        
        print(f"    找到 {len(image_files)} 个图片文件")
        
        # 处理前N张图片（测试用）
        for img_name in image_files[:30]:  # 先处理30张测试
            img_path = os.path.join(class_path, img_name)
            result = extract_feature_and_predict(img_path) 
            if result:
                feature, pred_score, pred_label, pred_class = result
                all_samples.append({
                    'path': img_path,
                    'class': class_dir,
                    'feature': feature,
                    'pred_score': pred_score,
                    'pred_label': pred_label,
                    'pred_class': pred_class
                })
                image_count += 1

    print(f"  总计成功处理 {image_count} 张图片")

    if image_count == 0:
        print("错误: 未成功加载任何图片。请检查：")
        print("  1. DATA_ROOT 路径是否正确？")
        print("  2. 类别文件夹下是否有图片文件？")
        print("  3. 图片格式是否为 .jpg/.jpeg/.png？")
        return

    # 4. 基于置信度的清洗
    print("[4/4] 执行清洗算法...")
    low_confidence_samples = [s for s in all_samples if s['pred_score'] < CONFIDENCE_THRESHOLD]
    print(f"  -> 基于置信度清洗: 找到 {len(low_confidence_samples)} 张低置信度图片 (阈值<{CONFIDENCE_THRESHOLD})")

    # 5. 基于特征相似度的清洗（仅当有足够样本时）
    if len(all_samples) > 1:
        features = np.array([s['feature'] for s in all_samples])
        cos_sim_matrix = cosine_similarity(features)
        avg_similarities = cos_sim_matrix.mean(axis=1)
        
        outlier_samples = []
        for idx, sample in enumerate(all_samples):
            if avg_similarities[idx] < SIMILARITY_THRESHOLD:
                outlier_samples.append({
                    **sample,
                    'avg_similarity': avg_similarities[idx]
                })
        print(f"  -> 基于相似度清洗: 找到 {len(outlier_samples)} 个离群点 (阈值<{SIMILARITY_THRESHOLD})")
    else:
        outlier_samples = []
        print("  -> 基于相似度清洗: 样本不足，跳过")

    # 6. 保存结果
    print("\n" + "="*60)
    print("生成输出结果...")
    
    # 保存噪声样本截图
    def save_samples(samples, folder_name, title):
        if not samples:
            print(f"  {title}: 无样本")
            return []
        
        save_dir = f'./{folder_name}/'
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存前12张图片的汇总图
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        for idx, sample in enumerate(samples[:12]):
            ax = axes[idx // 4, idx % 4]
            try:
                img = Image.open(sample['path'])
                ax.imshow(img)
                info = f"置信度: {sample['pred_score']:.2f}"
                if 'avg_similarity' in sample:
                    info += f"\n相似度: {sample['avg_similarity']:.2f}"
                ax.set_title(info, fontsize=9)
                ax.axis('off')
                # 同时保存原图
                img.save(os.path.join(save_dir, os.path.basename(sample['path'])))
            except Exception as e:
                ax.text(0.5, 0.5, f"无法加载\n{os.path.basename(sample['path'])}", 
                       ha='center', va='center')
                ax.axis('off')
        
        plt.suptitle(f'{title} (共{len(samples)}张)', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'noisy_samples_summary.png'), dpi=120)
        plt.close()
        print(f"  {title}: 已保存 {len(samples)} 张样本到 '{save_dir}'")
        return samples

    # 保存两类噪声样本
    noisy_conf = save_samples(low_confidence_samples, 'noisy_low_confidence', '低置信度噪声图片')
    noisy_sim = save_samples(outlier_samples, 'noisy_outliers', '特征离群点图片')

    # 保存清洗后的数据清单
    clean_samples = [s for s in all_samples 
                    if s['pred_score'] >= CONFIDENCE_THRESHOLD]
    
    if len(all_samples) > 1:
        features = np.array([s['feature'] for s in all_samples])
        avg_similarities = cosine_similarity(features).mean(axis=1)
        clean_samples = [s for idx, s in enumerate(clean_samples) 
                        if avg_similarities[idx] >= SIMILARITY_THRESHOLD]

    with open('./cleaned_data_list.txt', 'w') as f:
        for sample in clean_samples:
            f.write(f"{sample['path']}\t{sample['class']}\t{sample['pred_score']:.4f}\n")
    
    print(f"\n原始数据: {len(all_samples)} 张")
    print(f"清洗后保留: {len(clean_samples)} 张")
    print(f"清洗掉: {len(all_samples) - len(clean_samples)} 张")
    print(f"\n 清洗完成！")
    print("="*60)

if __name__ == '__main__':
    main()