#!/usr/bin/env python3
"""
下载VSI-Bench数据集并转换为JSONL格式
VSI-Bench是一个视频空间推理数据集，视频文件需要从本地路径复制
"""

from datasets import load_dataset
import json
import os
import shutil
from collections import defaultdict

def download_vsibench(test_mode=False, max_samples=5, start_index=0):
    """下载VSI-Bench数据集并转换为JSONL格式"""
    
    print(f"开始处理VSI-Bench数据集... {'(测试模式，从索引' + str(start_index) + '开始处理' + str(max_samples) + '条数据)' if test_mode else ''}")
    
    # 源视频文件夹路径
    source_video_base = "dataset/VSI-Bench"
    
    # 目标文件夹路径
    target_video_folder = "dataset/VSI_videos"
    dataset_folder = "dataset"
    
    print(f"源视频路径: {source_video_base}")
    print(f"目标视频文件夹: {os.path.abspath(target_video_folder)}")
    
    # 创建目标文件夹
    os.makedirs(dataset_folder, exist_ok=True)
    os.makedirs(target_video_folder, exist_ok=True)
    
    source_video_available = os.path.exists(source_video_base)
    if source_video_available:
        print("✅ 源文件夹存在，将优先使用本地视频和parquet文件。")
    else:
        print(f"⚠️  源视频文件夹不存在: {source_video_base}")
        print("将继续尝试加载本地parquet或从Hub加载数据集元信息。")
        print("如果缺少本地视频文件，只会保留已存在于目标目录中的视频。")
    
    # 加载VSI-Bench数据集（从本地路径加载）
    try:
        # 先尝试从本地parquet文件加载
        parquet_path = os.path.join(source_video_base, "test-00000-of-00001.parquet")
        if os.path.exists(parquet_path):
            print(f"📂 从本地加载数据集: {parquet_path}")
            ds = load_dataset("parquet", data_files={"test": parquet_path})
            test_data = ds['test']
            print(f"✅ 数据集加载成功！数据量: {len(test_data)}")
        else:
            # 如果本地没有parquet文件，尝试从Hub加载
            print(f"⚠️  本地parquet文件不存在，尝试从Hub加载...")
            ds = load_dataset("nyu-visionx/VSI-Bench")
            test_data = ds['test']
            print(f"✅ 数据集加载成功！数据量: {len(test_data)}")
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        return
    
    # 统计需要的视频文件
    video_files_needed = set()
    # 在测试模式下只处理指定范围的数据
    if test_mode:
        end_index = min(start_index + max_samples, len(test_data))
        process_data = [test_data[i] for i in range(start_index, end_index)]
    else:
        process_data = test_data
    
    for sample in process_data:
        dataset_name = sample['dataset']
        scene_name = sample['scene_name']
        video_path = f"{dataset_name}/{scene_name}.mp4"
        video_files_needed.add(video_path)
    
    print(f"📊 需要的视频文件数量: {len(video_files_needed)}")
    
    # 复制视频文件
    copied_videos = set()
    failed_videos = []
    
    print("📹 开始准备视频文件...")
    for video_path in video_files_needed:
        source_path = os.path.join(source_video_base, video_path)
        target_path = os.path.join(target_video_folder, video_path.replace('/', '_'))
        
        # 创建目标子文件夹
        target_dir = os.path.dirname(target_path)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
        
        if os.path.exists(target_path):
            print(f"  ⏭️  跳过: {video_path} (目标已存在)")
            copied_videos.add(video_path)
            continue

        if os.path.exists(source_path):
            try:
                shutil.copy2(source_path, target_path)
                print(f"  ✅ 复制: {video_path}")
                copied_videos.add(video_path)
            except Exception as e:
                print(f"  ❌ 复制失败: {video_path} - {e}")
                failed_videos.append(video_path)
        else:
            print(f"  ❌ 视频不可用: {video_path}")
            print(f"     缺少源文件: {source_path}")
            failed_videos.append(video_path)
    
    print(f"\n📊 视频复制结果:")
    print(f"  成功复制: {len(copied_videos)} 个")
    print(f"  复制失败: {len(failed_videos)} 个")
    
    if failed_videos:
        print(f"  失败的视频: {failed_videos[:5]}..." if len(failed_videos) > 5 else f"  失败的视频: {failed_videos}")
    
    # 转换为JSONL格式
    print("\n📝 开始转换为JSONL格式...")
    all_converted_data = []
    total_processed = 0
    skipped_no_video = 0
    
    # 统计各类信息
    dataset_stats = defaultdict(int)
    question_type_stats = defaultdict(int)
    
    for idx, sample in enumerate(process_data):
        try:
            dataset_name = sample['dataset']
            scene_name = sample['scene_name']
            video_path = f"{dataset_name}/{scene_name}.mp4"
            
            # 检查视频是否成功复制
            if video_path not in copied_videos:
                skipped_no_video += 1
                continue
            
            # 构建视频路径（相对于dataset目录）
            video_filename = video_path.replace('/', '_')
            video_relative_path = f"VSI_videos/{video_filename}"
            
            # 构建对话内容
            conversations = []
            
            # 获取基础信息
            question = sample.get('question', '')
            ground_truth = sample.get('ground_truth', '')
            options = sample.get('options')
            
            # 判断问题类型并构建相应的问题文本
            question_text = question
            output_type = "text"
            answer = str(ground_truth)
            
            # 判断是否为MCQ类型：ground_truth是字母且options不为None
            is_mcq = (isinstance(ground_truth, str) and 
                     len(ground_truth) == 1 and
                     ground_truth.isalpha() and 
                     options is not None)
            
            # 判断是否为Number类型：ground_truth是数字(或数字字符串)且options为None
            is_number = False
            if options is None:
                try:
                    # 尝试将ground_truth转换为数字
                    float(ground_truth)
                    is_number = True
                except (ValueError, TypeError):
                    is_number = False
            
            if is_mcq:
                # MCQ类型：添加选项信息
                output_type = "MCQ"
                if options and len(options) > 0:
                    question_text += "\nSelect from the following choices.\n"
                    for i, choice in enumerate(options):
                        question_text += f"({chr(65+i)}) {choice}\n"
                # 答案保持为字母格式
                answer = str(ground_truth)
                
            elif is_number:
                # Number类型：直接使用数字答案
                output_type = "Number"
                answer = str(ground_truth)
            else:
                # 其他类型：保持原格式
                output_type = "text"
                answer = str(ground_truth)
            
            # 添加人类问题
            if question_text:
                conversations.append({
                    "from": "human",
                    "value": question_text
                })
            
            # 添加答案
            if ground_truth is not None:
                conversations.append({
                    "from": "gpt",
                    "value": answer
                })
            
            # 构建JSON条目
            json_entry = {
                "id": f"VSIBench_{sample.get('id', idx)}",
                "image": [],  # VSI-Bench是视频数据集，没有静态图像
                "video": [video_relative_path],  # 视频路径
                "conversations": conversations,
                "task": sample.get('question_type', 'unknown'),
                "input_type": "video",
                "output_type": output_type,
                "data_source": "VSI-Bench",
                "others": {},
                "subtask": ""
            }
            
            all_converted_data.append(json_entry)
            total_processed += 1
            
            # 统计信息
            dataset_stats[dataset_name] += 1
            question_type_stats[sample.get('question_type', 'unknown')] += 1
            
            if total_processed % 500 == 0:
                print(f"  已处理 {total_processed} 条数据...")
                
        except Exception as e:
            print(f"  处理数据 {idx} 时出错: {e}")
            continue
    
    print(f"\n📊 数据处理完成!")
    print(f"  总原始数据: {len(process_data)} 条")
    print(f"  成功处理: {total_processed} 条")
    print(f"  跳过(无视频): {skipped_no_video} 条")
    
    # 保存JSONL文件
    json_filename = 'VSI_Bench_test.jsonl' if test_mode else 'VSI_Bench.jsonl'
    json_path = f'dataset/{json_filename}'
    print(f"\n💾 保存JSONL文件到: {os.path.abspath(json_path)}")
    
    if all_converted_data:
        with open(json_path, 'w', encoding='utf-8') as f:
            for item in all_converted_data:
                json.dump(item, f, ensure_ascii=False, separators=(',', ':'))
                f.write('\n')
        print(f"✅ JSONL文件保存成功!")
    else:
        print("⚠️  警告: 没有数据需要保存!")
    
    # 输出统计信息
    print(f"\n📈 数据统计:")
    print(f"  总数据量: {len(all_converted_data)} 条")
    print(f"  JSONL文件: {json_path}")
    print(f"  视频文件夹: {target_video_folder}")
    
    print(f"\n📊 数据来源分布:")
    for dataset, count in sorted(dataset_stats.items()):
        print(f"  {dataset}: {count} 条")
    
    print(f"\n📊 问题类型分布:")
    for question_type, count in sorted(question_type_stats.items()):
        print(f"  {question_type}: {count} 条")
    
    # 打印第一条数据作为示例
    if all_converted_data:
        print(f"\n📄 第一条数据示例:")
        print(json.dumps(all_converted_data[0], ensure_ascii=False, indent=2))
    
    print(f"\n🎉 VSI-Bench数据集处理完成!")


if __name__ == "__main__":
    # download_vsibench(test_mode=True, max_samples=5, start_index=956)
    download_vsibench(test_mode=False)
