#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO11婴儿关键点检测训练脚本
使用config.yaml中的配置参数进行训练
"""

import os
import sys
import torch
import argparse
import traceback
from pathlib import Path

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from model import create_model
from trainer import YOLOTrainer
from config_utils import (
    load_training_config,
    validate_config,
    print_config_summary,
    check_environment,
    set_seed,
    create_experiment_name,
)


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='YOLO11婴儿关键点检测训练')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--device', type=str, default='auto',
                       help='训练设备 (auto/cpu/cuda)')
    parser.add_argument('--exp-name', type=str, default=None,
                       help='实验名称')
    parser.add_argument('--debug', action='store_true',
                       help='调试模式（减少训练轮数）')
    parser.add_argument('--no-early-stop', action='store_true',
                        help='禁用早停 (覆盖配置文件 early_stop)')
    parser.add_argument('--early-stop', action='store_true',
                        help='启用早停 (覆盖配置文件 early_stop)')
    
    args = parser.parse_args()
    
    print("🚀 YOLO11婴儿关键点检测训练")
    print("="*60)
    
    try:
        # 1. 检查环境
        if not check_environment():
            print("✗ 环境检查失败")
            return

        # 2. 加载配置（集中到 config_utils）
        config = load_training_config(args.config)

        # 调试模式：减少训练轮数
        if args.debug:
            config.update({
                'epochs': 5,
                'val_interval': 1,
                'save_interval': 2,
                'patience': 3
            })
            print("🐛 调试模式：使用较少的训练轮数")

        # 覆盖 early_stop 设置
        if args.no_early_stop:
            config['early_stop'] = False
            print("🚫 已通过命令行禁用早停")
        elif args.early_stop:
            config['early_stop'] = True
            print("✅ 已通过命令行启用早停")

        # 3. 验证配置（集中到 config_utils）
        if not validate_config(config):
            print("✗ 配置验证失败")
            return

        # 4. 打印配置摘要（集中到 config_utils）
        print_config_summary(config)

        # 5. 设置设备
        if args.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(args.device)
        print(f"\n🔧 使用设备: {device}")

        # 设置随机种子（集中到 config_utils）
        set_seed(config.get('seed', 42))

        # 6. 创建实验名称
        exp_name = args.exp_name or create_experiment_name(config)

        # 更新保存路径
        config['save_dir'] = os.path.join(config['save_dir'], exp_name)
        config['log_dir'] = os.path.join(config['log_dir'], exp_name)

        print(f"📝 实验名称: {exp_name}")
        print(f"💾 模型保存路径: {config['save_dir']}")
        print(f"📊 日志保存路径: {config['log_dir']}")

        # 7. 创建模型
        print("\n🏗️ 创建模型...")
        model = create_model(
            nc=config.get('nc', 1),
            nk=config.get('nk', 21)
        )
        model = model.to(device)

        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ 模型创建成功")
        print(f"  - 总参数量: {total_params:,}")
        print(f"  - 可训练参数: {trainable_params:,}")

        # 8. 创建训练器
        print("\n🎯 初始化训练器...")
        trainer = YOLOTrainer(model, config)
        print("✓ 训练器初始化完成")

        # 9. 开始训练
        print("\n🏃‍♂️ 开始训练...")
        print("="*60)

        trainer.train(resume_from=args.resume)

        # 10. 训练完成
        print("\n🎉 训练完成!")
        training_summary = trainer.get_training_summary()

        print("\n📈 训练总结:")
        print(f"  - 总轮数: {training_summary['total_epochs']}")
        print(f"  - 最佳验证损失: {training_summary['best_loss']:.4f}")
        if training_summary['final_train_loss']:
            print(f"  - 最终训练损失: {training_summary['final_train_loss']:.4f}")
        if training_summary['final_val_loss']:
            print(f"  - 最终验证损失: {training_summary['final_val_loss']:.4f}")

        print(f"\n💾 训练结果保存在: {config['save_dir']}")
        print(f"📊 训练日志保存在: {config['log_dir']}")

        # 11. 模型评估（可选）
        print("\n📊 正在进行模型评估...")
        try:
            eval_results = trainer.evaluate_model()
            print("✓ 模型评估完成")
            print(f"  - 验证总损失: {eval_results['validation_loss']:.4f}")
            for k, v in eval_results.get('metrics', {}).items():
                try:
                    print(f"  - {k}: {float(v):.4f}")
                except Exception:
                    pass
        except Exception as e:
            print(f"⚠ 模型评估失败: {e}")

        print("\n" + "="*60)
        print("🎯 训练任务完成!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n⚠ 训练被用户中断")
        
    except Exception as e:
        print(f"\n✗ 训练过程中发生错误: {e}")
        print("\n详细错误信息:")
        traceback.print_exc()
        
        print("\n💡 常见问题解决方案:")
        print("1. 检查数据路径是否正确")
        print("2. 确保数据集格式符合要求")
        print("3. 检查GPU内存是否足够（可尝试减小batch_size）")
        print("4. 确保所有依赖包已正确安装")


def quick_train():
    """快速训练函数（用于测试）"""
    print("🚀 快速训练模式")
    
    # 最小配置
    config = {
        'epochs': 3,
        'batch_size': 2,
        'lr': 0.001,
        'img_size': 320,  # 更小的图像尺寸
        'val_interval': 1,
        'save_interval': 1,
        'patience': 5,
        'data_root': 'dataset',
        'save_dir': 'runs/train/quick_test',
        'log_dir': 'runs/logs/quick_test',
        'nc': 1,
        'nk': 21
    }
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = create_model(nc=1, nk=21)
        model = model.to(device)
        
        trainer = YOLOTrainer(model, config)
        trainer.train()
        
        print("✓ 快速训练完成")
        
    except Exception as e:
        print(f"✗ 快速训练失败: {e}")


if __name__ == "__main__":
    # 检查是否有命令行参数
    if len(sys.argv) > 1:
        main()
    else:
        # 没有参数时提供选择
        print("YOLO11婴儿关键点检测训练脚本")
        print("="*40)
        print("1. 正常训练 (使用config.yaml)")
        print("2. 快速测试训练")
        print("3. 调试模式训练")
        print("4. 显示帮助信息")
        
        choice = input("\n请选择模式 (1-4): ").strip()
        
        if choice == "1":
            # 模拟命令行参数
            sys.argv = [sys.argv[0], '--config', 'config.yaml']
            main()
        elif choice == "2":
            quick_train()
        elif choice == "3":
            sys.argv = [sys.argv[0], '--config', 'config.yaml', '--debug']
            main()
        elif choice == "4":
            sys.argv = [sys.argv[0], '--help']
            main()
        else:
            print("无效选择，退出程序")