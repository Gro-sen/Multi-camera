"""
真实运行数据分析脚本
用于毕业论文的量化分析章节
"""
import json
import os
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import pandas as pd
import numpy as np


class RealDataAnalyzer:
    """真实运行数据分析器"""
    
    def __init__(self, cases_dir='kb/cases', alarms_dir='alarms'):
        self.cases_dir = Path(cases_dir)
        self.alarms_dir = Path(alarms_dir)
        self.cases_data = []
        self.load_all_cases()

    def analyze_consistency_rate(self):
        """画面描述与决策一致率（逻辑自洽性）统计"""
        total = 0
        consistent = 0
        inconsistent_cases = []

        for case in self.cases_data:
            vf = case.get('vision_facts', {})
            fd = case.get('final_decision', {})
            fr = case.get('face_recognition', {})
            
            # 提取关键状态
            has_person = vf.get('has_person', False)
            badge_status = vf.get('badge_status', '不适用')
            enter_restricted = vf.get('enter_restricted_area', False)
            has_fire = vf.get('has_fire_or_smoke', False)
            has_electric = vf.get('has_electric_risk', False)
            
            is_alarm = fd.get('is_alarm', '否')
            alarm_level = fd.get('alarm_level', '无')
            alarm_reason = fd.get('alarm_reason', '')
            
            # 人脸状态简化
            face_matched = fr.get('matched', False) if fr else False
            face_detected = (fr.get('detected_faces', 0) > 0) if fr else False
            
            # --- 核心一致性校验逻辑 ---
            is_consistent = True
            reason_for_inconsistency = ""

            # 1. 环境风险硬性规则（最高优先级）
            # 如果有火灾/电气风险，必须报警且等级较高
            if has_fire or has_electric:
                if is_alarm != "是":
                    is_consistent = False
                    reason_for_inconsistency = "存在环境风险但未报警"
            
            # 2. 禁区入侵规则
            elif enter_restricted:
                # 进入禁区通常应报警，除非是极特殊情况（这里简化为必须报警）
                if is_alarm != "是":
                    is_consistent = False
                    reason_for_inconsistency = "进入禁区但未报警"

            # 3. 人员场景的逻辑自洽（最复杂的部分）
            elif has_person:
                # 场景 A: 白名单 + 佩戴工牌 -> 应无报警
                if face_matched and badge_status == "佩戴":
                    if is_alarm == "是":
                        is_consistent = False
                        reason_for_inconsistency = "白名单佩戴工牌却触发报警"
                
                # 场景 B: 陌生人 + 禁区(已在上面处理) / 陌生人 + 普通区 -> 应报警
                elif face_detected and not face_matched:
                    if is_alarm != "是":
                        is_consistent = False
                        reason_for_inconsistency = "陌生人未触发报警"

                # 场景 C: 有人但没检测到脸（可能是背影）+ 没戴工牌 -> 通常应报警(一般)
                elif not face_detected and badge_status == "未佩戴":
                    if is_alarm != "是":
                        is_consistent = False
                        reason_for_inconsistency = "未戴工牌且无脸未报警"

            # 4. 无人场景
            else: # not has_person
                if is_alarm == "是":
                    # 除非是纯环境风险（如没人但着火了），否则没人不应报人员相关的警
                    if not (has_fire or has_electric):
                        is_consistent = False
                        reason_for_inconsistency = "无人场景却触发人员相关报警"

            # 统计
            total += 1
            if is_consistent:
                consistent += 1
            else:
                inconsistent_cases.append({
                    "case_id": case.get('case_id'),
                    "reason": reason_for_inconsistency,
                    "vision": vf,
                    "decision": fd
                })

        if total > 0:
            rate = consistent / total * 100
            print("=" * 60)
            print("【分析8】画面描述与决策逻辑自洽率")
            print("=" * 60)
            print(f"总案例数: {total}")
            print(f"逻辑自洽案例数: {consistent} ({rate:.2f}%)")
            print(f"逻辑冲突案例数: {len(inconsistent_cases)}")
            
            # 打印前5个冲突原因，方便分析
            for item in inconsistent_cases[:5]:
                print(f"  - 案例 {item['case_id']}: {item['reason']}")
        else:
            print("无可用数据进行一致率统计")
    
    def load_all_cases(self):
        """加载所有案例数据"""
        print(f"正在加载案例数据...")
        case_files = list(self.cases_dir.glob('*.json'))
        print(f"找到 {len(case_files)} 个案例文件")
        
        for case_file in case_files:
            try:
                with open(case_file, 'r', encoding='utf-8') as f:
                    case_data = json.load(f)
                    self.cases_data.append(case_data)
            except Exception as e:
                print(f"加载失败 {case_file}: {e}")
        
        print(f"成功加载 {len(self.cases_data)} 个案例\n")
    
    def analyze_alarm_distribution(self):
        """1. 报警等级分布分析"""
        alarm_levels = [case.get('alarm_level', '未知') for case in self.cases_data]
        level_counter = Counter(alarm_levels)
        
        print("=" * 60)
        print("【分析1】报警等级分布")
        print("=" * 60)
        total = len(self.cases_data)
        for level, count in sorted(level_counter.items()):
            percentage = (count / total) * 100
            print(f"{level:8s}: {count:5d} 例 ({percentage:5.2f}%)")
        print(f"\n总计: {total} 例\n")
        
        return level_counter
    
    def analyze_alarm_reasons(self):
        """2. 报警原因分类统计"""
        reasons = [case.get('alarm_reason', '未知') for case in self.cases_data]
        reason_counter = Counter(reasons)
        
        print("=" * 60)
        print("【分析2】报警原因TOP 10")
        print("=" * 60)
        for reason, count in reason_counter.most_common(10):
            percentage = (count / len(self.cases_data)) * 100
            print(f"{reason:30s}: {count:5d} 例 ({percentage:5.2f}%)")
        print()
        
        return reason_counter
    
    def analyze_confidence_distribution(self):
        """3. 置信度分布分析"""
        confidences = [case.get('confidence', 0) for case in self.cases_data if case.get('confidence')]
        
        print("=" * 60)
        print("【分析3】置信度统计分析")
        print("=" * 60)
        print(f"平均置信度: {np.mean(confidences):.4f}")
        print(f"中位数:     {np.median(confidences):.4f}")
        print(f"标准差:     {np.std(confidences):.4f}")
        print(f"最小值:     {np.min(confidences):.4f}")
        print(f"最大值:     {np.max(confidences):.4f}")
        print(f"P50:        {np.percentile(confidences, 50):.4f}")
        print(f"P95:        {np.percentile(confidences, 95):.4f}")
        print(f"P99:        {np.percentile(confidences, 99):.4f}")
        print()
        
        return confidences
    
    def analyze_face_recognition_stats(self):
        """4. 人脸识别统计分析"""
        face_enabled = sum(1 for c in self.cases_data if c.get('face_recognition', {}).get('enabled'))
        faces_detected = sum(1 for c in self.cases_data if c.get('face_recognition', {}).get('detected_faces', 0) > 0)
        faces_matched = sum(1 for c in self.cases_data if c.get('face_recognition', {}).get('matched', False))
        
        print("=" * 60)
        print("【分析4】人脸识别统计")
        print("=" * 60)
        total = len(self.cases_data)
        print(f"启用人脸识别: {face_enabled}/{total} ({face_enabled/total*100:.2f}%)")
        print(f"检测到人脸:   {faces_detected}/{total} ({faces_detected/total*100:.2f}%)")
        print(f"成功匹配:     {faces_matched}/{total} ({faces_matched/total*100:.2f}%)")
        
        if faces_detected > 0:
            match_rate = faces_matched / faces_detected * 100
            print(f"匹配成功率:   {match_rate:.2f}%")
        print()
    
    def analyze_camera_distribution(self):
        """5. 摄像头分布分析"""
        cameras = [case.get('camera_id', 'unknown') for case in self.cases_data]
        camera_counter = Counter(cameras)
        
        print("=" * 60)
        print("【分析5】摄像头数据统计")
        print("=" * 60)
        for cam, count in sorted(camera_counter.items()):
            percentage = (count / len(self.cases_data)) * 100
            print(f"{cam:10s}: {count:5d} 例 ({percentage:5.2f}%)")
        print()
        
        return camera_counter
    
    def analyze_time_series(self):
        """6. 时间序列分析"""
        timestamps = []
        for case in self.cases_data:
            ts_str = case.get('timestamp', '')
            try:
                ts = datetime.fromisoformat(ts_str)
                timestamps.append(ts)
            except:
                continue
        
        if not timestamps:
            print("无法解析时间戳")
            return
        
        # 按日期统计
        date_counter = Counter([ts.strftime('%Y-%m-%d') for ts in timestamps])
        
        print("=" * 60)
        print("【分析6】时间分布统计")
        print("=" * 60)
        print(f"时间跨度: {min(timestamps)} 至 {max(timestamps)}")
        print(f"总天数:   {(max(timestamps) - min(timestamps)).days + 1} 天")
        print(f"日均案例: {len(timestamps) / max(1, (max(timestamps) - min(timestamps)).days + 1):.2f} 例/天")
        print()
        
        # 按小时统计
        hour_counter = Counter([ts.hour for ts in timestamps])
        print("按小时分布:")
        for hour in range(24):
            count = hour_counter.get(hour, 0)
            bar = '█' * (count // 5)
            print(f"{hour:02d}:00 - {count:4d} 例 {bar}")
        print()
        
        return date_counter, hour_counter
    
    def analyze_vision_facts(self):
        """7. 视觉事实分析"""
        has_person_count = sum(1 for c in self.cases_data if c.get('vision_facts', {}).get('has_person'))
        badge_not_worn = sum(1 for c in self.cases_data if c.get('vision_facts', {}).get('badge_status') == '未佩戴')
        restricted_area = sum(1 for c in self.cases_data if c.get('vision_facts', {}).get('enter_restricted_area'))
        fire_smoke = sum(1 for c in self.cases_data if c.get('vision_facts', {}).get('has_fire_or_smoke'))
        electric_risk = sum(1 for c in self.cases_data if c.get('vision_facts', {}).get('has_electric_risk'))
        
        print("=" * 60)
        print("【分析7】视觉检测结果统计")
        print("=" * 60)
        total = len(self.cases_data)
        print(f"检测到人:           {has_person_count:5d} ({has_person_count/total*100:.2f}%)")
        print(f"未佩戴工牌:         {badge_not_worn:5d} ({badge_not_worn/total*100:.2f}%)")
        print(f"进入限制区域:       {restricted_area:5d} ({restricted_area/total*100:.2f}%)")
        print(f"发现火灾/烟雾:      {fire_smoke:5d} ({fire_smoke/total*100:.2f}%)")
        print(f"电气安全隐患:       {electric_risk:5d} ({electric_risk/total*100:.2f}%)")
        print()
    
    def generate_charts(self, output_dir='analysis_results/charts'):
        """生成可视化图表"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("=" * 60)
        print("正在生成可视化图表...")
        print("=" * 60)
        
        # 图表1: 报警等级分布饼图
        fig, ax = plt.subplots(figsize=(10, 8))
        alarm_levels = [case.get('alarm_level', '未知') for case in self.cases_data]
        level_counter = Counter(alarm_levels)
        
        colors = ['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
        wedges, texts, autotexts = ax.pie(
            level_counter.values(), 
            labels=level_counter.keys(),
            autopct='%1.1f%%',
            colors=colors[:len(level_counter)],
            startangle=90
        )
        ax.set_title('报警等级分布', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'alarm_level_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 已保存: alarm_level_distribution.png")
        
        # 图表2: 报警原因TOP10柱状图
        fig, ax = plt.subplots(figsize=(12, 6))
        reasons = [case.get('alarm_reason', '未知') for case in self.cases_data]
        reason_counter = Counter(reasons)
        top_10 = reason_counter.most_common(10)
        
        x_pos = range(len(top_10))
        bars = ax.bar(x_pos, [count for _, count in top_10], color='#3498db')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([reason for reason, _ in top_10], rotation=45, ha='right')
        ax.set_ylabel('案例数量', fontsize=12)
        ax.set_title('报警原因TOP 10', fontsize=16, fontweight='bold')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path / 'top_alarm_reasons.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 已保存: top_alarm_reasons.png")
        
        # 图表3: 置信度分布直方图
        fig, ax = plt.subplots(figsize=(10, 6))
        confidences = [case.get('confidence', 0) for case in self.cases_data if case.get('confidence')]
        
        ax.hist(confidences, bins=30, color='#1abc9c', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(confidences), color='red', linestyle='--', linewidth=2, label=f'平均值: {np.mean(confidences):.3f}')
        ax.axvline(np.median(confidences), color='orange', linestyle='--', linewidth=2, label=f'中位数: {np.median(confidences):.3f}')
        ax.set_xlabel('置信度', fontsize=12)
        ax.set_ylabel('频数', fontsize=12)
        ax.set_title('置信度分布', fontsize=16, fontweight='bold')
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_path / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 已保存: confidence_distribution.png")
        
        # 图表4: 时间序列趋势图
        fig, ax = plt.subplots(figsize=(14, 6))
        timestamps = []
        for case in self.cases_data:
            ts_str = case.get('timestamp', '')
            try:
                ts = datetime.fromisoformat(ts_str)
                timestamps.append(ts)
            except:
                continue
        
        if timestamps:
            date_counter = Counter([ts.strftime('%Y-%m-%d') for ts in timestamps])
            dates = sorted(date_counter.keys())
            counts = [date_counter[date] for date in dates]
            
            ax.plot(range(len(dates)), counts, marker='o', markersize=4, linewidth=1.5, color='#e74c3c')
            ax.set_xticks(range(len(dates)))
            ax.set_xticklabels(dates, rotation=45, ha='right')
            ax.set_xlabel('日期', fontsize=12)
            ax.set_ylabel('案例数量', fontsize=12)
            ax.set_title('每日案例数量趋势', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_path / 'daily_trend.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ 已保存: daily_trend.png")
        
        # 图表5: 小时分布热力图
        fig, ax = plt.subplots(figsize=(12, 6))
        hour_counter = Counter([ts.hour for ts in timestamps])
        hours = list(range(24))
        hour_counts = [hour_counter.get(h, 0) for h in hours]
        
        bars = ax.bar(hours, hour_counts, color='#9b59b6', alpha=0.7)
        ax.set_xlabel('小时', fontsize=12)
        ax.set_ylabel('案例数量', fontsize=12)
        ax.set_title('24小时案例分布', fontsize=16, fontweight='bold')
        ax.set_xticks(hours)
        plt.tight_layout()
        plt.savefig(output_path / 'hourly_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 已保存: hourly_distribution.png")
        
        # 图表6: 摄像头对比图
        fig, ax = plt.subplots(figsize=(10, 6))
        cameras = [case.get('camera_id', 'unknown') for case in self.cases_data]
        camera_counter = Counter(cameras)
        
        bars = ax.bar(camera_counter.keys(), camera_counter.values(), color=['#3498db', '#e67e22'])
        ax.set_xlabel('摄像头ID', fontsize=12)
        ax.set_ylabel('案例数量', fontsize=12)
        ax.set_title('各摄像头案例数量对比', fontsize=16, fontweight='bold')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(output_path / 'camera_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 已保存: camera_comparison.png")
        
        print(f"\n所有图表已保存至: {output_path}\n")
    
    def generate_report(self, output_file='analysis_results/report.txt'):
        """生成完整分析报告"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Multi-camera安防监控系统 - 真实运行数据分析报告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"案例总数: {len(self.cases_data)}\n")
            f.write(f"数据来源: kb/cases/\n\n")
            
            # 写入各项分析结果
            f.write("\n" + "=" * 80 + "\n")
            f.write("1. 报警等级分布\n")
            f.write("=" * 80 + "\n")
            alarm_levels = [case.get('alarm_level', '未知') for case in self.cases_data]
            level_counter = Counter(alarm_levels)
            total = len(self.cases_data)
            for level, count in sorted(level_counter.items()):
                percentage = (count / total) * 100
                f.write(f"{level:8s}: {count:5d} 例 ({percentage:5.2f}%)\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("2. 报警原因TOP 10\n")
            f.write("=" * 80 + "\n")
            reasons = [case.get('alarm_reason', '未知') for case in self.cases_data]
            reason_counter = Counter(reasons)
            for reason, count in reason_counter.most_common(10):
                percentage = (count / total) * 100
                f.write(f"{reason:30s}: {count:5d} 例 ({percentage:5.2f}%)\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("3. 置信度统计\n")
            f.write("=" * 80 + "\n")
            confidences = [case.get('confidence', 0) for case in self.cases_data if case.get('confidence')]
            f.write(f"平均置信度: {np.mean(confidences):.4f}\n")
            f.write(f"中位数:     {np.median(confidences):.4f}\n")
            f.write(f"标准差:     {np.std(confidences):.4f}\n")
            f.write(f"P50:        {np.percentile(confidences, 50):.4f}\n")
            f.write(f"P95:        {np.percentile(confidences, 95):.4f}\n")
            f.write(f"P99:        {np.percentile(confidences, 99):.4f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("4. 人脸识别统计\n")
            f.write("=" * 80 + "\n")
            face_enabled = sum(1 for c in self.cases_data if c.get('face_recognition', {}).get('enabled'))
            faces_detected = sum(1 for c in self.cases_data if c.get('face_recognition', {}).get('detected_faces', 0) > 0)
            faces_matched = sum(1 for c in self.cases_data if c.get('face_recognition', {}).get('matched', False))
            f.write(f"启用人脸识别: {face_enabled}/{total} ({face_enabled/total*100:.2f}%)\n")
            f.write(f"检测到人脸:   {faces_detected}/{total} ({faces_detected/total*100:.2f}%)\n")
            f.write(f"成功匹配:     {faces_matched}/{total} ({faces_matched/total*100:.2f}%)\n")
            if faces_detected > 0:
                f.write(f"匹配成功率:   {faces_matched/faces_detected*100:.2f}%\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("5. 视觉检测结果统计\n")
            f.write("=" * 80 + "\n")
            has_person = sum(1 for c in self.cases_data if c.get('vision_facts', {}).get('has_person'))
            badge_not_worn = sum(1 for c in self.cases_data if c.get('vision_facts', {}).get('badge_status') == '未佩戴')
            restricted = sum(1 for c in self.cases_data if c.get('vision_facts', {}).get('enter_restricted_area'))
            fire = sum(1 for c in self.cases_data if c.get('vision_facts', {}).get('has_fire_or_smoke'))
            electric = sum(1 for c in self.cases_data if c.get('vision_facts', {}).get('has_electric_risk'))
            f.write(f"检测到人:           {has_person:5d} ({has_person/total*100:.2f}%)\n")
            f.write(f"未佩戴工牌:         {badge_not_worn:5d} ({badge_not_worn/total*100:.2f}%)\n")
            f.write(f"进入限制区域:       {restricted:5d} ({restricted/total*100:.2f}%)\n")
            f.write(f"发现火灾/烟雾:      {fire:5d} ({fire/total*100:.2f}%)\n")
            f.write(f"电气安全隐患:       {electric:5d} ({electric/total*100:.2f}%)\n")
        
        print(f"分析报告已保存至: {output_path}\n")


if __name__ == '__main__':
    analyzer = RealDataAnalyzer()
    
    # 执行各项分析
    analyzer.analyze_alarm_distribution()
    analyzer.analyze_alarm_reasons()
    analyzer.analyze_confidence_distribution()
    analyzer.analyze_face_recognition_stats()
    analyzer.analyze_camera_distribution()
    analyzer.analyze_time_series()
    analyzer.analyze_vision_facts()
    analyzer.analyze_consistency_rate()
    
    # 生成图表和报告
    analyzer.generate_charts()
    analyzer.generate_report()
    
    print("=" * 60)
    print("✅ 分析完成！")
    print("=" * 60)