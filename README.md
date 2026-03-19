# CF Multi-Agent Simulation

这是一个用于囊性纤维化（CF）多角色对话模拟的小项目，包含：

- `CF Specialist 🩺`
- `GP 👨‍⚕️`
- `Patient 🧑‍🦽`

核心机制：

1. **QTMD 提示结构**（Q/T/M/D）  
2. **自适应权重调度**（`adaptive_weight.py`）  
3. **博弈论增强模块**（`game_theory.py`）  
   - 更快的信息收集（coordination signal）
   - 以“未满足需求压力”驱动的平衡治疗方案（Nash bargaining 风格评分）
4. **场景预设**（`scenario_presets.py`）  
   - 预设 Patient / GP / CF Specialist 的核心需求  
   - 预设“患者第一次去 CF team”常见问题

## Quick Start

```bash
python main.py \
  --query "How should a balanced CF treatment plan be coordinated?" \
  --use_R 1 \
  --adaptive 1 \
  --use_game_theory 1 \
  --scenario first_visit \
  --rounds 4
```

输出日志目录（默认 `logs/`）：

- `metrics.csv`
- `weights.csv`
- `dialogues.jsonl`
