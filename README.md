# PHYBench Pipeline 建设计划

## 项目概述
**目标**: 构建大规模AI自动化流水线，实现从PDF原始文件到规范化JSON题目的全自动转换，支持动态题库扩展和大规模题海集建设。

**当前状况**: 人工流程效率低下（200人次/月 → 500题）
**目标效率**: AI自动化流水线（预期 >5000题/月）

---

## 系统要求

- **Python**: >= 3.12
- **操作系统**: Windows, macOS, Linux

## 安装

### 从源码安装（推荐）

```bash
# 克隆项目
git clone https://github.com/StephenQSstarThomas/phy-pipeline.git
cd phy-pipeline

# 安装项目
pip install -e .
```

### 开发环境安装

如果你想参与开发，请安装开发依赖：

```bash
# 安装项目和开发依赖
pip install -e .[dev]
```

开发依赖包括：
- pytest 8.3.3 - 测试框架
- pytest-asyncio 0.24.0 - 异步测试支持
- pytest-cov 5.0.0 - 代码覆盖率
- black 24.8.0 - 代码格式化
- isort 5.13.2 - import排序
- mypy 1.11.2 - 类型检查

## 使用方法

### 命令行接口

处理分离的题目和答案文件：
```bash
phy-pipeline process-simple questions.md answers.md --output-file output.json
```

处理混合的题目答案文件：
```bash
phy-pipeline process-mixed mixed_content.md --output-file mixed_output.json
```

题目答案匹配：
```bash
phy-pipeline match questions.md answers.md --output-file matched.json --confidence-threshold 0.8
```

### Python API

```python
from phy_pipeline import SimpleQAProcessor, MixedQAProcessor, QAMatcher

# 处理分离的题目和答案
processor = SimpleQAProcessor()
processor.process_files("questions.md", "answers.md", "output.json")

# 处理混合格式
mixed_processor = MixedQAProcessor()
mixed_processor.process_file("mixed.md", "output.json")

# 智能匹配
matcher = QAMatcher()
pairs = matcher.match_files("questions.md", "answers.md")
```

---

## Phase 1: 核心转换流水线 (Process 1-2)

### 1.1 PDF到Markdown转换模块
**技术栈**: MinerU API集成 + 本地处理能力
**已实现**: PHYBench内部题库的markdown格式（可直接转到1.2）

#### 具体编码任务:
- **pdf_processor.py**: 
  - 集成MinerU API调用接口
  - 支持批量PDF文件处理
  - 实现断点续传和错误重试机制
  - 添加进度跟踪和日志记录
- **markdown_parser.py**:
  - 解析MinerU输出的markdown格式
  - 识别数学公式、图片引用、表格等元素
  - 清理和标准化markdown内容

#### 预期输出:
```
input/raw_pdfs/exam_2024_q.pdf + exam_2024_a.pdf
→ output/markdown/exam_2024_q.md + exam_2024_a.md
```

### 1.2 题目-答案智能匹配模块 ✅ **已完成**
**技术方案**: AI辅助 + Rule-based双重验证

#### 具体编码任务:
- **qa_matcher.py**: ✅ 完成
  - 实现基于语义相似度的AI匹配算法
  - 开发rule-based匹配规则（题号、页码、关键词等）
  - 构建匹配置信度评分系统
  - 添加人工审核接口（低置信度案例）
- **simple_qa_processor.py**: ✅ 完成 
  - 简化版QA处理器，专门处理chap1案例
  - 支持官方API和中转站API调用
  - 实现内容修正和LaTeX格式优化
  - 智能题目-答案匹配
- **mixed_qa_processor.py**: ✅ 完成
  - 处理混合题目解答格式（题目与解答在同一文档）
  - 基于规则的题目解答智能分割
  - LLM辅助内容修正和格式优化
  - 自动图片引用提取和标准化

#### 实际输出:
```
output/
├── processed_chap1_corrected.json (17个QA对，分离式)
└── mixed_qa_processed.json (3个QA对，混合式)
```

**处理结果**:
- ✅ **分离式处理**: 从chap1.md + Solutions_June_2014.md → 17个QA对
- ✅ **混合式处理**: 从大题典.md → 3个QA对
- ✅ 应用AI内容修正和LaTeX格式化
- ✅ 生成规范JSON格式输出
- ✅ 自动图片引用处理 (<fig_001> 格式)

### 1.3 JSON格式转换模块
**核心功能**: 标准化数据结构转换

#### 具体编码任务:
- **json_converter.py**:
  - 实现markdown到规范JSON的转换
  - 处理LaTeX数学公式的保留和转换
  - 生成唯一题目ID（基于内容hash）
  - 实现题目分类算法（MATHEMATICAL, PHYSICS, etc.）
- **image_processor.py**:
  - 提取并处理markdown中的图片链接
  - 实现图片本地化存储
  - 生成规范的图片引用ID（fig_001格式）
  - 优化图片压缩和格式标准化

#### 目标JSON结构:
```json
{
    "id": "auto_generated_hash_id",
    "question": "Question content with <fig_001> references",
    "answer": "Answer content with <fig_002> references", 
    "class": "MATHEMATICAL",
    "figure": {
        "fig_001": "local_path_or_url",
        "fig_002": "local_path_or_url"
    },
    "metadata": {
        "source_file": "exam_2024.pdf",
        "difficulty": "auto_assessed",
        "topics": ["vector_analysis", "calculus"],
        "processing_timestamp": "2024-xx-xx"
    }
}
```

---

## Phase 2: 网站对接与展示 (Process 3)

### 2.1 网站API对接模块
#### 具体编码任务:
- **website_uploader.py**:
  - 实现与现有网站API的对接
  - 支持批量上传JSON文件
  - 添加上传状态跟踪和错误处理
  - 实现增量更新机制

### 2.2 UI优化建议
#### 具体编码任务:
- **frontend_enhancements.js**:
  - 优化图片与文字的混合排版
  - 实现LaTeX公式的实时渲染
  - 添加题目预览和编辑功能
  - 改进响应式设计

---

## Phase 3: 查重系统设计 (Process 5)

### 3.1 多层次查重架构
**技术方案**: 文本相似度 + 语义相似度 + 结构相似度

#### 具体编码任务:
- **duplicate_detector.py**:
  - **文本层面**: 实现编辑距离算法（Levenshtein）
  - **语义层面**: 集成sentence-transformers进行语义向量比较
  - **结构层面**: 比较题目结构（小题数量、公式数量等）
  - **图片层面**: 实现图像特征匹配（ORB/SIFT）

- **similarity_calculator.py**:
  - 多维度相似度加权计算
  - 可配置的相似度阈值设置
  - 生成详细的相似度报告

- **dedup_manager.py**:
  - 实现查重工作流管理
  - 支持批量查重处理
  - 提供查重结果可视化界面
  - 添加人工确认机制

#### 查重流程:
```
新题目 → 特征提取 → 与题库比较 → 相似度计算 → 阈值判断 → 人工确认 → 入库/拒绝
```

### 3.2 查重数据库设计
#### 具体编码任务:
- **dedup_database.py**:
  - 设计高效的相似度索引结构
  - 实现增量式特征库更新
  - 添加查重历史记录功能

---

## Phase 4: AI辅助题目修改 (Extension 4)

### 4.1 智能编辑建议模块
#### 具体编码任务:
- **ai_editor.py**:
  - 集成GPT/Claude API进行题目质量评估
  - 实现题目难度自动评级
  - 生成题目改进建议
  - 支持批量题目优化

- **quality_assessor.py**:
  - 检查题目完整性（题目、答案、图片）
  - 验证数学公式语法正确性
  - 评估题目清晰度和准确性

---

## Phase 5: 系统集成与优化

### 5.1 流水线调度器
#### 具体编码任务:
- **pipeline_orchestrator.py**:
  - 实现端到端流水线调度
  - 支持并行处理和负载均衡
  - 添加任务队列和状态管理
  - 实现失败重试和错误恢复

- **monitoring_dashboard.py**:
  - 实时处理进度监控
  - 系统性能指标统计
  - 错误日志分析和报警

### 5.2 配置管理与部署
#### 具体编码任务:
- **config_manager.py**:
  - 统一配置文件管理
  - 支持不同环境配置切换
  - 敏感信息加密存储

- **deployment_scripts/**:
  - Docker容器化配置
  - CI/CD流水线设置
  - 自动化测试脚本

---

## 技术栈总结

### 核心依赖:
- **AI模型**: OpenAI GPT-4 (v1.51.2), sentence-transformers (v3.1.1), LiteLLM (v1.48.14)
- **图像处理**: OpenCV (v4.10.0.84), Pillow (v10.4.0), imagehash (v4.3.1)
- **数据处理**: NumPy (v2.1.1), Pandas (v2.2.3), Pydantic (v2.9.2)
- **文件处理**: aiofiles (v24.1.0), beautifulsoup4 (v4.12.3), lxml (v5.3.0), markdownify (v0.13.1)
- **Web框架**: FastAPI (v0.115.0), uvicorn (v0.30.6)
- **CLI工具**: Typer (v0.12.5), Rich (v13.8.1)
- **网络请求**: requests (v2.32.3)
- **环境配置**: python-dotenv (v1.0.1)

### 性能目标:
- **处理速度**: >100题/小时
- **准确率**: 题目匹配 >95%, 查重检测 >98%
- **系统可用性**: 99.5%
- **扩展性**: 支持10万+题目规模

### 版本要求:
- **Python**: 最低要求 Python 3.12+
- **依赖管理**: 使用精确版本锁定确保环境一致性
- **开发工具**: 集成了最新版本的代码质量工具

---

## 开发者指南

### 项目结构
```
phy-pipeline/
├── src/phy_pipeline/          # 主要代码
│   ├── __init__.py
│   ├── cli.py                 # 命令行接口
│   ├── qa_matcher.py          # Q&A匹配器
│   ├── simple_qa_processor.py # 简单Q&A处理器
│   └── mixed_qa_processor.py  # 混合Q&A处理器
├── tests/                     # 测试文件
├── docs/                      # 文档
├── pyproject.toml             # 项目配置
├── README.md                  # 项目说明
└── .gitignore                # Git忽略文件
```

### 贡献指南

#### 开发环境设置
1. Fork 项目: https://github.com/StephenQSstarThomas/phy-pipeline
2. 克隆你的 fork:
   ```bash
   git clone https://github.com/your-username/phy-pipeline.git
   cd phy-pipeline
   ```
3. 安装开发依赖:
   ```bash
   pip install -e .[dev]
   ```

#### 代码质量
运行代码格式化和检查：
```bash
# 代码格式化
black src/ tests/
isort src/ tests/

# 类型检查
mypy src/

# 运行测试
pytest
```

#### 提交流程
1. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
2. 确保代码通过所有检查
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

---

## 项目链接

- **主页**: https://github.com/StephenQSstarThomas/phy-pipeline
- **源码**: https://github.com/StephenQSstarThomas/phy-pipeline.git
- **问题反馈**: https://github.com/StephenQSstarThomas/phy-pipeline/issues

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。
