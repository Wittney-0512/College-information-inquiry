### 国际高校信息/交互/数媒/AI+设计 调研工作流 - README

## 项目简介
- 通过自动检索与抽取，生成目标院校在以下方面的结构化报告（Markdown）：
  - 教育教学/学科特色/知识体系（课程清单、创新教学活动、教育模式）
  - 设计与学术研究成果（竞赛奖项、论文等，优先近年且附来源）
  - 学生培养与就业（分本科/硕士/博士规模、教育模式细节、近五年就业率/去向/薪资）
- 使用数据源：Tavily、Google Scholar（SerpAPI）、OpenAlex、Crossref、Wikidata、官网网页与 PDF。

## 环境要求
- 操作系统：Windows 10/11（其他平台也可）
- Python：3.9+（建议 3.10/3.11）
- 可联网，目标站点允许访问

## 关键依赖
- LLM 与链路：langchain、langchain-openai
- 搜索/数据源：tavily-python、serpapi、requests
- 解析：beautifulsoup4、pypdf
- 配置：python-dotenv

安装所有依赖（PowerShell）：
```bash
pip install -U langchain langchain-openai tavily-python serpapi python-dotenv requests beautifulsoup4 pypdf
```

## 环境变量配置
- 必需
  - OPENAI_API_KEY
- 建议
  - TAVILY_API_KEY（提升官网权威域命中率）
  - SERPAPI_API_KEY（Google Scholar 抓取）
- 可选（美国高校官方指标）
  - COLLEGE_SCORECARD_API_KEY

Windows PowerShell 持久设置（重开终端后生效）：
```bash
setx OPENAI_API_KEY "你的key"
setx TAVILY_API_KEY "你的key"
setx SERPAPI_API_KEY "你的key"
setx COLLEGE_SCORECARD_API_KEY "你的key"
```

或创建 `.env` 文件（与 `workflow.py` 同目录）：
```bash
OPENAI_API_KEY=你的key
TAVILY_API_KEY=你的key
SERPAPI_API_KEY=你的key
COLLEGE_SCORECARD_API_KEY=你的key
```

## 快速开始
1) 克隆/进入目录：
```bash
cd F:\xxx\xxx
```
2) 创建虚拟环境（可选但推荐）：
```bash
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```
3) 安装依赖（见上）
4) 配置环境变量（见上）
5) 运行：
```bash
python workflow.py
```
6) 按提示输入高校名称等参数，执行结束后在当前目录生成报告文件：
- `高校名_信息艺术设计专业调研报告_YYYYMMDDHHMMSS.md`

## 工作流说明（简版）
- 多路检索：
  - Tavily：权威域（.edu/.ac.uk/.edu.cn/.ac.cn）页面与目录直链
  - PDF/网页正文抓取：课程/培养方案/就业报告优先，抽取正文
  - Google Scholar（SerpAPI）：分页抓取近年论文列表
  - OpenAlex/Crossref：近年论文（最新优先，含 venue/年份/引用/DOI）
  - Wikidata：院系/单位信息补充
- 正则预抽：从网页/PDF中抽取近五年就业率/薪资，辅助 LLM 抽取
- LLM 抽取：按定制 Schema 输出结构化 JSON（创新教学活动、奖项、论文、培养规模、教育模式细节、就业5年逐年）
- 报告生成：汇总为 Markdown（保留来源可追溯性）

## 可配置项（在 `workflow.py` 内）
- 抓取强度：
  - Tavily `max_results=16`
  - 网页正文数量上限 `N=28`
  - PDF 优先抓取前 `10` 个
- 论文条数：OpenAlex `take=10`；整体目标 5–10 篇
- 输入时间范围：默认“近5年（2021-2025）”，可在启动时修改

## 常见问题
- ImportError/ModuleNotFoundError：确认依赖已安装，且使用激活的虚拟环境
- 超时/返回少：
  - 检查网络与 API Key 配额
  - 增大 `timeout` 或减少并发/抓取上限
- 论文为空：
  - 优化目标高校英文名
  - 调整年份范围（确保包含近年）
- 中文路径/空格：运行命令中的路径请使用引号

## 安全与合规
- 不要把 API Key 写入代码库，使用环境变量或 `.env`（忽略提交）
- 仅抓取公开页面与 PDF，不进行登录/爬墙行为

## 目录结构（最小）
- `workflow.py`：主工作流（你当前文件）
- `README.md`：说明文档（本文件）
- `.env`（可选）：环境变量文件
- `*_信息艺术设计专业调研报告_YYYYMMDDHHMMSS.md`：生成的报告