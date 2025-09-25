import os
import re
import json
import time
from datetime import datetime
from urllib.parse import quote, urlparse
from typing import List, Dict, Tuple, Optional
import concurrent.futures
import threading

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from tavily import TavilyClient
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

# ==============================
# 环境初始化与依赖说明
# ==============================
# 依赖安装（PowerShell）：
#   pip install -U langchain langchain-openai tavily-python serpapi python-dotenv requests beautifulsoup4 pypdf
# 必要环境变量：
#   OPENAI_API_KEY（必需）
#   TAVILY_API_KEY（建议，用于官网权威域检索）
#   SERPAPI_API_KEY（建议，用于 Google Scholar）
#   COLLEGE_SCORECARD_API_KEY（可选，仅对美国高校有效）
load_dotenv()
tavily_api_key = os.getenv("TAVILY_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")
college_scorecard_api_key = os.getenv("COLLEGE_SCORECARD_API_KEY")

# ==============================
# LLM 初始化（用于抽取与报告）
# ==============================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=openai_api_key,
    temperature=0.2
)

# ==============================
# 智能学校信息识别和标准化
# ==============================
def identify_and_standardize_university(user_input_name: str) -> Dict[str, str]:
    """智能识别并标准化大学信息"""
    print(f"🔍 正在智能识别学校信息: {user_input_name}")

    result = {
        "original_input": user_input_name,
        "standard_name_en": "",
        "standard_name_cn": "",
        "official_domain": "",
        "alternative_names": [],
        "country": "",
        "confidence": "low"
    }

    if not tavily_api_key:
        print("⚠️  未配置Tavily API，使用基础映射...")
        return fallback_university_mapping(user_input_name)

    client = TavilyClient(api_key=tavily_api_key)

    # 构建多样化搜索查询
    search_queries = [
        f'"{user_input_name}" university official website',
        f'"{user_input_name}" 大学 官网',
        f'{user_input_name} university homepage',
        f'{user_input_name} college official site',
        f'"{user_input_name}" university domain edu',
        f'"{user_input_name}" 高校 官方网站'
    ]

    print(f"📡 正在进行{len(search_queries)}次智能搜索...")

    # 收集所有搜索结果
    all_results = []
    for i, query in enumerate(search_queries):
        try:
            print(f"  搜索 {i+1}/{len(search_queries)}: {query[:40]}...")
            res = client.search(query=query, search_depth="basic", max_results=5)
            all_results.extend(res.get("results", []))
        except Exception as e:
            print(f"  ❌ 搜索失败: {str(e)[:30]}")
            continue

    if not all_results:
        print("⚠️  智能搜索无结果，使用备用方案...")
        return fallback_university_mapping(user_input_name)

    # 分析搜索结果
    print(f"📊 分析{len(all_results)}条搜索结果...")

    # 提取官方域名
    official_domains = []
    edu_domains = []

    for item in all_results:
        url = item.get("url", "")
        title = item.get("title", "").lower()
        content = item.get("content", "").lower()

        if url:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # 识别官方教育域名
            if any(pattern in domain for pattern in ['.edu', '.ac.', '.edu.cn', 'university', 'college']):
                if '.edu' in domain or '.edu.cn' in domain:
                    edu_domains.append(domain)
                elif 'university' in title or 'college' in title:
                    official_domains.append(domain)

    # 选择最可能的官方域名
    if edu_domains:
        result["official_domain"] = edu_domains[0]  # 优先选择.edu域名
        result["confidence"] = "high"
    elif official_domains:
        result["official_domain"] = official_domains[0]
        result["confidence"] = "medium"

    # 使用LLM进一步分析和标准化
    if result["official_domain"]:
        print(f"🎯 发现官方域名: {result['official_domain']}")
        enhanced_result = enhance_university_info_with_llm(user_input_name, all_results, result)
        return enhanced_result
    else:
        print("⚠️  未找到确定的官方域名，使用备用方案...")
        return fallback_university_mapping(user_input_name)

def enhance_university_info_with_llm(user_input: str, search_results: List[Dict], base_result: Dict) -> Dict:
    """使用LLM增强大学信息提取"""
    try:
        # 构建搜索结果摘要
        results_summary = []
        for result in search_results[:10]:  # 只取前10个结果
            title = result.get("title", "")
            url = result.get("url", "")
            content = result.get("content", "")[:200]  # 限制内容长度
            results_summary.append(f"Title: {title}\nURL: {url}\nContent: {content}\n---")

        results_text = "\n".join(results_summary)

        # LLM提示
        prompt = f"""
分析以下搜索结果，为大学"{user_input}"提取标准化信息：

搜索结果：
{results_text}

请提取以下信息并以JSON格式返回：
{{
    "standard_name_en": "标准英文名称",
    "standard_name_cn": "标准中文名称(如果有)",
    "official_domain": "官方网站域名",
    "alternative_names": ["别名1", "别名2"],
    "country": "所在国家",
    "confidence": "high/medium/low"
}}

要求：
1. standard_name_en必须是完整的官方英文名称
2. 如果有中文名称，填入standard_name_cn
3. official_domain只包含域名部分（如mit.edu）
4. alternative_names包含所有可能的别名
5. 严格按JSON格式返回
"""

        response = llm.invoke(prompt)

        # 解析LLM响应
        import json
        try:
            llm_result = json.loads(response.content)

            # 合并结果
            enhanced_result = base_result.copy()
            enhanced_result.update(llm_result)

            print(f"✅ LLM增强成功: {enhanced_result['standard_name_en']}")
            return enhanced_result

        except json.JSONDecodeError:
            print("⚠️  LLM响应解析失败，使用基础结果")
            return base_result

    except Exception as e:
        print(f"⚠️  LLM增强失败: {str(e)[:50]}")
        return base_result

def fallback_university_mapping(university: str) -> Dict[str, str]:
    """备用大学映射表"""
    mapping_table = {
        # 英文名称映射
        "mit": {
            "standard_name_en": "Massachusetts Institute of Technology",
            "standard_name_cn": "麻省理工学院",
            "official_domain": "mit.edu",
            "country": "United States",
            "confidence": "high"
        },
        "massachusetts institute of technology": {
            "standard_name_en": "Massachusetts Institute of Technology",
            "standard_name_cn": "麻省理工学院",
            "official_domain": "mit.edu",
            "country": "United States",
            "confidence": "high"
        },
        "stanford": {
            "standard_name_en": "Stanford University",
            "standard_name_cn": "斯坦福大学",
            "official_domain": "stanford.edu",
            "country": "United States",
            "confidence": "high"
        },
        "carnegie mellon": {
            "standard_name_en": "Carnegie Mellon University",
            "standard_name_cn": "卡内基梅隆大学",
            "official_domain": "cmu.edu",
            "country": "United States",
            "confidence": "high"
        },
        "royal college of art": {
            "standard_name_en": "Royal College of Art",
            "standard_name_cn": "英国皇家艺术学院",
            "official_domain": "rca.ac.uk",
            "country": "United Kingdom",
            "confidence": "high"
        },

        # 中文名称映射
        "麻省理工": {
            "standard_name_en": "Massachusetts Institute of Technology",
            "standard_name_cn": "麻省理工学院",
            "official_domain": "mit.edu",
            "country": "United States",
            "confidence": "high"
        },
        "麻省理工学院": {
            "standard_name_en": "Massachusetts Institute of Technology",
            "standard_name_cn": "麻省理工学院",
            "official_domain": "mit.edu",
            "country": "United States",
            "confidence": "high"
        },
        "斯坦福": {
            "standard_name_en": "Stanford University",
            "standard_name_cn": "斯坦福大学",
            "official_domain": "stanford.edu",
            "country": "United States",
            "confidence": "high"
        },
        "斯坦福大学": {
            "standard_name_en": "Stanford University",
            "standard_name_cn": "斯坦福大学",
            "official_domain": "stanford.edu",
            "country": "United States",
            "confidence": "high"
        },
        "清华": {
            "standard_name_en": "Tsinghua University",
            "standard_name_cn": "清华大学",
            "official_domain": "tsinghua.edu.cn",
            "country": "China",
            "confidence": "high"
        },
        "清华大学": {
            "standard_name_en": "Tsinghua University",
            "standard_name_cn": "清华大学",
            "official_domain": "tsinghua.edu.cn",
            "country": "China",
            "confidence": "high"
        },
        "北大": {
            "standard_name_en": "Peking University",
            "standard_name_cn": "北京大学",
            "official_domain": "pku.edu.cn",
            "country": "China",
            "confidence": "high"
        },
        "北京大学": {
            "standard_name_en": "Peking University",
            "standard_name_cn": "北京大学",
            "official_domain": "pku.edu.cn",
            "country": "China",
            "confidence": "high"
        },
        "皇家艺术学院": {
            "standard_name_en": "Royal College of Art",
            "standard_name_cn": "英国皇家艺术学院",
            "official_domain": "rca.ac.uk",
            "country": "United Kingdom",
            "confidence": "high"
        }
    }

    # 查找匹配
    university_lower = university.lower().strip()

    if university_lower in mapping_table:
        result = mapping_table[university_lower].copy()
        result["original_input"] = university
        result["alternative_names"] = [university]
        print(f"✅ 备用映射匹配: {result['standard_name_en']}")
        return result

    # 模糊匹配
    for key, info in mapping_table.items():
        if key in university_lower or university_lower in key:
            result = info.copy()
            result["original_input"] = university
            result["alternative_names"] = [university]
            result["confidence"] = "medium"
            print(f"✅ 备用映射模糊匹配: {result['standard_name_en']}")
            return result

    # 无匹配时的默认返回
    print(f"⚠️  无法识别学校: {university}，返回原始输入")
    return {
        "original_input": university,
        "standard_name_en": university,
        "standard_name_cn": "",
        "official_domain": "",
        "alternative_names": [university],
        "country": "",
        "confidence": "low"
    }

# 兼容性函数
def get_university_domain(university: str) -> str:
    """获取大学域名 - 兼容性函数"""
    # 这个函数现在只是为了保持向后兼容
    # 实际应该使用identify_and_standardize_university
    university_info = identify_and_standardize_university(university)
    return university_info.get("official_domain", "")

# ==============================
# 增强版网页正文与 PDF 抽取工具
# ==============================
def extract_text_from_html(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()
        text = soup.get_text("\n", strip=True)
        text = re.sub(r"\n{2,}", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()
    except Exception:
        return ""

def fetch_page_text(url: str, timeout: int = 25) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
        resp.raise_for_status()
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "text/html" not in ctype and not resp.text.lstrip().startswith("<!DOCTYPE html"):
            return ""
        return extract_text_from_html(resp.text)
    except Exception:
        return ""

def fetch_pdf_text(url: str, max_pages: int = 20) -> str:
    try:
        resp = requests.get(url, timeout=40)
        resp.raise_for_status()
        from io import BytesIO
        reader = PdfReader(BytesIO(resp.content))
        pages = []
        for i, page in enumerate(reader.pages[:max_pages]):
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        text = "\n".join(pages)
        text = re.sub(r"\n{2,}", "\n", text)
        return text.strip()
    except Exception:
        return ""

# ==============================
# 专业课程信息抓取工具
# ==============================
def fetch_structured_course_data(url: str) -> List[Dict]:
    """结构化解析课程数据 - 优化版"""
    try:
        print(f"        正在解析课程页面...")
        response = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        courses = []

        # 扩展课程选择器，包含更多可能的课程容器
        course_selectors = [
            'table.course', 'table.curriculum', 'table[class*="course"]',
            'ul.course-list', 'ol.course-list', 'div.course-item',
            'div[class*="course"]', 'div[class*="curriculum"]',
            'div[class*="subject"]', 'div[class*="class"]',
            'tr', 'li', 'p', 'div'  # 更通用的选择器
        ]

        course_elements = []
        for selector in course_selectors:
            elements = soup.select(selector)
            if elements:
                course_elements.extend(elements)
                if len(course_elements) > 50:  # 增加元素数量
                    break

        print(f"          找到{len(course_elements)}个潜在课程元素")

        # 大幅扩展课程格式正则表达式
        course_patterns = [
            # 标准学术格式
            r'([A-Z]{2,5}\.?\s*\d{3}[A-Z]?(?:J|H)?)\s+(.+?)\s*\((\d+(?:\s+\d+)*)\)',
            r'([A-Z]{2,5}\s*\d{3}[A-Z]?)\s+(.+?)(?:\s*[-–]\s*(.+))?',

            # 学分格式
            r'(.+?)\s*\((\d+)\s*credits?\)',
            r'(.+?)\s*\((\d+)\s*units?\)',
            r'(.+?)\s*\((\d+)\s*ECTS\)',

            # 编号格式
            r'\d+\.\s*(.+?)(?:\s*[-–]\s*(.+))?',
            r'(\d+)\s+(.+?)(?:\s*[-–]\s*(.+))?',

            # 冒号分隔格式
            r'^(.+?):\s*(.+?)$',
            r'^(.+?)\s*-\s*(.+?)$',

            # 简单文本格式（包含设计相关关键词）
            r'^([A-Z][^.!?]*(?:Design|Interaction|Computer|Interface|Media|Art|Digital|Information|HCI|UX|UI)[^.!?]*)$',
            r'^(.{10,100})$'  # 长度适中的文本行
        ]

        # 记录实际抓取的文本样本用于调试
        debug_samples = []

        for element in course_elements:
            course_text = element.get_text(strip=True)

            # 记录前10个样本用于调试
            if len(debug_samples) < 10:
                debug_samples.append(course_text[:100])

            # 调整过滤条件
            if len(course_text) < 5 or len(course_text) > 300:
                continue

            # 跳过明显的非课程内容
            skip_keywords = ['cookie', 'privacy', 'terms', 'footer', 'header', 'menu', 'navigation', 'subscribe', 'login']
            if any(keyword in course_text.lower() for keyword in skip_keywords):
                continue

            # 尝试各种模式匹配
            for pattern in course_patterns:
                try:
                    matches = re.finditer(pattern, course_text, re.IGNORECASE | re.MULTILINE)
                    for match in matches:
                        groups = match.groups()

                        if len(groups) >= 1:
                            # 更灵活的数据提取
                            if len(groups) >= 2 and groups[1]:
                                course_data = {
                                    "code": groups[0].strip() if groups[0] else "",
                                    "name": groups[1].strip() if groups[1] else "",
                                    "credits": groups[2].strip() if len(groups) > 2 and groups[2] else "",
                                    "source_url": url,
                                    "raw_text": course_text[:200],
                                    "pattern_used": pattern[:50]
                                }
                            else:
                                # 单个匹配组的情况
                                text = groups[0].strip()
                                course_data = {
                                    "code": "",
                                    "name": text,
                                    "credits": "",
                                    "source_url": url,
                                    "raw_text": course_text[:200],
                                    "pattern_used": pattern[:50]
                                }

                            # 更宽松的验证条件
                            if (course_data["name"] and
                                len(course_data["name"]) > 4 and
                                not any(existing["name"].lower() == course_data["name"].lower() for existing in courses)):
                                courses.append(course_data)

                except Exception as e:
                    continue

        # 输出调试信息
        print(f"          文本样本: {debug_samples[:3]}")
        print(f"          成功解析{len(courses)}门课程")

        return courses[:100]  # 增加返回数量限制

    except Exception as e:
        print(f"        ✗ 解析失败: {str(e)[:50]}")
        return []

def extract_numbers_with_context(text: str) -> List[Dict]:
    """从文本中提取带上下文的数字信息"""
    patterns = [
        # 学生数量
        r'(approximately\s+|over\s+|about\s+)?(\d{1,5})\s+(students?|graduates?|undergraduates?|postgraduates?)',
        r'(students?|graduates?|undergraduates?|postgraduates?)\s*:?\s*(\d{1,5})',
        r'enrollment\s*:?\s*(\d{1,5})',

        # 教师数量
        r'(\d{1,4})\s+(faculty|professors?|staff|instructors?)',
        r'(faculty|professors?|staff|instructors?)\s*:?\s*(\d{1,4})',

        # 百分比数据
        r'(\d{1,3})%\s+(employment|graduation|completion)',
        r'(employment|graduation|completion)\s+rate\s*:?\s*(\d{1,3})%',

        # 年份相关
        r'in\s+(20\d{2})[,\s]+(\d{1,5})\s+(students?|graduates?)',
        r'(20\d{2})\s+class\s*:?\s*(\d{1,4})\s+(students?|graduates?)'
    ]

    extracted_numbers = []

    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            groups = match.groups()

            # 获取匹配周围的上下文
            start = max(0, match.start() - 80)
            end = min(len(text), match.end() + 80)
            context = text[start:end].strip()

            # 提取数字和类别
            number = ""
            category = ""
            confidence = "medium"

            for group in groups:
                if group and group.isdigit():
                    number = group
                elif group and not group.isdigit():
                    if any(word in group.lower() for word in ["student", "faculty", "professor", "enrollment", "graduation"]):
                        category = group
                    elif "approximate" in group.lower() or "about" in group.lower():
                        confidence = "low"

            if number and category:
                extracted_numbers.append({
                    "number": number,
                    "category": category,
                    "context": context,
                    "confidence": confidence,
                    "pattern_matched": pattern
                })

    return extracted_numbers[:20]  # 限制返回数量

# ==============================
# 详细课程信息抓取函数
# ==============================
def fetch_detailed_course_info(standard_name_en: str, university_domain: str,
                              standard_name_cn: str = "", original_name: str = "") -> Dict[str, any]:
    """专门抓取课程详细信息 - 使用标准化学校信息"""
    course_info = {
        "course_catalog": [],
        "enrollment_stats": {},
        "faculty_info": []
    }

    if not tavily_api_key:
        return course_info

    client = TavilyClient(api_key=tavily_api_key)

    # 🆕 智能构造课程查询 - 避免中英文混合
    def get_course_search_names() -> Dict[str, List[str]]:
        """获取适合课程搜索的名称分类"""
        english_names = [standard_name_en] if standard_name_en else []
        chinese_names = [standard_name_cn] if standard_name_cn else []

        # 添加常见缩写
        if "Massachusetts Institute of Technology" in standard_name_en:
            english_names.append("MIT")
        elif "Stanford University" in standard_name_en:
            english_names.append("Stanford")
        elif "Carnegie Mellon University" in standard_name_en:
            english_names.extend(["CMU", "Carnegie Mellon"])
        elif "Royal College of Art" in standard_name_en:
            english_names.append("RCA")

        # 处理原始输入
        if original_name:
            if any(ord(c) > 127 for c in original_name):  # 包含中文
                chinese_names.append(original_name)
            else:  # 英文输入
                english_names.append(original_name)

        return {
            "english": list(dict.fromkeys(english_names))[:2],  # 去重并限制数量
            "chinese": list(dict.fromkeys(chinese_names))[:2]
        }

    search_names = get_course_search_names()
    course_patterns = []

    # 优先使用官方域名查询（通常是英文）
    if university_domain:
        course_patterns.extend([
            f'site:{university_domain} "course catalog" OR "curriculum"',
            f'site:{university_domain} "graduate courses" OR "course list"',
            f'site:{university_domain} "program requirements" OR "degree requirements"',
            f'site:{university_domain} "course schedule" OR "class schedule"',
            f'site:{university_domain} "admissions" OR "enrollment"'
        ])

    # 英文名称查询（主要用于国际搜索）
    for name in search_names["english"]:
        course_patterns.extend([
            f'"{name}" "course catalog" curriculum',
            f'"{name}" "graduate courses" requirements',
            f'"{name}" "enrollment statistics" OR "student numbers"',
            f'"{name}" "faculty" OR "professors" OR "staff"',
            f'"{name}" "program structure" OR "degree plan"'
        ])

    # 中文名称查询（用于中文教育网站和平台）
    for name in search_names["chinese"]:
        course_patterns.extend([
            f'"{name}" "课程设置" OR "专业课程"',
            f'"{name}" "研究生课程" OR "课程目录"',
            f'"{name}" "招生信息" OR "学生统计"',
            f'"{name}" "师资队伍" OR "教授名单"',
            f'"{name}" "培养方案" OR "学位要求"'
        ])

    print(f"    📚 正在抓取课程详细信息...")
    print(f"    🔍 英文搜索名称: {search_names['english']}")
    print(f"    🔍 中文搜索名称: {search_names['chinese']}")
    print(f"    📊 将执行{len(course_patterns)}个搜索查询...")

    for i, pattern in enumerate(course_patterns):
        try:
            print(f"      课程查询 {i+1}/{len(course_patterns)}: {pattern[:60]}...")
            res = client.search(
                query=pattern,
                search_depth="advanced",
                max_results=10,
                include_raw_content=False
            )

            for item in res.get("results", []):
                url = item.get("url", "")
                if not url:
                    continue

                # 专门解析课程页面
                if any(keyword in url.lower() for keyword in ["course", "curriculum", "catalog", "program"]):
                    print(f"        解析课程页面: {url[:60]}...")
                    detailed_content = fetch_structured_course_data(url)
                    if detailed_content:
                        course_info["course_catalog"].extend(detailed_content)
                        print(f"        ✓ 提取到{len(detailed_content)}门课程")

                # 提取统计数据
                elif any(keyword in url.lower() for keyword in ["statistics", "data", "facts", "numbers", "enrollment"]):
                    print(f"        解析统计页面: {url[:60]}...")
                    text = fetch_page_text(url)
                    if text:
                        stats = extract_numbers_with_context(text)
                        if stats:
                            course_info["enrollment_stats"][url] = stats
                            print(f"        ✓ 提取到{len(stats)}项统计数据")

        except Exception as e:
            print(f"        ✗ 查询失败: {str(e)[:50]}")
            continue

    print(f"    ✓ 课程信息抓取完成，共{len(course_info['course_catalog'])}门课程")
    return course_info

# ==============================
# 并行抓取工具（增强版带进度）
# ==============================
def parallel_fetch_pages(urls: List[str], max_workers: int = 12) -> List[Tuple[str, str]]:
    """并行抓取网页内容"""
    results = []

    def fetch_single(url):
        try:
            text = fetch_page_text(url)
            return (url, text) if len(text) >= 200 else None
        except Exception:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(fetch_single, url): url for url in urls}
        completed = 0
        total = len(urls)

        for future in concurrent.futures.as_completed(future_to_url):
            result = future.result()
            completed += 1
            if completed % 10 == 0 or completed == total:
                print(f"    网页抓取进度: {completed}/{total} ({completed/total*100:.1f}%)")
            if result:
                results.append(result)

    return results

def parallel_fetch_pdfs(urls: List[str], max_workers: int = 8) -> List[Tuple[str, str]]:
    """并行抓取PDF内容"""
    results = []

    def fetch_single_pdf(url):
        try:
            text = fetch_pdf_text(url, max_pages=20)
            return (url, text) if len(text) >= 200 else None
        except Exception:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(fetch_single_pdf, url): url for url in urls}
        completed = 0
        total = len(urls)

        for future in concurrent.futures.as_completed(future_to_url):
            result = future.result()
            completed += 1
            print(f"    PDF抓取进度: {completed}/{total} ({completed/total*100:.1f}%)")
            if result:
                results.append(result)

    return results

# ==============================
# 就业数据预抽（从网页/PDF中正则抓近年"就业率/薪资"）
# ==============================
def preextract_employment_stats(text: str, years: List[int]) -> Dict[str, Dict[str, str]]:
    """
    返回格式：
    { "2021": {"employment_rate": "95%", "median_salary": "$85,000"},
      "2022": {"employment_rate": "93%", "median_salary": "£30,000"} }
    """
    out: Dict[str, Dict[str, str]] = {}
    if not text:
        return out
    # 统一空白
    t = re.sub(r"[ \t]+", " ", text)
    # 百分比就业率：e.g., 2022 employment rate 95% | 就业率 95% (2022)
    pct_patterns = [
        r"(?P<y>20[1-3]\d)[^\n]{0,40}?(employment rate|就业率|就业去向落实率)[^\n]{0,20}?(?P<val>\d{1,3}\s?%)",
        r"(employment rate|就业率|就业去向落实率)[^\n]{0,30}?(?P<val>\d{1,3}\s?%)[^\n]{0,40}?(?P<y>20[1-3]\d)"
    ]
    # 薪资：中位数/平均 e.g., median salary $85,000 | 平均薪资 20,000 RMB/月
    sal_patterns = [
        r"(?P<y>20[1-3]\d)[^\n]{0,40}?(median salary|平均薪资|薪资中位数|平均年薪)[^\n]{0,20}?(?P<val>[\$£€]?\s?\d{2,3}[,\.]?\d{0,3}(?:\s?(USD|RMB|CNY|GBP|EUR|元|万元|万|k))?)",
        r"(median salary|平均薪资|薪资中位数|平均年薪)[^\n]{0,30}?(?P<val>[\$£€]?\s?\d{2,3}[,\.]?\d{0,3}(?:\s?(USD|RMB|CNY|GBP|EUR|元|万元|万|k))?)[^\n]{0,40}?(?P<y>20[1-3]\d)"
    ]
    yset = set(years)
    def clamp_year(y: str) -> str:
        try:
            yi = int(y)
            return str(yi) if yi in yset else ""
        except Exception:
            return ""
    for pat in pct_patterns:
        for m in re.finditer(pat, t, flags=re.I):
            y = clamp_year(m.group("y"))
            val = m.group("val").replace(" ", "")
            if y:
                out.setdefault(y, {})
                # 取最大值（避免低可信覆盖高可信）
                if "employment_rate" not in out[y] or (val.endswith("%") and int(re.sub(r"\D","",val)) > int(re.sub(r"\D","",out[y]["employment_rate"]))):
                    out[y]["employment_rate"] = val
    for pat in sal_patterns:
        for m in re.finditer(pat, t, flags=re.I):
            y = clamp_year(m.group("y"))
            val = m.group("val").strip()
            if y:
                out.setdefault(y, {})
                if "median_salary" not in out[y]:
                    out[y]["median_salary"] = val
    return out

# ==============================
# 增强版社交媒体和学术平台内容抓取（带详细进度提示）
# ==============================
def fetch_social_media_content(standard_name_en: str, majors: str,
                              standard_name_cn: str = "", original_name: str = "") -> List[str]:
    """抓取社交媒体和学术平台内容 - 智能选择平台适配的学校名称"""
    content = []

    # 🆕 智能构建平台适配的名称策略
    def get_platform_optimized_names(platform_type: str) -> List[str]:
        """根据平台类型返回最适合的学校名称"""

        # 准备名称候选列表
        english_names = [standard_name_en] if standard_name_en else []
        chinese_names = [standard_name_cn] if standard_name_cn else []
        original_names = [original_name] if original_name else []

        # 添加常见英文缩写 (先检查是否非空)
        if standard_name_en and "Massachusetts Institute of Technology" in standard_name_en:
            english_names.append("MIT")
        elif standard_name_en and "Stanford University" in standard_name_en:
            english_names.append("Stanford")
        elif standard_name_en and "Carnegie Mellon University" in standard_name_en:
            english_names.extend(["CMU", "Carnegie Mellon"])
        elif standard_name_en and "Royal College of Art" in standard_name_en:
            english_names.extend(["RCA"])

        # 📝 添加调试信息
        # print(f"      调试 - {platform_type}: en={english_names}, cn={chinese_names}, orig={original_names}")

        if platform_type == "english_professional":
            # 英文职业平台：优先标准英文名 > 缩写 > 原始输入（如果是英文）
            names = english_names.copy()
            if original_name and not any(ord(c) > 127 for c in original_name):  # 检查是否包含中文字符
                names.append(original_name)
            return names[:2]  # 限制数量

        elif platform_type == "english_creative":
            # 英文创意平台：优先缩写 > 标准英文名
            names = []
            if len(english_names) > 1:  # 如果有缩写
                names.extend(english_names[1:])  # 先添加缩写
                names.append(english_names[0])   # 再添加完整名称
            else:
                names.extend(english_names)
            return names[:2]

        elif platform_type == "english_academic":
            # 英文学术平台：优先完整标准名称 > 缩写
            return english_names[:2]

        elif platform_type == "english_media":
            # 英文媒体平台：均衡使用标准名称和缩写
            return english_names[:2]

        elif platform_type == "chinese":
            # 中文平台：优先中文名 > 英文名
            names = chinese_names.copy()
            if original_name and any(ord(c) > 127 for c in original_name):  # 包含中文
                names.append(original_name)
            names.extend(english_names[:1])  # 添加一个英文名作为备选
            return names[:2]

        else:
            # 通用策略：英文名优先
            return english_names[:2]

    # 根据平台特性构建查询
    query_groups = {
        "LinkedIn职业平台": [],
        "技术博客平台": [],
        "设计作品平台": [],
        "设计媒体网站": [],
        "学术和技术平台": [],
        "科技新闻平台": []
    }

    # LinkedIn职业平台 - 英文职业平台策略
    linkedin_names = get_platform_optimized_names("english_professional")
    for name in linkedin_names:
        query_groups["LinkedIn职业平台"].extend([
            f'site:linkedin.com "{name}" "graduate" HCI UX UI',
            f'site:linkedin.com "{name}" "alumni" "human computer interaction"',
            f'site:linkedin.com "{name}" "student" "interaction design"',
            f'site:linkedin.com "{name}" "tech" "design" job'
        ])

    # 技术博客平台 - 英文学术策略
    blog_names = get_platform_optimized_names("english_academic")
    for name in blog_names:
        query_groups["技术博客平台"].extend([
            f'site:medium.com "{name}" HCI "user interface"',
            f'site:medium.com "{name}" "design program" interaction',
            f'site:medium.com "{name}" "student experience" UX',
            f'site:dev.to "{name}" "human computer interaction"'
        ])

    # 设计作品平台 - 英文创意策略
    creative_names = get_platform_optimized_names("english_creative")
    for name in creative_names:
        query_groups["设计作品平台"].extend([
            f'site:behance.net "{name}" interaction design',
            f'site:behance.net "{name}" "user interface" UI',
            f'site:dribbble.com "{name}" UX design',
            f'site:dribbble.com "{name}" "user experience"'
        ])

    # 设计媒体网站 - 英文媒体策略
    media_names = get_platform_optimized_names("english_media")
    for name in media_names:
        query_groups["设计媒体网站"].extend([
            f'site:core77.com "{name}" interaction',
            f'site:designboom.com "{name}" digital',
            f'site:dezeen.com "{name}" technology',
            f'site:fastcompany.com "{name}" "design" innovation'
        ])

    # 学术和技术平台 - 英文学术策略
    academic_names = get_platform_optimized_names("english_academic")
    for name in academic_names:
        query_groups["学术和技术平台"].extend([
            f'site:acm.org "{name}" CHI UIST',
            f'site:ieee.org "{name}" "human computer interaction"',
            f'site:arxiv.org "{name}" HCI interaction',
            f'site:researchgate.net "{name}" "user interface"'
        ])

    # 科技新闻平台 - 英文媒体策略
    tech_names = get_platform_optimized_names("english_media")
    for name in tech_names:
        query_groups["科技新闻平台"].extend([
            f'site:techcrunch.com "{name}" startup design',
            f'site:wired.com "{name}" technology innovation',
            f'site:theverge.com "{name}" tech design',
            f'site:ycombinator.com "{name}" product design'
        ])

    # 统计实际使用的名称
    all_used_names = set()
    for names in [linkedin_names, blog_names, creative_names, media_names, academic_names, tech_names]:
        all_used_names.update(names)

    print(f"    🌐 将搜索多个社交媒体和设计平台...")
    print(f"    📝 智能适配平台名称: {list(all_used_names)}")
    print(f"    🎯 LinkedIn用名: {linkedin_names}")
    print(f"    🎨 设计平台用名: {creative_names}")
    print(f"    📚 学术平台用名: {academic_names}")

    if not tavily_api_key:
        print("    [跳过] 未配置Tavily API Key，社交媒体抓取不可用")
        return content

    client = TavilyClient(api_key=tavily_api_key)
    total_groups = len(query_groups)
    current_group = 0

    for group_name, queries in query_groups.items():
        current_group += 1
        print(f"  正在搜索{group_name}... ({current_group}/{total_groups})")

    for i, query in enumerate(all_queries):
        try:
            print(f"    查询 {i+1}/{len(all_queries)}: {query[:50]}...")
            res = client.search(query=query, max_results=5, search_depth="basic")

            for item in res.get("results", []):
                url = item.get("url", "")
                if not url:
                    continue

                print(f"      正在抓取: {url[:60]}...")
                text = fetch_page_text(url)

                if len(text) >= 300:
                    content.append(f"[社交媒体] {url}\n{text[:6000]}")
                    print(f"      ✓ 成功抓取 ({len(text)}字符)")
                else:
                    print(f"      ✗ 内容过短 ({len(text)}字符)")

                time.sleep(0.3)  # 避免请求过快

        except Exception as e:
            print(f"      ✗ 查询失败: {str(e)[:50]}")
            continue

    print(f"  社交媒体抓取完成，共获取到{len(content)}条内容")
    return content

# ==============================
# 外部搜索与数据源工具
# ==============================
class ResearchTools:
    @staticmethod
    def init_tavily_tool():
        if not tavily_api_key:
            return None
        client = TavilyClient(api_key=tavily_api_key)
        def run(query: str):
            try:
                res = client.search(
                    query=query,
                    search_depth="advanced",
                    include_answer=True,
                    max_results=8
                )
                answer = res.get("answer") or ""
                sources = "\n".join([item.get("url", "") for item in res.get("results", []) if item.get("url")])
                out = (answer.strip() + ("\n\nSources:\n" + sources if sources else "")).strip()
                return out or "[No results]"
            except Exception as e:
                return f"[Tavily 调用失败] {e}"
        return Tool(name="TavilyUniversitySearch", func=run, description="Tavily 高校官网检索（优先权威域）")

    @staticmethod
    def init_google_scholar_tool():
        if not serpapi_api_key:
            return None
        def run(query: str, start: int = 0, ylo: str = "", yhi: str = ""):
            try:
                url = "https://serpapi.com/search.json"
                params = {"engine": "google_scholar", "q": query, "api_key": serpapi_api_key, "num": 10, "start": start}
                if ylo: params["as_ylo"] = ylo
                if yhi: params["as_yhi"] = yhi
                r = requests.get(url, params=params, timeout=30); r.raise_for_status()
                data = r.json()
                items = data.get("organic_results", [])
                lines = []
                for it in items:
                    title = it.get("title", "")
                    authors = (it.get("publication_info") or {}).get("summary", "")
                    link = it.get("link", "")
                    cited = (it.get("inline_links") or {}).get("cited_by", {}).get("total", "")
                    year = (it.get("publication_info") or {}).get("year", "")
                    lines.append(f"{title} | {authors} | {year} | Cited: {cited} | {link}")
                return "\n".join(lines) if lines else "[No results]"
            except Exception as e:
                return f"[Google Scholar(SerpAPI) 调用失败] {e}"
        return Tool(name="GoogleScholarResearch", func=run, description="Google Scholar 学术补充（SerpAPI）")

def openalex_recent_papers_latest(institution: str, ylo: str, yhi: str, limit: int = 30, take: int = 10) -> str:
    try:
        filter_parts = []
        if ylo: filter_parts.append(f"from_publication_date:{ylo}-01-01")
        if yhi: filter_parts.append(f"to_publication_date:{yhi}-12-31")
        filter_q = ",".join(filter_parts)
        search_q = quote(institution.strip())
        url = f"https://api.openalex.org/works?search={search_q}&filter={filter_q}&per_page={limit}&sort=publication_date:desc"
        r = requests.get(url, timeout=30); r.raise_for_status()
        data = r.json().get("results", [])
        rows = []
        for w in data:
            title = w.get("title","").strip()
            year = (w.get("publication_year") or "")
            cited = w.get("cited_by_count", 0)
            venue = (w.get("host_venue") or {}).get("display_name","").strip()
            auths = [a.get("author",{}).get("display_name","") for a in (w.get("authorships") or [])]
            authors = ", ".join([a for a in auths if a][:8])
            doi = w.get("doi","")
            if title and year and venue:
                rows.append(f"{title} | {authors} | {venue} | {year} | Cited: {cited} | {doi or ''}")
            if len(rows) >= take:
                break
        return "\n".join(rows) if rows else "[OpenAlex(最新) 无结果]"
    except Exception as e:
        return f"[OpenAlex(最新) 错误] {e}"

def openalex_recent_papers_optimized(keywords: str, ylo: str, yhi: str, limit: int = 20, take: int = 5) -> str:
    """优化的OpenAlex论文搜索"""
    try:
        # 构建更精确的搜索查询
        search_parts = []

        # 处理机构名称
        if any(name in keywords for name in ["MIT", "Massachusetts Institute"]):
            search_parts.append("authorships.institutions.display_name:Massachusetts Institute of Technology")
        elif "Stanford" in keywords:
            search_parts.append("authorships.institutions.display_name:Stanford University")
        elif any(name in keywords for name in ["CMU", "Carnegie Mellon"]):
            search_parts.append("authorships.institutions.display_name:Carnegie Mellon University")
        else:
            # 通用机构搜索
            inst_name = keywords.split()[0]
            search_parts.append(f"authorships.institutions.display_name:{inst_name}")

        # 添加主题过滤
        if "CHI" in keywords or "human computer interaction" in keywords:
            search_parts.append("concepts.display_name:Human-computer interaction")
        elif "UX" in keywords or "user interface" in keywords:
            search_parts.append("concepts.display_name:User interface")

        # 构建最终查询
        if search_parts:
            filter_q = ",".join(search_parts)
        else:
            # 回退到关键词搜索
            search_q = quote(keywords.strip())
            filter_q = f"title_and_abstract.search:{search_q}"

        # 添加时间过滤
        if ylo and yhi:
            filter_q += f",from_publication_date:{ylo}-01-01,to_publication_date:{yhi}-12-31"

        url = f"https://api.openalex.org/works?filter={filter_q}&per_page={limit}&sort=cited_by_count:desc"

        headers = {"User-Agent": "Academic Research Tool (mailto:research@example.com)"}
        r = requests.get(url, timeout=30, headers=headers)
        r.raise_for_status()

        data = r.json().get("results", [])
        rows = []

        for w in data:
            title = w.get("title", "").strip()
            year = w.get("publication_year", "")
            cited = w.get("cited_by_count", 0)
            venue = (w.get("host_venue") or {}).get("display_name", "").strip()

            # 获取第一作者和机构
            authorships = w.get("authorships", [])
            if authorships:
                first_author = authorships[0].get("author", {}).get("display_name", "")
                institutions = [auth.get("institutions", [{}])[0].get("display_name", "")
                             for auth in authorships[:3] if auth.get("institutions")]
                author_info = f"{first_author} ({', '.join(institutions[:2])})"
            else:
                author_info = "Unknown"

            doi = w.get("doi", "")

            if title and year and cited >= 1:  # 至少有1次引用
                rows.append(f"{title} | {author_info} | {venue} | {year} | Cited: {cited} | {doi or ''}")

            if len(rows) >= take:
                break

        return "\n".join(rows) if rows else "[OpenAlex 无结果]"

    except Exception as e:
        return f"[OpenAlex 错误] {e}"

def crossref_recent_papers(institution: str, ylo: str, yhi: str, rows: int = 30) -> str:
    try:
        params = {
            "query.affiliation": institution,
            "filter": f"from-pub-date:{ylo}-01-01,to-pub-date:{yhi}-12-31" if ylo and yhi else None,
            "rows": rows,
            "sort": "is-referenced-by-count",
            "order": "desc"
        }
        params = {k:v for k,v in params.items() if v}
        r = requests.get("https://api.crossref.org/works", params=params, timeout=30)
        r.raise_for_status()
        items = r.json().get("message",{}).get("items",[])
        lines = []
        for it in items:
            title = (it.get("title") or [""])[0]
            year = (it.get("issued") or {}).get("date-parts", [[None]])[0][0]
            cited = it.get("is-referenced-by-count", 0)
            container = it.get("container-title", [""])[0]
            doi = it.get("DOI","")
            if doi:
                lines.append(f"{title} | {container} | {year} | Cited: {cited} | https://doi.org/{doi}")
            else:
                lines.append(f"{title} | {container} | {year} | Cited: {cited}")
        return "\n".join(lines) if lines else "[Crossref 无结果]"
    except Exception as e:
        return f"[Crossref 错误] {e}"

def enhanced_academic_search(standard_name_en: str, majors: str, year_from: str, year_to: str,
                           standard_name_cn: str = "", original_name: str = "") -> str:
    """增强版学术论文搜索 - 使用标准化学校信息"""
    all_papers = []

    print("    📚 正在进行优化的多关键词学术搜索...")

    # 🆕 智能使用多种名称变体构建搜索
    all_search_names = [name for name in [standard_name_en, original_name, standard_name_cn] if name]

    # 优化关键词组合策略
    # 1. 更精确的机构名称处理 - 使用标准化名称
    institution_variants = list(all_search_names)  # 使用所有有效的名称变体

    # 添加常见缩写和别名（如果适用）
    if "Massachusetts Institute of Technology" in standard_name_en:
        institution_variants.append("MIT")
    elif "Stanford University" in standard_name_en:
        institution_variants.append("Stanford")
    elif "Carnegie Mellon University" in standard_name_en:
        institution_variants.extend(["CMU", "Carnegie Mellon"])
    elif "Royal College of Art" in standard_name_en:
        institution_variants.extend(["RCA", "Royal College of Art"])

    # 去重
    institution_variants = list(dict.fromkeys(institution_variants))

    # 2. 更有针对性的关键词组合
    base_hci_terms = ["human computer interaction", "HCI", "user interface", "UX", "interaction design"]
    specific_terms = ["information design", "digital media", "AI design", "user experience"]

    keywords_combinations = []

    # 为每个机构变体创建搜索组合
    for inst in institution_variants:
        for hci_term in base_hci_terms:
            keywords_combinations.append(f'"{inst}" {hci_term}')

        for spec_term in specific_terms:
            keywords_combinations.append(f'"{inst}" {spec_term}')

    # 3. 添加特定会议和期刊的搜索
    venues = ["CHI", "UIST", "CSCW", "DIS", "TEI", "ASSETS", "UbiComp"]
    for inst in institution_variants[:2]:  # 限制机构数量，避免过多查询
        for venue in venues:
            keywords_combinations.append(f'"{inst}" {venue}')

    print(f"    🔍 将执行{len(keywords_combinations)}次优化搜索...")
    print(f"    📝 使用机构名称变体: {institution_variants[:3]}")  # 只显示前3个避免输出过长

    # OpenAlex 优化搜索
    print("    OpenAlex搜索中...")
    success_count = 0
    for i, keywords in enumerate(keywords_combinations):
        try:
            print(f"      搜索 {i+1}/{len(keywords_combinations)}: {keywords[:50]}...")

            # 优化OpenAlex API调用
            papers = openalex_recent_papers_optimized(keywords, year_from, year_to, limit=20, take=5)
            if papers and "[OpenAlex 无结果]" not in papers:
                all_papers.append(papers)
                success_count += 1
                print(f"      ✓ 找到相关论文")
            else:
                print(f"      - 无相关结果")

            # 添加短暂延迟避免API限制
            time.sleep(0.2)

        except Exception as e:
            print(f"      ✗ 搜索失败: {str(e)[:30]}")
            continue

    print(f"    OpenAlex搜索完成: {success_count}/{len(keywords_combinations)} 成功")

    # Crossref 搜索（减少数量但提高质量）
    print("    Crossref搜索中...")
    crossref_success = 0
    for i, keywords in enumerate(keywords_combinations[:10]):  # 只用前10个最重要的
        try:
            print(f"      Crossref搜索 {i+1}/10: {keywords[:50]}...")
            papers = crossref_recent_papers(keywords, year_from, year_to, rows=15)
            if papers and "[Crossref 无结果]" not in papers:
                all_papers.append(papers)
                crossref_success += 1
                print(f"      ✓ 找到相关论文")
            else:
                print(f"      - 无相关结果")
        except Exception as e:
            print(f"      ✗ 搜索失败: {str(e)[:30]}")
            continue

    print(f"    Crossref搜索完成: {crossref_success}/10 成功")
    print(f"    学术搜索总计完成，共收集到{len(all_papers)}组论文数据")

    return "\n\n".join(all_papers) if all_papers else "[学术搜索无结果]"

def college_scorecard_programs(school_name: str) -> str:
    api_key = college_scorecard_api_key
    if not api_key:
        return "[CollegeScorecard 未配置 API Key]"
    try:
        fields = [
            "id","school.name","latest.student.size",
            "latest.earnings.10_yrs_after_entry.median",
            "latest.completion.rate_suppressed.overall",
            "latest.admissions.admission_rate.overall"
        ]
        url = "https://api.data.gov/ed/collegescorecard/v1/schools"
        params = {"api_key": api_key, "school.name": school_name, "per_page": 1, "fields": ",".join(fields)}
        r = requests.get(url, params=params, timeout=30); r.raise_for_status()
        res = r.json().get("results", [])
        return json.dumps(res, ensure_ascii=False, indent=2) if res else "[CollegeScorecard 无结果]"
    except Exception as e:
        return f"[CollegeScorecard 错误] {e}"

def wikidata_programs_and_units(institution: str, limit: int = 30) -> str:
    try:
        query = f"""
        SELECT ?unit ?unitLabel WHERE {{
          ?inst rdfs:label "{institution}"@en.
          ?unit wdt:P361 ?inst.
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }} LIMIT {limit}
        """
        url = "https://query.wikidata.org/sparql"
        headers = {"Accept": "application/sparql-results+json"}
        r = requests.get(url, params={"query": query}, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json().get("results",{}).get("bindings",[])
        lines = [b["unitLabel"]["value"] for b in data if "unitLabel" in b]
        return "\n".join(lines) if lines else "[Wikidata 无结果]"
    except Exception as e:
        return f"[Wikidata 错误] {e}"

# ==============================
# 数据验证和补充机制
# ==============================
def verify_and_enrich_data(extracted_data: Dict, university: str) -> Dict:
    """验证并丰富提取的数据"""

    # 验证数字的合理性
    if "student_employment" in extracted_data:
        student_data = extracted_data["student_employment"]
        if "student_scale" in student_data:
            for year_data in student_data["student_scale"]:
                count = year_data.get("count", "")
                if isinstance(count, str) and count.isdigit() and int(count) > 10000:
                    year_data["count"] = "暂未公开（数据异常）"
                    year_data["note"] = "原始数据疑似异常，已标记"

    # 检查缺失的关键信息
    missing_fields = []
    education_data = extracted_data.get("education_teaching", {})
    research_data = extracted_data.get("research_achievements", {})

    if not education_data.get("core_courses") or education_data.get("core_courses") == ["暂未公开"]:
        missing_fields.append("核心课程")
    if not research_data.get("academic_papers_latest"):
        missing_fields.append("学术论文")

    if missing_fields:
        print(f"  检测到缺失字段: {', '.join(missing_fields)}，进行补充搜索...")
        additional_data = targeted_search_for_missing_fields(university, missing_fields)
        extracted_data = merge_additional_data(extracted_data, additional_data)

    return extracted_data

def targeted_search_for_missing_fields(university: str, missing_fields: List[str]) -> Dict:
    """针对缺失字段的定向搜索 - 智能语言适配"""
    additional_data = {}

    if not tavily_api_key:
        return additional_data

    client = TavilyClient(api_key=tavily_api_key)

    # 🆕 智能判断大学名称的语言类型
    def determine_university_language(name: str):
        """判断大学名称的主要语言"""
        return "chinese" if any(ord(c) > 127 for c in name) else "english"

    university_language = determine_university_language(university)

    # 根据语言类型构建不同的查询模板
    if university_language == "english":
        field_queries = {
            "核心课程": [
                f'"{university}" "required courses" OR "core curriculum"',
                f'"{university}" "graduate courses" course list',
                f'"{university}" "program requirements" OR "degree requirements"'
            ],
            "学术论文": [
                f'"{university}" faculty publications recent papers',
                f'"{university}" research output OR "research publications"',
                f'"{university}" CHI UIST CSCW papers OR publications'
            ]
        }
    else:  # 中文名称
        field_queries = {
            "核心课程": [
                f'"{university}" "必修课程" OR "核心课程"',
                f'"{university}" "研究生课程" 课程表',
                f'"{university}" "培养方案" OR "学位要求"'
            ],
            "学术论文": [
                f'"{university}" 教师 学术论文 最新成果',
                f'"{university}" 科研产出 OR "学术发表"',
                f'"{university}" CHI UIST CSCW 论文 OR 学术成果'
            ]
        }

    print(f"    🔍 补充搜索使用 {university_language} 语言策略")

    for field in missing_fields:
        if field in field_queries:
            field_data = []
            for query in field_queries[field]:
                try:
                    print(f"    补充搜索 {field}: {query[:50]}...")
                    results = client.search(query=query, search_depth="advanced", max_results=8)
                    for result in results.get("results", []):
                        content = fetch_page_text(result.get("url"))
                        if len(content) > 1000:
                            field_data.append(content[:5000])
                except Exception:
                    continue
            additional_data[field] = field_data

    return additional_data

def merge_additional_data(original_data: Dict, additional_data: Dict) -> Dict:
    """合并补充数据到原始数据中"""
    # 这里可以实现更复杂的数据合并逻辑
    # 暂时简单地在data_source中添加补充搜索标记
    if additional_data:
        sources = original_data.get("data_source", [])
        sources.append("补充定向搜索")
        original_data["data_source"] = sources

    return original_data

# ==============================
# 交互输入
# ==============================
def get_user_input():
    print("=== 信息艺术设计国际高校调研 ===")
    target_university = input("1. 输入目标高校名称（如：MIT、Royal College of Art）：").strip()
    
    research_majors = ["信息设计", "人机交互", "数字媒体艺术", "AI+设计"]
    print(f"2. 调研专业范畴（默认：{','.join(research_majors)}）\n   如需调整，输入序号（逗号分隔，如1,3）；无需调整按回车：")
    for i, major in enumerate(research_majors, 1):
        print(f"   {i}. {major}")
    major_input = input().strip()
    if major_input:
        selected_indices = [int(idx)-1 for idx in major_input.split(",") if idx.isdigit() and 0 <= int(idx)-1 < len(research_majors)]
        picked = [research_majors[idx] for idx in selected_indices]
        if picked:
            research_majors = picked
    
    research_dimensions = ["教育教学与学科特色", "设计与学术研究成果", "学生培养与就业"]
    print(f"3. 调研核心维度（默认：{','.join(research_dimensions)}）\n   如需调整，输入序号（逗号分隔，如1,2）；无需调整按回车：")
    for i, dim in enumerate(research_dimensions, 1):
        print(f"   {i}. {dim}")
    dim_input = input().strip()
    if dim_input:
        selected_indices = [int(idx)-1 for idx in dim_input.split(",") if idx.isdigit() and 0 <= int(idx)-1 < len(research_dimensions)]
        picked = [research_dimensions[idx] for idx in selected_indices]
        if picked:
            research_dimensions = picked
    
    data_time_range = input("4. 数据时间范围（默认：近5年（2021-2025）），如需调整请输入：").strip() or "近5年（2021-2025）"
    
    return {
        "target_university": target_university,
        "research_majors": research_majors,
        "research_dimensions": research_dimensions,
        "data_time_range": data_time_range
    }

# ==============================
# 参数汇总与检索词生成（扩展查询多样性）
# ==============================
def summarize_research_params(user_input):
    major_en_map = {
        "信息设计": "Information Design",
        "人机交互": "Human-Computer Interaction",
        "数字媒体艺术": "Digital Media Art", 
        "AI+设计": "AI+Design"
    }
    dim_en_map = {
        "教育教学与学科特色": "Education & Discipline Features",
        "设计与学术研究成果": "Design & Academic Achievements",
        "学生培养与就业": "Student Training & Employment"
    }
    ylo = yhi = ""
    m = re.search(r"(\d{4}).*?(\d{4})", user_input["data_time_range"])
    if m: ylo, yhi = m.group(1), m.group(2)
    
    # 🆕 智能识别和标准化大学信息
    print("\n🔍 === 智能识别学校信息 ===")
    university_info = identify_and_standardize_university(user_input["target_university"])

    summary = {
        "target_university": user_input["target_university"],  # 保留用户原始输入
        "university_info": university_info,  # 新增：标准化的大学信息
        "standard_name_en": university_info["standard_name_en"],  # 标准英文名
        "standard_name_cn": university_info["standard_name_cn"],  # 标准中文名
        "official_domain": university_info["official_domain"],   # 官方域名
        "research_majors_cn": ",".join(user_input["research_majors"]),
        "research_majors_en": " ".join([major_en_map[mj] for mj in user_input["research_majors"]]),
        "research_dimensions_cn": ",".join(user_input["research_dimensions"]),
        "research_dimensions_en": " ".join([dim_en_map[d] for d in user_input["research_dimensions"]]),
        "data_time_range": user_input["data_time_range"],
        "year_from": ylo, "year_to": yhi
    }
    
    # 🆕 智能分类学校名称，避免中英文混合查询
    def categorize_university_names():
        """将学校名称按语言分类"""
        english_names = []
        chinese_names = []
    
        # 处理标准英文名
        if summary['standard_name_en']:
            english_names.append(summary['standard_name_en'])

        # 处理标准中文名
        if summary['standard_name_cn']:
            chinese_names.append(summary['standard_name_cn'])

        # 处理原始输入
        original = summary['target_university']
        if original:
            if any(ord(c) > 127 for c in original):  # 包含中文字符
                chinese_names.append(original)
            else:  # 英文输入
                english_names.append(original)

        # 添加英文缩写
        if "Massachusetts Institute of Technology" in summary['standard_name_en']:
            english_names.append("MIT")
        elif "Stanford University" in summary['standard_name_en']:
            english_names.append("Stanford")
        elif "Carnegie Mellon University" in summary['standard_name_en']:
            english_names.extend(["CMU", "Carnegie Mellon"])
        elif "Royal College of Art" in summary['standard_name_en']:
            english_names.append("RCA")

        # 去重
        english_names = list(dict.fromkeys(english_names))
        chinese_names = list(dict.fromkeys(chinese_names))

        return {
            "english": english_names[:2],  # 限制数量
            "chinese": chinese_names[:2]
        }

    name_categories = categorize_university_names()

    print(f"📝 智能分类名称:")
    print(f"   英文名称: {name_categories['english']}")
    print(f"   中文名称: {name_categories['chinese']}")

    # 分别生成英文和中文查询
    base_queries = []

    # 英文查询（用于国际平台和英文网站）
    for name in name_categories['english']:
        base_queries.extend([
            f'"{name}" {summary["research_majors_en"]} program curriculum',
            f'"{name}" design school faculty research',
            f'"{name}" student work portfolio graduation project',
            f'"{name}" alumni career employment outcome',
            f'"{name}" competition award prize winner',
            f'"{name}" lab studio facility equipment'
        ])

    # 中文查询（用于中文教育网站和平台）
    for name in name_categories['chinese']:
        base_queries.extend([
            f'"{name}" {summary["research_majors_cn"]} 专业 课程设置',
            f'"{name}" 设计学院 师资 研究方向',
            f'"{name}" 学生作品 毕业设计 作品集',
            f'"{name}" 校友 就业去向 职业发展',
            f'"{name}" 竞赛获奖 设计大赛 奖项',
            f'"{name}" 实验室 工作室 教学设施'
        ])

    # 添加更多细分查询
    specific_queries = []

    # 英文细分查询
    for name in name_categories['english']:
        specific_queries.extend([
            f'"{name}" "information design" OR "interaction design"',
            f'"{name}" "digital media" OR "new media art"',
            f'"{name}" "AI design" OR "artificial intelligence design"',
            f'"{name}" "employment report" OR "career outcomes"',
            f'"{name}" "course catalog" OR "curriculum guide"',
            f'"{name}" "student showcase" OR "degree show"',
            f'"{name}" "research publication" OR "faculty research"'
        ])
    
    # 中文细分查询
    for name in name_categories['chinese']:
        specific_queries.extend([
            f'"{name}" "信息设计" OR "交互设计"',
            f'"{name}" "数字媒体" OR "新媒体艺术"',
            f'"{name}" "AI设计" OR "人工智能设计"',
            f'"{name}" "就业报告" OR "就业质量报告"',
            f'"{name}" "课程目录" OR "培养方案"',
            f'"{name}" "学生展览" OR "毕业展"',
            f'"{name}" "学术论文" OR "科研成果"'
        ])

    # 添加官方域名优化查询
    if summary['official_domain']:
        domain_queries = [
            f'site:{summary["official_domain"]} {summary["research_majors_en"]}',
            f'site:{summary["official_domain"]} courses curriculum',
            f'site:{summary["official_domain"]} admissions employment',
            f'site:{summary["official_domain"]} research publications'
        ]
        base_queries.extend(domain_queries)

    # 添加新闻媒体和设计平台搜索
    media_queries = []

    # 优先使用英文名称进行媒体搜索
    primary_names = name_categories['english'][:2]  # 英文名称优先
    if not primary_names and name_categories['chinese']:  # 如果没有英文名称，使用中文
        primary_names = name_categories['chinese'][:1]

    for name in primary_names:
        media_queries.extend([
            f'"{name}" site:dezeen.com OR site:core77.com OR site:designboom.com',
            f'"{name}" site:behance.net OR site:dribbble.com',
            f'"{name}" site:linkedin.com "graduate" OR "alumni"',
            f'"{name}" site:medium.com OR site:blog.com',
            f'"{name}" site:youtube.com "program" OR "course"',
            f'"{name}" site:fastcompany.com OR site:wired.com OR site:techcrunch.com'
        ])
    
    tavily_queries = base_queries + specific_queries + media_queries
    summary["tavily_queries"] = tavily_queries
    
    # 学术搜索关键词 - 优先使用标准英文名
    scholar_q = f'"{summary["standard_name_en"]}" {summary["research_majors_en"]} academic papers'
    if ylo and yhi: scholar_q += f" {ylo}-{yhi}"
    summary["scholar_keyword"] = scholar_q
    summary["tavily_keyword"] = tavily_queries[0]
    return summary

# ==============================
# 智能数据处理和分类提取
# ==============================
def intelligent_data_processing(passages: List[str], summary: Dict) -> Dict[str, str]:
    """智能数据处理和分类提取"""

    # 按内容类型分类
    categorized_data = {
        "course_content": [],
        "research_content": [],
        "employment_content": [],
        "general_content": []
    }

    # 关键词分类
    course_keywords = ["course", "curriculum", "syllabus", "program", "study", "degree", "module"]
    research_keywords = ["research", "paper", "publication", "project", "study", "award", "competition"]
    employment_keywords = ["employment", "career", "job", "salary", "graduate", "alumni", "outcome"]

    for passage in passages:
        content_lower = passage.lower()

        # 计算相关性分数
        course_score = sum(1 for kw in course_keywords if kw in content_lower)
        research_score = sum(1 for kw in research_keywords if kw in content_lower)
        employment_score = sum(1 for kw in employment_keywords if kw in content_lower)

        # 分类到最相关的类别
        max_score = max(course_score, research_score, employment_score)
        if max_score == 0:
            if len(passage) > 500:  # 只保留较长的通用内容
                categorized_data["general_content"].append(passage[:3000])
        elif course_score == max_score:
            categorized_data["course_content"].append(passage[:4000])
        elif research_score == max_score:
            categorized_data["research_content"].append(passage[:4000])
        else:
            categorized_data["employment_content"].append(passage[:4000])

    # 为每个类别保留最相关的内容
    processed_data = {}
    for category, contents in categorized_data.items():
        if contents:
            # 按长度和相关性排序，保留最好的内容
            sorted_contents = sorted(contents, key=len, reverse=True)
            total_length = 0
            selected_contents = []

            for content in sorted_contents:
                if total_length + len(content) <= 30000:  # 每类别最多3万字符
                    selected_contents.append(content)
                    total_length += len(content)
                else:
                    break

            processed_data[category] = "\n\n".join(selected_contents)
        else:
            processed_data[category] = ""

    return processed_data

# ==============================
# 强化版检索链：大幅扩展抓取范围和深度
# ==============================
def create_data_retrieval_chain():
    t_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None
    serpapi_key = serpapi_api_key
    
    def run(vars):
        summary = vars["summary"]
        university = summary["target_university"]

        # 🆕 使用智能识别的标准化信息
        university_info = summary.get("university_info", {})
        standard_name_en = summary.get("standard_name_en", university)
        standard_name_cn = summary.get("standard_name_cn", "")
        university_domain = summary.get("official_domain", "")

        print(f"📋 使用标准化学校信息:")
        print(f"   原始输入: {university}")
        print(f"   标准英文名: {standard_name_en}")
        print(f"   标准中文名: {standard_name_cn}")
        print(f"   官方域名: {university_domain}")

        tavily_queries = summary.get("tavily_queries") or [summary.get("tavily_keyword", "")]
        all_urls, passages = [], []
        
        print("正在进行多源数据检索...")
        
        # 1) 详细课程信息抓取 - 使用标准化信息
        print("- 详细课程信息抓取")
        course_info = fetch_detailed_course_info(standard_name_en, university_domain,
                                                standard_name_cn, university)
        
        # 2) Tavily：合并多查询 URL，大幅增加结果数
        if t_client:
            print("- Tavily 官网检索")
            for i, q in enumerate(tavily_queries):
                try:
                    # 增加每次搜索的结果数
                    res = t_client.search(query=q, search_depth="advanced", include_answer=True, max_results=25)
                    for item in (res.get("results") or []):
                        url = item.get("url")
                        if url and url not in all_urls:
                            all_urls.append(url)
                    print(f"  查询 {i+1}/{len(tavily_queries)} 完成")
                except Exception as e:
                    passages.append(f"[Tavily 错误] {e}")
        
        # 2.1 添加更多专业设计网站和平台
        design_sites = [
            "behance.net", "dribbble.com", "core77.com", "designboom.com",
            "dezeen.com", "fastcompany.com", "wired.com", "techcrunch.com",
            "medium.com", "linkedin.com", "youtube.com", "vimeo.com"
        ]
        
        print("- 设计媒体平台检索")
        for site in design_sites:
            if t_client:
                site_query = f'"{summary["target_university"]}" site:{site} {summary["research_majors_en"]}'
                try:
                    res = t_client.search(query=site_query, search_depth="basic", max_results=15)
                    for item in (res.get("results") or []):
                        url = item.get("url")
                        if url and url not in all_urls:
                            all_urls.append(url)
                except Exception:
                    pass
        
        # 2.2 竞赛奖项定向检索
        print("- 竞赛奖项专项检索")
        award_keywords = [
            "award", "awards", "competition", "prize", "winners", "winning",
            "Red Dot", "iF Design", "IDEA", "D&AD", "AIGA", "ACM CHI", "IEEE", "UX awards",
            "获奖", "竞赛", "奖项", "大赛", "金奖", "银奖", "铜奖"
        ]
        if t_client:
            q_awards = f"{summary['target_university']} {summary['research_majors_en']} " + " ".join(award_keywords)
            try:
                res_award = t_client.search(query=q_awards, search_depth="advanced", include_answer=False, max_results=20)
                for item in (res_award.get("results") or []):
                    url = item.get("url")
                    if url and url not in all_urls:
                        all_urls.append(url)
            except Exception:
                pass
        
        # 2.3 补充学院常见目录直链
        print("- 学院目录直链补充")
        direct_paths = ["news", "events", "labs", "research", "projects", "studio", "curriculum", "program", "syllabus", "handbook", "courses", "admissions", "students", "faculty", "about"]
        domain_roots = list({re.sub(r"/+$", "", re.sub(r"^(https?://[^/]+).*$", r"\1", u)) for u in all_urls})
        for root in domain_roots[:12]:
            for p in direct_paths:
                candidate = f"{root}/{p}/"
                if candidate not in all_urls:
                    all_urls.append(candidate)
        
        print(f"收集到 {len(all_urls)} 个URL，开始并行抓取...")
        
        # 3) 大幅增加网页抓取数量，使用并行处理
        N = 80  # 从28增加到80
        pdf_candidates = []
        pdf_pref = ["employment", "graduate", "outcomes", "career", "就业", "毕业", "去向", "培养方案", "课程大纲", "handbook", "syllabus"]
        
        # 分离PDF和网页URL
        web_urls = []
        for url in all_urls:
            if url.lower().endswith(".pdf") or ".pdf" in url.lower():
                if any(k in url.lower() for k in pdf_pref):
                    pdf_candidates.insert(0, url)
                else:
                    pdf_candidates.append(url)
            else:
                web_urls.append(url)
        
        # 近五年年份列表
        yfrom = int(summary.get("year_from") or "2021")
        yto = int(summary.get("year_to") or "2025")
        years = list(range(min(yfrom, yto), max(yfrom, yto)+1))
        pre_emp: Dict[str, Dict[str, str]] = {}
        
        # 并行抓取网页内容
        print("- 并行抓取网页内容")
        page_results = parallel_fetch_pages(web_urls[:N])
        for url, text in page_results:
            # 增加内容截取长度，降低质量阈值
            passages.append(f"[Source] {url}\n{text[:12000]}")  # 从8000增加到12000
            # 预抽就业
            tmp = preextract_employment_stats(text, years)
            for y, kv in tmp.items():
                pre_emp.setdefault(y, {}).update({k:v for k,v in kv.items() if v})
        
        # 并行抓取PDF内容，增加PDF抓取数量
        print("- 并行抓取PDF内容")
        pdf_results = parallel_fetch_pdfs(pdf_candidates[:20])  # 从10增加到20
        for pdf_url, pdf_text in pdf_results:
            passages.append(f"[PDF] {pdf_url}\n{pdf_text[:12000]}")
            tmp = preextract_employment_stats(pdf_text, years)
            for y, kv in tmp.items():
                pre_emp.setdefault(y, {}).update({k:v for k,v in kv.items() if v})
        
        # 4) 添加社交媒体和新闻内容 - 使用标准化信息
        print("- 社交媒体和设计媒体内容抓取")
        social_content = fetch_social_media_content(
            standard_name_en,
            summary["research_majors_en"],
            standard_name_cn,
            university
        )
        passages.extend(social_content)
        
        # 5) Google Scholar：分页 + 年份范围
        print("- Google Scholar 学术检索")
        scholar_lines = []
        if serpapi_key:
            base_url = "https://serpapi.com/search.json"
            for start in (0, 10, 20, 30):  # 增加分页数
                try:
                    params = {"engine": "google_scholar","q": summary["scholar_keyword"],"api_key": serpapi_key,"num": 10,"start": start}
                    if summary.get("year_from"): params["as_ylo"] = summary["year_from"]
                    if summary.get("year_to"): params["as_yhi"] = summary["year_to"]
                    r = requests.get(base_url, params=params, timeout=30); r.raise_for_status()
                    data = r.json()
                    for it in data.get("organic_results", []):
                        title = it.get("title", "")
                        authors = (it.get("publication_info") or {}).get("summary", "")
                        link = it.get("link", "")
                        cited = (it.get("inline_links") or {}).get("cited_by", {}).get("total", "")
                        year = (it.get("publication_info") or {}).get("year", "")
                        scholar_lines.append(f"{title} | {authors} | {year} | Cited: {cited} | {link}")
                    print(f"  Scholar 分页 {start//10 + 1} 完成")
                except Exception as e:
                    scholar_lines.append(f"[Scholar 错误] {e}")
        
        # 6) 增强版学术搜索：OpenAlex / Crossref 多关键词组合 - 使用标准化信息
        print("- 增强版学术数据库检索")
        enhanced_academic_block = enhanced_academic_search(
            standard_name_en,
            summary["research_majors_en"],
            summary.get("year_from", ""),
            summary.get("year_to", ""),
            standard_name_cn,
            university
        )
        
        # 7) Wikidata 补充
        print("- Wikidata 组织信息补充")
        wd_block = wikidata_programs_and_units(summary["target_university"])
        
        # 8) 合并大语料 + 预抽就业
        sources_block = "\n".join([u for u in (web_urls[:N] + [pdf_url for pdf_url, _ in pdf_results])])
        scholar_block = "\n".join(scholar_lines)
        pre_emp_lines = []
        for y in sorted(pre_emp.keys()):
            emp = pre_emp[y]
            rate = emp.get("employment_rate", "")
            sal = emp.get("median_salary", "")
            if rate or sal:
                pre_emp_lines.append(f"{y}: employment_rate={rate or '-'}; median_salary={sal or '-'}")
        pre_emp_block = "\n".join(pre_emp_lines) if pre_emp_lines else "[无]"
        
        # 9) 包含课程信息
        course_block = ""
        if course_info["course_catalog"]:
            course_lines = []
            for course in course_info["course_catalog"][:20]:  # 限制显示数量
                course_lines.append(f"{course.get('code', '')} - {course.get('name', '')}")
            course_block = "\n".join(course_lines)
        else:
            course_block = "[课程信息未找到]"
        
        print(f"数据检索完成！共收集到 {len(passages)} 个内容片段")
        
        # 智能数据处理和分类
        print("- 智能数据处理和分类")
        processed_data = intelligent_data_processing(passages, summary)

        # 构建最终数据，避免长度限制问题
        merged = f"""
[智能分类处理数据]

[课程相关内容]
{processed_data.get('course_content', '暂无')}

[研究相关内容]
{processed_data.get('research_content', '暂无')}

[就业相关内容]
{processed_data.get('employment_content', '暂无')}

[其他相关内容]
{processed_data.get('general_content', '暂无')[:10000]}

[详细课程信息]
{course_block}

[Google Scholar 学术论文]
{scholar_block if scholar_block else '[无]'}

[增强版学术数据库检索]
{enhanced_academic_block}

[统计数据预抽取]
{pre_emp_block}
"""

        print(f"智能数据处理完成！总长度: {len(merged)} 字符")
        return merged
    
    return type("EnhancedRetrieval", (), {"run": lambda self, vars: run(vars)})()

# ==============================
# 结构化抽取链（细化活动/奖项/就业5年，加入类型枚举与强约束）
# ==============================
def create_structured_extraction_chain():
    # 使用原生字典定义，避免JSON字符串中的引号嵌套问题
    education_teaching_schema = {
        "major_setup": "专业设置及方向，如人机交互-智能界面设计，未公开则填暂未公开",
        "core_courses": "核心课程清单，完整课程名数组，未公开则填暂未公开",
        "innovative_teaching": {
            "type_enum": [
                "Workshop/工作坊", "Studio/工作室制", "Project-based/项目制", "Competition-led/竞赛驱动",
                "Interdisciplinary/跨学科", "Industry Collaboration/企业合作", "Service Learning/社区服务",
                "Capstone/毕业设计", "Field Trip/驻校驻厂实地课程", "Online/Hybrid/在线或混合式",
                "Seminar/系列讲座", "Peer Review/同伴评审"
            ],
            "items": [
                {"name": "活动课程名称", "type": "从type_enum中选择或填写", "year": "年份或学期",
                 "summary": "一句话简介50-120字", "link": "来源URL"}
            ],
            "note": "未公开则items为空数组"
        },
        "education_model": "教育模式数组，如Studio制导师制项目制Industry合作等，未公开则填暂未公开"
    }
    
    research_achievements_schema = {
        "competition_awards": [
            {"award_name": "奖项名", "level": "级别类别如国际国家校级金奖等", "year": "年份",
             "team_or_person": "获奖主体", "project": "作品论文项目名称", "link": "来源URL"}
        ],
        "academic_papers_latest": [
            {"title": "标题", "authors": "作者", "venue": "期刊会议", "year": "年份",
             "citations": "引用数", "doi_or_link": "DOI或链接"}
        ]
    }
    
    student_employment_schema = {
        "student_scale": [
            {"year": "年份", "level": "层次本科硕士博士", "count": "人数若未公开填暂未公开"}
        ],
        "education_model_detail": [
            {"model": "模式名称如Studio导师制Industry合作等", "description": "简述如何组织评价考核",
             "evidence": "来源要点或URL"}
        ],
        "employment_data_5y": [
            {"year": "年份", "employment_rate": "就业率百分比",
             "directions": ["主要去向及占比如互联网产品设计40%"], "median_salary": "若有",
             "notes": "其它指标口径说明", "source": "来源URL或正则预抽"}
        ]
    }
    
    response_schemas = [
        ResponseSchema(
            name="education_teaching",
            description=json.dumps(education_teaching_schema, ensure_ascii=False)
        ),
        ResponseSchema(
            name="research_achievements",
            description=json.dumps(research_achievements_schema, ensure_ascii=False)
        ),
        ResponseSchema(
            name="student_employment",
            description=json.dumps(student_employment_schema, ensure_ascii=False)
        ),
        ResponseSchema(
            name="data_source",
            description="数据来源列表数组，如大学官网课程页Graduate Outcomes 2022 PDF OpenAlex Crossref Google Scholar SerpAPI"
        )
    ]
    
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    extraction_prompt = PromptTemplate(
        input_variables=["retrieved_data", "summary", "format_instructions"],
        template=(
            "从以下大规模检索数据中，按要求提取结构化信息。注意：数据必须真实可追溯，缺失填暂未公开。\n"
            "=== 检索数据（包含大量页面正文、PDF、社交媒体、OpenAlex/Crossref/Scholar摘要、预抽就业）===\n{retrieved_data}\n"
            "=== 强约束 ===\n"
            "1. 仅与{summary[target_university]}、{summary[research_majors_cn]}相关；时间范围优先近五年（{summary[year_from]}–{summary[year_to]}），更早但重要的数据可标注年份。\n"
            "2. 创新教学活动：输出items列表（名称/类型/年份/简介/链接），类型从枚举里选择或填写，尽可能多列出5-12条；\n"
            "3. 竞赛奖项：尽可能多列举（名称/级别/年份/主体/项目/链接），每条提供链接，包括学生个人获奖和学校整体荣誉；\n"
            "4. 学术论文：从OpenAlex/Crossref/Scholar中选最新且已公开的8–15条，包含标题、作者、期刊会议、年份、引用数、DOI或链接；\n"
            "5. 学生培养与就业：\n"
            "   - 学生规模按本科硕士博士分年列人数（未知用暂未公开）；\n"
            "   - 教育模式给出模式名+执行方式评价要点（可引用证据要点URL）；\n"
            "   - 就业数据按近五年逐条列出，每条含就业率、主要去向含占比、薪资若有、来源URL；不得只给平均值；\n"
            "   - 可参考预抽近五年就业率薪资中的正则预抽结果，若与官方冲突以官方优先；\n"
            "6. 从社交媒体和设计平台内容中提取学生作品、就业去向、行业认可等信息；\n"
            "7. 未找到的字段必须填暂未公开，严禁编造。\n"
            "8. 输出格式：\n{format_instructions}\n"
            "仅返回JSON。"
        )
    )
    
    def run_with_retry(vars, max_retries=3):
        """带重试机制的数据抽取"""
        retrieved = vars["retrieved_data"]
        summary = vars["summary"]
        
        # 数据分块处理，避免单次请求过大
        max_length = 150000  # 减少到15万字符
        if len(retrieved) > max_length:
            print(f"数据量过大（{len(retrieved)}字符），截取到{max_length}字符")
            retrieved = retrieved[:max_length]
            
        for attempt in range(max_retries):
            try:
                print(f"尝试第 {attempt + 1} 次数据抽取...")
                
                prompt = extraction_prompt.format(
                    retrieved_data=retrieved,
                    summary=summary,
                    format_instructions=format_instructions
                )
                
                # 创建带超时的LLM
                from langchain_openai import ChatOpenAI
                extraction_llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    openai_api_key=openai_api_key,
                    temperature=0.2,
                    timeout=300,  # 5分钟超时
                    max_retries=2
                )
                
                resp = extraction_llm.invoke(prompt)
                result = output_parser.parse(resp.content)
                print("数据抽取成功！")
                return result
                
            except Exception as e:
                print(f"第 {attempt + 1} 次尝试失败: {str(e)}")
                if attempt < max_retries - 1:
                    print("等待10秒后重试...")
                    time.sleep(10)
                else:
                    print("所有重试都失败，返回简化版结果...")
                    # 返回一个基本的结构，避免程序崩溃
                    return {
                        "education_teaching": {
                            "major_setup": "数据抽取失败，请检查网络连接或稍后重试",
                            "core_courses": ["暂未公开"],
                            "innovative_teaching": {"items": []},
                            "education_model": ["暂未公开"]
                        },
                        "research_achievements": {
                            "competition_awards": [],
                            "academic_papers_latest": []
                        },
                        "student_employment": {
                            "student_scale": [],
                            "education_model_detail": [],
                            "employment_data_5y": []
                        },
                        "data_source": ["数据抽取失败"]
                    }
    
    def run(vars):
        return run_with_retry(vars)
    
    return type("EnhancedExtract", (), {"run": lambda self, vars: run(vars)})()

# ==============================
# 报告生成链（Markdown）
# ==============================
def create_markdown_generation_chain():
    markdown_prompt = PromptTemplate(
        input_variables=["structured_data", "summary", "current_time"],
        template=(
            "# {summary[target_university]} 信息艺术设计相关专业调研报告\n"
            "> 生成时间：{current_time}\n"
            "> 数据来源：多平台综合检索（官网、学术数据库、社交媒体、设计平台等）\n"
            "\n## 一、调研参数汇总\n"
            "| 维度 | 内容 |\n"
            "|---|---|\n"
            "| 目标高校 | {summary[target_university]} |\n"
            "| 专业范畴 | {summary[research_majors_cn]} |\n"
            "| 核心调研维度 | {summary[research_dimensions_cn]} |\n"
            "| 数据时间范围 | {summary[data_time_range]} |\n"
            "\n## 二、教育教学与学科特色\n"
            "基于structured_data.education_teaching，列出：\n"
            "- 专业设置及方向\n- 核心课程清单\n- 创新教学活动（按条：名称/类型/年份/简介/链接）\n- 教育模式（模式名与执行/评价要点）\n"
            "\n## 三、设计与学术研究成果（近年）\n"
            "竞赛奖项：按条列出（名称/级别/年份/主体/项目/链接）。\n"
            "学术论文（最新8–15条）：标题/作者/期刊或会议/年份/引用数/DOI或链接。\n"
            "\n## 四、学生培养与就业（近五年分年列出）\n"
            "- 学生规模：按本科/硕士/博士分年列人数。\n"
            "- 教育模式细节：模式名+执行方式/评价要点（附证据/URL）。\n"
            "- 就业数据（逐年）：就业率、主要去向含占比、薪资若有与来源。\n"
            "\n## 五、数据来源说明\n"
            "本报告数据来源于多个平台的综合检索，包括但不限于：\n"
            "- 官方网站和学术机构页面\n"
            "- 学术数据库（OpenAlex、Crossref、Google Scholar）\n"
            "- 社交媒体和专业平台（LinkedIn、Medium、Behance等）\n"
            "- 设计媒体和新闻网站（Core77、Dezeen、Fast Company等）\n"
            "\n提示：以下为结构化数据（JSON，仅供参考，不要原样输出）：\n"
            "```\n{structured_data}\n```\n"
        )
    )
    
    def run(vars):
        prompt = markdown_prompt.format(
            structured_data=json.dumps(vars["structured_data"], ensure_ascii=False, indent=2),
            summary=vars["summary"],
            current_time=vars["current_time"]
        )
        print("正在生成Markdown报告...")
        return llm.invoke(prompt).content
    
    return type("EnhancedReport", (), {"run": lambda self, vars: run(vars)})()

# ==============================
# 主流程
# ==============================
def main():
    if not openai_api_key:
        print("[警告] 未检测到 OPENAI_API_KEY，程序将报错。")
    if not tavily_api_key:
        print("[提示] 未检测到 TAVILY_API_KEY，官网检索将不可用。")
    if not serpapi_api_key:
        print("[提示] 未检测到 SERPAPI_API_KEY，Google Scholar 将不可用。")
    if not college_scorecard_api_key:
        print("[提示] 未检测到 COLLEGE_SCORECARD_API_KEY，美国高校官方指标将不可用。")
    
    user_input = get_user_input()
    summary = summarize_research_params(user_input)
    print(f"\n=== 参数汇总 ===\n{json.dumps(summary, ensure_ascii=False, indent=2)}\n")
    
    print("\n=== 步骤1/3：大规模多源数据检索 ===")
    print("（官网+学术数据库+社交媒体+设计平台+新闻媒体+就业预抽）")
    retrieval_chain = create_data_retrieval_chain()
    retrieved_data = retrieval_chain.run({"summary": summary})
    print(f"\n检索数据预览（前1000字）：\n{retrieved_data[:1000]}...\n")
    
    print("\n=== 步骤2/3：增强版数据结构化提取 ===")
    # 使用模块化结构化抽取链（方案D）
    extraction_chain = create_modular_structured_extraction_chain()
    structured_data = extraction_chain.run({
        "retrieved_data": retrieved_data,
        "summary": summary
    })
    
    # 数据验证和补充
    structured_data = verify_and_enrich_data(structured_data, summary["target_university"])
    
    if college_scorecard_api_key:
        print("- 补充美国高校官方数据")
        scorecard_json = college_scorecard_programs(summary["target_university"])
        structured_data["data_source"] = (structured_data.get("data_source") or []) + ["College Scorecard(US)"]
        structured_data["us_scorecard_raw"] = scorecard_json
    
    print(f"\n结构化数据：\n{json.dumps(structured_data, ensure_ascii=False, indent=2)}\n")
    
    print("\n=== 步骤3/3：生成增强版Markdown报告 ===")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    markdown_chain = create_markdown_generation_chain()
    markdown_report = markdown_chain.run({
        "structured_data": structured_data,
        "summary": summary,
        "current_time": current_time
    })
    
    safe_univ_name = summary["target_university"].replace("/", "-").replace("\\", "-").replace(":", "-")
    report_filename = f"{safe_univ_name}_信息艺术设计专业调研报告_增强版_{datetime.now().strftime('%Y%m%d%H%M%S')}.md"
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(markdown_report)
    
    print(f"\n=== 增强版调研完成 ===")
    print(f"1. 报告文件：{report_filename}")
    print(f"2. 数据源覆盖：官网、学术库、社交媒体、设计平台、新闻媒体等")
    print(f"3. 抓取规模：大幅扩展至80+网页、20+PDF、多平台社交媒体内容")
    print(f"4. 报告预览（前1200字）：\n{markdown_report[:1200]}...")

# ==============================
# 方案D：按维度分别处理 + 模块化LLM提取
# ==============================

def modular_data_processing_and_extraction(summary, retrieved_data):
    """按模块分别处理数据并进行LLM提取"""

    # 1. 解析retrieved_data中的各个模块数据
    modules_data = parse_modules_data(retrieved_data)

    # 2. 分别对每个模块进行LLM处理
    processed_modules = {}

    # 📝 添加模块数据统计
    print(f"📊 模块数据统计:")
    for module_name, data in modules_data.items():
        print(f"   {module_name}: {len(data)} 字符")

    # 处理课程模块
    if modules_data.get("course_data"):
        print("正在处理课程数据模块...")
        processed_modules["education_teaching"] = extract_module_data(
            modules_data["course_data"],
            "education_teaching",
            summary
        )

    # 处理研究模块
    if modules_data.get("research_data"):
        print("正在处理研究数据模块...")
        processed_modules["research_achievements"] = extract_module_data(
            modules_data["research_data"],
            "research_achievements",
            summary
        )

    # 处理就业模块
    if modules_data.get("employment_data"):
        print("正在处理就业数据模块...")
        processed_modules["student_employment"] = extract_module_data(
            modules_data["employment_data"],
            "student_employment",
            summary
        )

    # 3. 合并结果
    final_result = merge_module_results(processed_modules, summary)
    return final_result

def parse_modules_data(retrieved_data):
    """解析检索数据为不同模块"""
    modules = {
        "course_data": "",
        "research_data": "",
        "employment_data": "",
        "other_data": ""
    }

    # 按标记分割数据
    sections = retrieved_data.split("\n[")

    for section in sections:
        if section.startswith("详细课程信息]") or section.startswith("课程相关内容]") or "course" in section.lower():
            modules["course_data"] += "[" + section if not section.startswith("[") else section
            modules["course_data"] += "\n"
        elif (section.startswith("研究相关内容]") or section.startswith("Google Scholar") or
              section.startswith("增强版学术数据库") or "research" in section.lower() or
              "paper" in section.lower() or "academic" in section.lower() or
              "scholar" in section.lower() or "openalex" in section.lower() or
              "crossref" in section.lower()):
            modules["research_data"] += "[" + section if not section.startswith("[") else section
            modules["research_data"] += "\n"
        elif (section.startswith("就业相关内容]") or section.startswith("统计数据预抽取]") or
              "employment" in section.lower() or "salary" in section.lower() or
              "career" in section.lower()):
            modules["employment_data"] += "[" + section if not section.startswith("[") else section
            modules["employment_data"] += "\n"
        else:
            modules["other_data"] += "[" + section if not section.startswith("[") else section
            modules["other_data"] += "\n"

    return modules

def extract_module_data(module_data, module_type, summary):
    """对单个模块数据进行LLM提取"""
    # 限制数据长度，避免单个模块过大
    if len(module_data) > 100000:  # 10万字符限制
        print(f"模块数据过大({len(module_data)}字符)，截取到100000字符")
        module_data = module_data[:100000]

    if len(module_data) < 100:  # 如果数据太少，跳过
        print(f"{module_type}模块数据不足，跳过处理")
        return get_default_module_structure(module_type)

    # 创建对应模块的schema
    if module_type == "education_teaching":
        schema = ResponseSchema(
            name="education_teaching",
            description=json.dumps({
                "major_setup": "专业设置及方向，如人机交互-智能界面设计，未公开则填暂未公开",
                "core_courses": "核心课程清单，完整课程名数组，未公开则填暂未公开",
                "innovative_teaching": {
                    "type_enum": [
                        "Workshop/工作坊", "Studio/工作室制", "Project-based/项目制", "Competition-led/竞赛驱动",
                        "Interdisciplinary/跨学科", "Industry Collaboration/企业合作", "Service Learning/社区服务",
                        "Capstone/毕业设计", "Field Trip/驻校驻厂实地课程", "Online/Hybrid/在线或混合式",
                        "Seminar/系列讲座", "Peer Review/同伴评审"
                    ],
                    "items": [
                        {"name": "活动课程名称", "type": "从type_enum中选择或填写", "year": "年份或学期",
                         "summary": "一句话简介50-120字", "link": "来源URL"}
                    ],
                    "note": "未公开则items为空数组"
                },
                "education_model": "教育模式数组，如Studio制导师制项目制Industry合作等，未公开则填暂未公开"
            }, ensure_ascii=False)
        )
    elif module_type == "research_achievements":
        schema = ResponseSchema(
            name="research_achievements",
            description=json.dumps({
                "competition_awards": [
                    {"award_name": "奖项名", "level": "级别类别如国际国家校级金奖等", "year": "年份",
                     "team_or_person": "获奖主体", "project": "作品论文项目名称", "link": "来源URL"}
                ],
                "academic_papers_latest": [
                    {"title": "标题", "authors": "作者", "venue": "期刊会议", "year": "年份",
                     "citations": "引用数", "doi_or_link": "DOI或链接"}
                ]
            }, ensure_ascii=False)
        )
    elif module_type == "student_employment":
        schema = ResponseSchema(
            name="student_employment",
            description=json.dumps({
                "student_scale": [
                    {"year": "年份", "level": "层次本科硕士博士", "count": "人数若未公开填暂未公开"}
                ],
                "education_model_detail": [
                    {"model": "模式名称如Studio导师制Industry合作等", "description": "简述如何组织评价考核",
                     "evidence": "来源要点或URL"}
                ],
                "employment_data_5y": [
                    {"year": "年份", "employment_rate": "就业率百分比",
                     "directions": ["主要去向及占比如互联网产品设计40%"], "median_salary": "若有",
                     "notes": "其它指标口径说明", "source": "来源URL或正则预抽"}
                ]
            }, ensure_ascii=False)
        )
    else:
        return get_default_module_structure(module_type)

    # 创建输出解析器
    output_parser = StructuredOutputParser.from_response_schemas([schema])
    format_instructions = output_parser.get_format_instructions()

    # 创建提示模板
    prompt_template = PromptTemplate(
        input_variables=["retrieved_data", "summary", "format_instructions", "module_type"],
        template=(
            "从以下{module_type}模块数据中，按要求提取结构化信息。注意：数据必须真实可追溯，缺失填暂未公开。\n"
            "=== {module_type}模块检索数据 ===\n{retrieved_data}\n"
            "=== 强约束 ===\n"
            "1. 仅与{summary[target_university]}、{summary[research_majors_cn]}相关；时间范围优先近五年（{summary[year_from]}–{summary[year_to]}），更早但重要的数据可标注年份。\n"
            "2. 未找到的字段必须填暂未公开，严禁编造。\n"
            "3. 从提供的数据中尽可能多地提取相关信息。\n"
            "4. 输出格式：\n{format_instructions}\n"
            "仅返回JSON。"
        )
    )

    # 创建带超时的LLM
    from langchain_openai import ChatOpenAI
    extraction_llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=openai_api_key,
        temperature=0.2,
        timeout=300,  # 5分钟超时
        max_retries=2
    )

    # 生成提示
    prompt = prompt_template.format(
        retrieved_data=module_data,
        summary=summary,
        format_instructions=format_instructions,
        module_type=module_type
    )

    try:
        # 调用LLM
        resp = extraction_llm.invoke(prompt)
        result = output_parser.parse(resp.content)
        print(f"{module_type}模块处理成功！")
        return result.get(module_type, get_default_module_structure(module_type))
    except Exception as e:
        print(f"{module_type}模块处理失败: {str(e)}")
        # 返回默认结构
        return get_default_module_structure(module_type)

def get_default_module_structure(module_type):
    """获取模块默认结构"""
    if module_type == "education_teaching":
        return {
            "major_setup": "暂未公开",
            "core_courses": ["暂未公开"],
            "innovative_teaching": {"items": []},
            "education_model": ["暂未公开"]
        }
    elif module_type == "research_achievements":
        return {
            "competition_awards": [],
            "academic_papers_latest": []
        }
    elif module_type == "student_employment":
        return {
            "student_scale": [],
            "education_model_detail": [],
            "employment_data_5y": []
        }
    return {}

def merge_module_results(processed_modules, summary):
    """合并各模块处理结果"""
    merged_result = {
        "education_teaching": processed_modules.get("education_teaching", get_default_module_structure("education_teaching")),
        "research_achievements": processed_modules.get("research_achievements", get_default_module_structure("research_achievements")),
        "student_employment": processed_modules.get("student_employment", get_default_module_structure("student_employment")),
        "data_source": ["模块化智能提取"]
    }

    return merged_result

# 修改后的结构化抽取链
def create_modular_structured_extraction_chain():
    """创建模块化结构化抽取链"""
    def run(vars):
        retrieved_data = vars["retrieved_data"]
        summary = vars["summary"]

        print("开始模块化数据处理和智能提取...")
        structured_data = modular_data_processing_and_extraction(summary, retrieved_data)
        print("模块化数据处理完成！")

        return structured_data

    return type("ModularEnhancedExtract", (), {"run": lambda self, vars: run(vars)})()


if __name__ == "__main__":
    main()
