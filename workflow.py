import os
import re
import json
import time
from datetime import datetime
from urllib.parse import quote
from typing import List, Dict, Tuple
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
# 网页正文与 PDF 抽取工具
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
# 并行抓取工具
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
        for future in concurrent.futures.as_completed(future_to_url):
            result = future.result()
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
        for future in concurrent.futures.as_completed(future_to_url):
            result = future.result()
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
# 社交媒体和新闻内容抓取
# ==============================
def fetch_social_media_content(university: str, majors: str) -> List[str]:
    """抓取社交媒体和新闻网站内容"""
    content = []
    
    # LinkedIn 搜索
    linkedin_queries = [
        f'site:linkedin.com "{university}" "graduate" "{majors}"',
        f'site:linkedin.com "{university}" "alumni" "design"',
        f'site:linkedin.com "{university}" "student" "portfolio"'
    ]
    
    # Medium 博客搜索
    medium_queries = [
        f'site:medium.com "{university}" "{majors}"',
        f'site:medium.com "{university}" "design program"',
        f'site:medium.com "{university}" "student experience"'
    ]
    
    # 设计媒体搜索
    design_media_queries = [
        f'site:core77.com "{university}"',
        f'site:designboom.com "{university}"',
        f'site:dezeen.com "{university}"',
        f'site:fastcompany.com "{university}" "design"',
        f'site:behance.net "{university}"',
        f'site:dribbble.com "{university}"'
    ]
    
    all_queries = linkedin_queries + medium_queries + design_media_queries
    
    if tavily_api_key:
        client = TavilyClient(api_key=tavily_api_key)
        for query in all_queries:
            try:
                res = client.search(query=query, max_results=10)
                for item in res.get("results", []):
                    url = item.get("url", "")
                    text = fetch_page_text(url)
                    if len(text) >= 300:
                        content.append(f"[Social/Media] {url}\n{text[:8000]}")
            except Exception:
                continue
    
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

def enhanced_academic_search(institution: str, majors: str, year_from: str, year_to: str) -> str:
    """增强版学术论文搜索"""
    all_papers = []
    
    # 多关键词组合搜索
    keywords_combinations = [
        f'"{institution}" "{majors}" design research',
        f'"{institution}" "human computer interaction"',
        f'"{institution}" "digital media art"',
        f'"{institution}" "information visualization"',
        f'"{institution}" "user experience design"',
        f'"{institution}" "artificial intelligence design"',
        f'"{institution}" "interactive design"',
        f'"{institution}" "new media art"'
    ]
    
    # OpenAlex 多查询
    for keywords in keywords_combinations:
        try:
            papers = openalex_recent_papers_latest(keywords, year_from, year_to, limit=20, take=5)
            if papers and "[OpenAlex(最新) 无结果]" not in papers:
                all_papers.append(papers)
        except Exception:
            continue
    
    # Crossref 多查询  
    for keywords in keywords_combinations:
        try:
            papers = crossref_recent_papers(keywords, year_from, year_to, rows=20)
            if papers and "[Crossref 无结果]" not in papers:
                all_papers.append(papers)
        except Exception:
            continue
    
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

    summary = {
        "target_university": user_input["target_university"],
        "research_majors_cn": ",".join(user_input["research_majors"]),
        "research_majors_en": " ".join([major_en_map[mj] for mj in user_input["research_majors"]]),
        "research_dimensions_cn": ",".join(user_input["research_dimensions"]),
        "research_dimensions_en": " ".join([dim_en_map[d] for d in user_input["research_dimensions"]]),
        "data_time_range": user_input["data_time_range"],
        "year_from": ylo, "year_to": yhi
    }

    # 大幅扩展查询多样性，移除域名限制
    base_queries = [
        f"{summary['target_university']} {summary['research_majors_en']} program curriculum",
        f"{summary['target_university']} {summary['research_majors_cn']} 专业 课程",
        f"{summary['target_university']} design school faculty research",
        f"{summary['target_university']} student work portfolio graduation project",
        f"{summary['target_university']} alumni career employment outcome",
        f"{summary['target_university']} competition award prize winner",
        f"{summary['target_university']} lab studio facility equipment"
    ]
    
    # 添加更多细分查询
    specific_queries = [
        f'"{summary["target_university"]}" "information design" OR "interaction design"',
        f'"{summary["target_university"]}" "digital media" OR "new media art"',
        f'"{summary["target_university"]}" "AI design" OR "artificial intelligence design"',
        f'"{summary["target_university"]}" "employment report" OR "career outcomes"',
        f'"{summary["target_university"]}" "course catalog" OR "curriculum guide"',
        f'"{summary["target_university"]}" "student showcase" OR "degree show"',
        f'"{summary["target_university"]}" "research publication" OR "faculty research"'
    ]
    
    # 添加新闻媒体和设计平台搜索
    media_queries = [
        f'"{summary["target_university"]}" site:dezeen.com OR site:core77.com OR site:designboom.com',
        f'"{summary["target_university"]}" site:behance.net OR site:dribbble.com',
        f'"{summary["target_university"]}" site:linkedin.com "graduate" OR "alumni"',
        f'"{summary["target_university"]}" site:medium.com OR site:blog.com',
        f'"{summary["target_university"]}" site:youtube.com "program" OR "course"',
        f'"{summary["target_university"]}" site:fastcompany.com OR site:wired.com OR site:techcrunch.com'
    ]
    
    tavily_queries = base_queries + specific_queries + media_queries
    summary["tavily_queries"] = tavily_queries

    scholar_q = f"{summary['target_university']} {summary['research_majors_en']} academic papers"
    if ylo and yhi: scholar_q += f" {ylo}-{yhi}"
    summary["scholar_keyword"] = scholar_q
    summary["tavily_keyword"] = tavily_queries[0]
    return summary

# ==============================
# 强化版检索链：大幅扩展抓取范围和深度
# ==============================
def create_data_retrieval_chain():
    t_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None
    serpapi_key = serpapi_api_key

    def run(vars):
        summary = vars["summary"]
        tavily_queries = summary.get("tavily_queries") or [summary.get("tavily_keyword", "")]
        all_urls, passages = [], []

        print("正在进行多源数据检索...")
        
        # 1) Tavily：合并多查询 URL，大幅增加结果数
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

        # 1.1 添加更多专业设计网站和平台
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

        # 1.2 竞赛奖项定向检索
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

        # 1.3 补充学院常见目录直链
        print("- 学院目录直链补充")
        direct_paths = ["news", "events", "labs", "research", "projects", "studio", "curriculum", "program", "syllabus", "handbook", "courses", "admissions", "students", "faculty", "about"]
        domain_roots = list({re.sub(r"/+$", "", re.sub(r"^(https?://[^/]+).*$", r"\1", u)) for u in all_urls})
        for root in domain_roots[:12]:
            for p in direct_paths:
                candidate = f"{root}/{p}/"
                if candidate not in all_urls:
                    all_urls.append(candidate)

        print(f"收集到 {len(all_urls)} 个URL，开始并行抓取...")

        # 2) 大幅增加网页抓取数量，使用并行处理
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

        # 3) 添加社交媒体和新闻内容
        print("- 社交媒体和设计媒体内容抓取")
        social_content = fetch_social_media_content(
            summary["target_university"], 
            summary["research_majors_en"]
        )
        passages.extend(social_content)

        # 4) Google Scholar：分页 + 年份范围
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

        # 5) 增强版学术搜索：OpenAlex / Crossref 多关键词组合
        print("- 增强版学术数据库检索") 
        enhanced_academic_block = enhanced_academic_search(
            summary["target_university"], 
            summary["research_majors_en"],
            summary.get("year_from", ""), 
            summary.get("year_to", "")
        )
        
        # 6) Wikidata 补充
        print("- Wikidata 组织信息补充")
        wd_block = wikidata_programs_and_units(summary["target_university"])

        # 7) 合并大语料 + 预抽就业
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

        print(f"数据检索完成！共收集到 {len(passages)} 个内容片段")

        merged = (
            f"[页面正文汇总（扩展到{len(page_results)}条网页 + {len(pdf_results)}条PDF + {len(social_content)}条社交媒体）]\n" + 
            ("\n\n".join(passages) if passages else "[无]") +
            f"\n\n[页面链接列表]\n{sources_block if sources_block else '[无]'}" +
            f"\n\n[Google Scholar 汇总（多分页）]\n{scholar_block if scholar_block else '[无]'}" +
            f"\n\n[增强版学术数据库检索（OpenAlex + Crossref 多关键词）]\n{enhanced_academic_block}" +
            f"\n\n[Wikidata 组织/院系]\n{wd_block}" +
            f"\n\n[预抽：近五年就业率/薪资（来自网页/PDF正则）]\n{pre_emp_block}" +
            f"\n\n[College Scorecard（如为美国高校，将在抽取后追加原始 JSON）]\n"
        )
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
    extraction_chain = create_structured_extraction_chain()
    structured_data = extraction_chain.run({
        "retrieved_data": retrieved_data,
        "summary": summary
    })

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

if __name__ == "__main__":
    main()