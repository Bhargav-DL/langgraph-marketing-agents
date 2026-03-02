from typing import Optional
from langchain_community.tools import DuckDuckGoSearchRun
from src.utils import create_call_qwen  # but we'll receive call_qwen as argument

# We'll define nodes as functions that take call_qwen as a closure or parameter.
# To avoid global, we'll define a class or use partial. For simplicity, we'll
# define the nodes inside a builder that receives call_qwen. But here we'll
# just write them assuming call_qwen is a global (set after loading).
# However, to keep it clean, we'll define them as functions that accept call_qwen.
# In graph_builder.py we'll create a closure by passing the call_qwen function.

from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional

class MarketingState(TypedDict):
    topic: str
    search_results: Optional[str] = None
    research: Optional[str] = None
    strategy: Optional[str] = None
    draft_content: Optional[str] = None
    final_content: Optional[str] = None

def create_search_node():
    search_tool = DuckDuckGoSearchRun()
    def search_node(state: MarketingState) -> dict:
        print(f"\n>>> [SEARCH] Searching for: {state['topic']}")
        try:
            queries = [
                f"{state['topic']} latest trends 2026",
                f"{state['topic']} market analysis",
                f"{state['topic']} competitor strategies"
            ]
            results = []
            for q in queries:
                res = search_tool.run(q)
                results.append(f"## Query: {q}\n{res}\n")
            combined = "\n".join(results)
            return {"search_results": combined}
        except Exception as e:
            return {"search_results": f"Search failed: {str(e)}"}
    return search_node

def create_researcher_node(call_qwen):
    def researcher_node(state: MarketingState) -> dict:
        print("\n>>> [RESEARCHER] Starting.")
        prompt = f"""You are a Market Research Analyst. Use both your knowledge and the search results below to conduct deep research on '{state['topic']}'.

WEB SEARCH RESULTS:
{state.get('search_results', 'No search results available')}

Based on this information, provide a comprehensive market research report including:
1. Current trends and viral topics
2. Competitor content analysis
3. Audience pain points and desires
4. 5-10 relevant keywords and hashtags
5. Any surprising insights

Format your response with clear bullet points and sections."""
        response = call_qwen(prompt)
        return {"research": response}
    return researcher_node

def create_strategist_node(call_qwen):
    def strategist_node(state: MarketingState) -> dict:
        print("\n>>> [STRATEGIST] Starting.")
        prompt = f"""You are a Content Strategist. Based on this research:
{state['research']}

Create a content strategy for '{state['topic']}'. Include:
1. Recommended content angles (3-5)
2. Target platform suggestions (LinkedIn, Twitter, blog, etc.)
3. Tone of voice guidelines
4. Hook ideas and CTAs"""
        response = call_qwen(prompt)
        return {"strategy": response}
    return strategist_node

def create_copywriter_node(call_qwen):
    from config import settings
    def copywriter_node(state: MarketingState) -> dict:
        print("\n>>> [COPYWRITER] Starting.")
        prompt = f"""You are a Creative Copywriter. Using this strategy:
{state['strategy']}

Write three content pieces for '{state['topic']}':
1. One LinkedIn thought-leadership post (300-400 words)
2. One Twitter/X thread (5-7 tweets)
3. One ad headline + description variation (3 options)"""
        # Use a smaller max_tokens for copywriter to speed up
        response = call_qwen(prompt, max_tokens=settings.COPYWRITER_MAX_TOKENS)
        return {"draft_content": response}
    return copywriter_node

def create_editor_node(call_qwen):
    def editor_node(state: MarketingState) -> dict:
        print("\n>>> [EDITOR] Starting.")
        prompt = f"""You are an SEO Editor. Optimize this draft content:
{state['draft_content']}

Ensure keywords are naturally integrated, improve readability, add hashtags where appropriate, and polish for brand consistency. Return the final, polished content ready for publishing."""
        response = call_qwen(prompt)
        return {"final_content": response}
    return editor_node