import asyncio
import logging
import time
import argparse
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Type

from github import Github
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

cache_memory: Dict[str, Any] = {}

def load_memory() -> Dict[str, Any]:
    logger.debug("Loading memory from cache")
    return cache_memory

def save_memory(memory: Dict[str, Any]) -> None:
    global cache_memory
    cache_memory = memory
    logger.debug("Memory saved to cache")

@dataclass
class ContextFile:
    name: str
    content: str

class ModelConfig(BaseModel):
    temperature: float = Field(0.0, ge=0.0, le=1.0)
    top_p: float = Field(0.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(0.0, ge=0.0, le=2.0)
    presence_penalty: float = Field(0.0, ge=0.0, le=2.0)
    parallel_tool_calls: bool = Field(True)
    max_tokens: int = Field(4000, gt=0)
    seed: int = Field(42, ge=0)

class GitHubRepoNavigator:
    def __init__(self, repo_full_name: str, access_token: Optional[str] = None):
        logger.info(f"Initializing GitHubRepoNavigator for repo: {repo_full_name}")
        self.github = Github(access_token) if access_token else None
        if not self.github:
            logger.error("GitHub access token is required")
            raise ValueError("GitHub access token is required")
        self.repo = self.github.get_repo(repo_full_name)
        self.file_index = self._build_file_index()
        logger.info(f"File index built with {len(self.file_index)} files")

    def _build_file_index(self) -> Dict[str, Dict[str, Any]]:
        logger.debug("Building file index from repo tree")
        tree = self.repo.get_git_tree(
            self.repo.get_branch(self.repo.default_branch).commit.sha,
            recursive=True
        ).tree
        index = {item.path: {"size": item.size, "sha": item.sha} for item in tree if item.type == 'blob'}
        logger.debug(f"File index contains {len(index)} blobs")
        return index

    def get_latest_changes(self, max_commits: int = 1) -> List[str]:
        logger.info(f"Fetching latest changes from last {max_commits} commits")
        commits = list(self.repo.get_commits())[:max_commits]
        files: Set[str] = set()
        for c in commits:
            for f in c.files:
                files.add(f.filename)
                print(f"modified file: {f.filename}")
        logger.info(f"Found {len(files)} changed files in latest commits")
        return list(files)

    def _parse_imports(self, content: str) -> List[str]:
        logger.debug("Parsing imports from file content")
        pattern = r"(?:from|import)\s+([\.\w_]+)"
        matches = re.findall(pattern, content)
        paths = []
        for m in matches:
            rel_ts = m.replace('.', '/') + '.ts'
            rel_py = m.replace('.', '/') + '.py'
            if rel_ts in self.file_index:
                paths.append(rel_ts)
            elif rel_py in self.file_index:
                paths.append(rel_py)
        logger.debug(f"Parsed {len(paths)} import dependencies")
        return paths

    def get_relevant_files(self, max_commits: int = 1, depth: int = 2) -> List[str]:
        logger.info(f"Getting relevant files with max_commits={max_commits} and depth={depth}")
        to_visit = set(self.get_latest_changes(max_commits))
        relevant: Set[str] = set()
        for d in range(depth + 1):
            logger.debug(f"Dependency resolution at depth {d}")
            new = set()
            for path in to_visit:
                if path in relevant:
                    continue
                relevant.add(path)
                content = self.get_file_content(path)
                deps = self._parse_imports(content)
                new.update(deps)
            to_visit = new
        logger.info(f"Total relevant files found: {len(relevant)}")
        return list(relevant)

    def get_file_content(self, file_path: str) -> str:
        logger.debug(f"Fetching content for file: {file_path}")
        if file_path not in self.file_index:
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        content = self.repo.get_contents(file_path).decoded_content.decode("utf-8", errors="ignore")
        logger.debug(f"Retrieved content for file: {file_path} (length: {len(content)})")
        return content
    
    def get_repo_summary(self) -> Dict[str, any]:
        """
        Return summary metadata useful for agents.
        """
        logger.info("Generating repository summary")
        summary = {
            "repo": self.repo.full_name,
            "description": self.repo.description,
            "file_count": len(self.file_index),
            "files": list(self.file_index.keys())[:20],
        }
        logger.debug(f"Repo summary: {summary}")
        return summary

    def create_issue(self, title: str, body: str, labels: Optional[List[str]] = None) -> None:
        logger.info(f"Creating GitHub issue: {title}")
        self.repo.create_issue(title=title, body=body, labels=labels or [])
        logger.info(f"Issue created: {title}")

class BaseAgentWrapper:
    def __init__(self, result_type: Type[BaseModel], system_prompt: str, model_name: str, api_key: str):
        logger.info(f"Initializing agent with model {model_name}")
        provider = OpenAIProvider(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        settings = ModelSettings()
        self.agent = Agent(
            model=OpenAIModel(model_name=model_name, provider=provider),
            output_type=result_type,
            system_prompt=system_prompt,
            retries=3,
            model_settings=settings,
        )
        self._register_tools()
        logger.info("Agent initialized and tools registered")

    def _register_tools(self) -> None:
        @self.agent.tool
        def get_repo_summary(ctx: RunContext) -> Dict[str, Any]:
            logger.info("Tool used: get_repo_summary")
            return ctx.deps.get_repo_summary()

        @self.agent.tool
        def get_latest_changes(ctx: RunContext) -> List[str]:
            logger.info(f"Tool used: get_latest_changes with max_commits={1}")
            return ctx.deps.get_latest_changes(2)

        @self.agent.tool
        def get_relevant_files(ctx: RunContext, depth: int = 2) -> List[str]:
            logger.info(f"Tool used: get_relevant_files with max_commits={1}, depth={depth}")
            return ctx.deps.get_relevant_files(2, depth)

        @self.agent.tool
        def get_file_content(ctx: RunContext, path: str) -> str:
            logger.info(f"Tool used: get_file_content with path={path}")
            return ctx.deps.get_file_content(path)

        @self.agent.tool
        def recall_memory(ctx: RunContext) -> Any:
            logger.info("Tool used: recall_memory")
            return load_memory()

        @self.agent.tool
        def store_memory(ctx: RunContext, memory: Dict[str, Any]) -> None:
            logger.info("Tool used: store_memory")
            save_memory(memory)
            
        @self.agent.tool
        def self_reflect(ctx: RunContext) -> str:
            logger.info("Tool used: self_reflect")
            mem = load_memory()
            return f"Reflection on summary: {mem}"
        
class QCAgent(BaseAgentWrapper):
    def __init__(self, result_type: Type[BaseModel], model_name: str, api_key: str):
        system_prompt = (
            "You are the ultimate Code Quality Control (QC) Agent."  
            " Fetch only relevant files based on latest commits and dependencies (depth 2)."  
            " Analyze code for structure, security, performance, and maintainability step-by-step."  
            " Store insights in memory. Return JSON: summary, status, metadata, code_issues, quality_issues, metrics."  
            "I have added a checklist file in the root directory named 'checklist'."
            "Use it to ensure all checks are performed."
        )
        logger.info("Initializing QCAgent")
        super().__init__(result_type, system_prompt, model_name, api_key)

    async def run(self, navigator: GitHubRepoNavigator) -> BaseModel:
        logger.info("Starting QC analysis run")
        mem = load_memory()
        files = navigator.get_relevant_files(max_commits=1, depth=2)
        logger.info(f"Files to analyze: {files}")
        start = time.perf_counter()
        result = await self.agent.run(
            f"Perform QC analysis on files: {files}",
            deps=navigator
        )
        exec_time = (time.perf_counter() - start) * 1000
        result.output.metadata.execution_time = exec_time
        logger.info(f"QC analysis completed in {exec_time:.2f} ms")
        save_memory({**mem, "last_summary": result.output.summary, "last_metrics": result.output.metrics.dict()})
        return result.output

class ReflectionAgent(BaseAgentWrapper):
    def __init__(self, result_type: Type[str], model_name: str, api_key: str):
        system_prompt = (
            "You are a self-reflection agent."  
            " Reflect on the provided QC summary and metrics."  
            " Suggest improvements and next steps."  
            " Return plain text reflection."  
        )
        logger.info("Initializing ReflectionAgent")
        super().__init__(result_type, system_prompt, model_name, api_key)

    async def run(self, navigator: GitHubRepoNavigator, summary: str, metrics: Dict[str, Any]) -> str:
        logger.info("Starting reflection run")
        prompt = f"Reflect on summary: {summary} with metrics: {metrics}"  
        reflection = await self.agent.run(prompt, deps=navigator)
        save_memory({**load_memory(), "reflection": reflection})
        logger.info("Reflection completed and saved to memory")
        return reflection.output

class CodeIssues(BaseModel): line: int; severity: str; message: str; error_code: str; fix: str; category: str
class QualityIssues(BaseModel): issue: str; severity: str; fix: str; code: str; impact: str; best_practice: str
class Metrics(BaseModel): complexity: int; maintainability: int; test_coverage: float; duplication: float
class Metadata(BaseModel): file: str; version: str; execution_time: Optional[float]
class Result(BaseModel): summary: str; status: str; metadata: Metadata; code_issues: List[CodeIssues]; quality_issues: List[QualityIssues]; metrics: Metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Run automated code quality control on a GitHub repository.")
    parser.add_argument("--repo", type=str, required=True, help="GitHub repository full name (e.g., 'owner/repo').")
    parser.add_argument("--token", type=str,required=True, help="GitHub access token.")
    parser.add_argument("--model", type=str, default="openai/gpt-4.1-mini", help="Model name to use for analysis.")
    parser.add_argument("--api_key", type=str, required=True, help="API key for the model provider.")
    return parser.parse_args()


async def main():
    args = parse_args()
    logger.info(f"Running QC on repository: {args.repo} with model: {args.model}")
    repo_full_name = args.repo
    access_token = args.token
    model_name = args.model
    api_key = args.api_key
    
    navigator = GitHubRepoNavigator(repo_full_name=repo_full_name, access_token=access_token)
    agent = QCAgent(result_type=Result, model_name=model_name, api_key=api_key)
    result = await agent.run(navigator)
    status = 'ERROR' if result.code_issues or result.quality_issues else 'SUCCESS'
    logger.info(f"QC Status: {status}")
    print(f"QC Status: {status}")
    if status == 'ERROR':
        markdown = [
            "**Status**: ERROR",
            "# Code Analysis Report",
            "## Status",
            f"- **Status**: `{status}`",
            f"- **Message**: {result.summary}",
            "## Metadata",
            f"- **File**: {result.metadata.file}",
            f"- **Version**: {result.metadata.version}",
            f"- **Execution Time**: {result.metadata.execution_time:.2f} ms",
            "## Code Issues",
            "| Line | Severity | Message | Error Code | Fix | Category |",
            "|------|----------|---------|------------|-----|----------|"
        ]
        for ci in result.code_issues:
            markdown.append(f"| {ci.line} | {ci.severity} | {ci.message} | {ci.error_code} | {ci.fix} | {ci.category} |")
        markdown.extend([
            "## Quality Issues",
            "| Issue | Severity | Fix | Code | Impact | Best Practice |",
            "|-------|----------|-----|------|---------|---------------|"
        ])
        for qi in result.quality_issues:
            markdown.append(f"| {qi.issue} | {qi.severity} | {qi.fix} | {qi.code} | {qi.impact} | {qi.best_practice} |")
        markdown.extend([
            "## Metrics",
            f"- **Complexity**: {result.metrics.complexity}",
            f"- **Maintainability**: {result.metrics.maintainability}",
            f"- **Test Coverage**: {result.metrics.test_coverage}",
            f"- **Duplication**: {result.metrics.duplication}"
        ])
        body = "\n".join(markdown)
        logger.info("Creating GitHub issue for QC report with errors")
        navigator.create_issue(title="Automated QC Report: Issues Found", body=body, labels=["quality", "automated-report"])

if __name__ == "__main__":
    asyncio.run(main())