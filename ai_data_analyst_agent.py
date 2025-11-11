"""
AI Data Analyst Agent
An intelligent agent that analyzes CSV data, generates visualizations,
builds dashboards, and creates comprehensive reports.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Visualization libraries
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# LangChain imports
# Using LangChain 0.1.20 compatible imports
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage

# AgentExecutor and create_openai_tools_agent - correct imports for LangChain 0.1.20
from langchain.agents import create_openai_tools_agent
from langchain.agents.agent import AgentExecutor

# Report generation
import markdown
from jinja2 import Template

# Set style
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")


class DataAnalystAgent:
    """AI-powered data analyst agent with tool capabilities"""
    
    def __init__(self, 
                 model_name: str = "gpt-4o-mini",
                 temperature: float = 0.1,
                 output_dir: str = "agent_outputs",
                 api_key: Optional[str] = None):
        """
        Initialize the data analyst agent
        
        Args:
            model_name: LLM model to use (gpt-4o-mini, gpt-4, claude-3-sonnet, etc.)
            temperature: Model temperature (lower = more deterministic)
            output_dir: Directory to save outputs
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        
        # Storage for current session
        self.current_df: Optional[pd.DataFrame] = None
        self.current_file_path: Optional[str] = None
        self.insights: List[Dict] = []
        self.visualizations: List[str] = []
        
        # Initialize tools
        self.tools = self._create_tools()
        
        # Create agent
        self.agent = self._create_agent()
        
    def _create_tools(self) -> List[Tool]:
        """Create tools for the agent"""
        
        def load_csv(file_path: str) -> str:
            """Load a CSV file and return basic information about it.
            
            Args:
                file_path: Path to the CSV file
                
            Returns:
                JSON string with file information
            """
            try:
                df = pd.read_csv(file_path)
                self.current_df = df
                self.current_file_path = file_path
                
                info = {
                    "status": "success",
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": df.columns.tolist(),
                    "data_types": df.dtypes.astype(str).to_dict(),
                    "missing_values": df.isnull().sum().to_dict(),
                    "sample_data": df.head(5).to_dict('records'),
                    "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2
                }
                return json.dumps(info, indent=2)
            except Exception as e:
                return f"Error loading CSV: {str(e)}"
        
        def analyze_data(analysis_type: str, columns: Optional[str] = None) -> str:
            """Perform data analysis on the loaded dataset.
            
            Args:
                analysis_type: Type of analysis (statistics, correlation, missing, 
                              distribution, outliers, trends)
                columns: Comma-separated column names (optional, uses all if not specified)
                
            Returns:
                JSON string with analysis results
            """
            if self.current_df is None:
                return "Error: No data loaded. Please load a CSV file first."
            
            try:
                df = self.current_df.copy()
                
                if columns:
                    cols = [c.strip() for c in columns.split(',')]
                    df = df[cols]
                
                results = {}
                
                if analysis_type == "statistics":
                    results = {
                        "descriptive_stats": df.describe().to_dict(),
                        "data_types": df.dtypes.astype(str).to_dict(),
                        "shape": df.shape
                    }
                
                elif analysis_type == "correlation":
                    numeric_df = df.select_dtypes(include=[np.number])
                    if len(numeric_df.columns) > 1:
                        corr = numeric_df.corr()
                        results = {
                            "correlation_matrix": corr.to_dict(),
                            "strong_correlations": self._find_strong_correlations(corr)
                        }
                    else:
                        results = {"message": "Need at least 2 numeric columns for correlation"}
                
                elif analysis_type == "missing":
                    results = {
                        "missing_counts": df.isnull().sum().to_dict(),
                        "missing_percentages": (df.isnull().sum() / len(df) * 100).to_dict(),
                        "columns_with_missing": df.columns[df.isnull().any()].tolist()
                    }
                
                elif analysis_type == "distribution":
                    numeric_df = df.select_dtypes(include=[np.number])
                    results = {}
                    for col in numeric_df.columns:
                        results[col] = {
                            "mean": float(numeric_df[col].mean()),
                            "median": float(numeric_df[col].median()),
                            "std": float(numeric_df[col].std()),
                            "min": float(numeric_df[col].min()),
                            "max": float(numeric_df[col].max()),
                            "skewness": float(numeric_df[col].skew()),
                            "kurtosis": float(numeric_df[col].kurtosis())
                        }
                
                elif analysis_type == "outliers":
                    numeric_df = df.select_dtypes(include=[np.number])
                    results = {}
                    for col in numeric_df.columns:
                        Q1 = numeric_df[col].quantile(0.25)
                        Q3 = numeric_df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                        results[col] = {
                            "outlier_count": len(outliers),
                            "outlier_percentage": len(outliers) / len(df) * 100,
                            "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)}
                        }
                
                # Store insight
                self.insights.append({
                    "type": analysis_type,
                    "timestamp": datetime.now().isoformat(),
                    "results": results
                })
                
                return json.dumps(results, indent=2, default=str)
                
            except Exception as e:
                return f"Error in analysis: {str(e)}"
        
        def create_visualization(chart_type: str, x_column: str, 
                               y_column: Optional[str] = None,
                               title: Optional[str] = None,
                               **kwargs) -> str:
            """Create and save a visualization.
            
            Args:
                chart_type: Type of chart (histogram, scatter, bar, line, box, heatmap, pairplot)
                x_column: Column name for x-axis
                y_column: Column name for y-axis (optional)
                title: Chart title (optional)
                **kwargs: Additional chart parameters
                
            Returns:
                Path to saved visualization
            """
            if self.current_df is None:
                return "Error: No data loaded. Please load a CSV file first."
            
            try:
                df = self.current_df.copy()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                fig_path = self.output_dir / f"viz_{chart_type}_{timestamp}.png"
                
                plt.figure(figsize=(12, 8))
                
                if chart_type == "histogram":
                    df[x_column].hist(bins=kwargs.get('bins', 30))
                    plt.xlabel(x_column)
                    plt.ylabel('Frequency')
                
                elif chart_type == "scatter":
                    if y_column:
                        plt.scatter(df[x_column], df[y_column], alpha=0.6)
                        plt.xlabel(x_column)
                        plt.ylabel(y_column)
                    else:
                        return "Error: scatter plot requires y_column"
                
                elif chart_type == "bar":
                    if y_column:
                        df.groupby(x_column)[y_column].mean().plot(kind='bar')
                    else:
                        df[x_column].value_counts().head(20).plot(kind='bar')
                    plt.xlabel(x_column)
                    plt.ylabel(y_column or 'Count')
                    plt.xticks(rotation=45, ha='right')
                
                elif chart_type == "line":
                    if y_column:
                        df.plot(x=x_column, y=y_column, kind='line')
                    else:
                        return "Error: line plot requires y_column"
                
                elif chart_type == "box":
                    if y_column:
                        df.boxplot(column=y_column, by=x_column)
                    else:
                        df[x_column].plot(kind='box')
                
                elif chart_type == "heatmap":
                    numeric_df = df.select_dtypes(include=[np.number])
                    if len(numeric_df.columns) > 1:
                        sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
                    else:
                        return "Error: heatmap requires multiple numeric columns"
                
                if title:
                    plt.title(title)
                else:
                    plt.title(f"{chart_type.title()} - {x_column}" + (f" vs {y_column}" if y_column else ""))
                
                plt.tight_layout()
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                self.visualizations.append(str(fig_path))
                return f"Visualization saved to: {fig_path}"
                
            except Exception as e:
                return f"Error creating visualization: {str(e)}"
        
        def create_plotly_viz(chart_type: str, x_column: str,
                             y_column: Optional[str] = None,
                             title: Optional[str] = None,
                             **kwargs) -> str:
            """Create an interactive Plotly visualization.
            
            Args:
                chart_type: Type of chart (scatter, bar, line, histogram, box, heatmap)
                x_column: Column name for x-axis
                y_column: Column name for y-axis (optional)
                title: Chart title (optional)
                
            Returns:
                Path to saved HTML file
            """
            if self.current_df is None:
                return "Error: No data loaded. Please load a CSV file first."
            
            try:
                df = self.current_df.copy()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                html_path = self.output_dir / f"plotly_{chart_type}_{timestamp}.html"
                
                if chart_type == "scatter":
                    fig = px.scatter(df, x=x_column, y=y_column, title=title)
                elif chart_type == "bar":
                    if y_column:
                        fig = px.bar(df, x=x_column, y=y_column, title=title)
                    else:
                        fig = px.bar(df[x_column].value_counts().head(20), title=title)
                elif chart_type == "line":
                    fig = px.line(df, x=x_column, y=y_column, title=title)
                elif chart_type == "histogram":
                    fig = px.histogram(df, x=x_column, title=title)
                elif chart_type == "box":
                    fig = px.box(df, x=x_column, y=y_column if y_column else x_column, title=title)
                elif chart_type == "heatmap":
                    numeric_df = df.select_dtypes(include=[np.number])
                    if len(numeric_df.columns) > 1:
                        corr = numeric_df.corr()
                        fig = px.imshow(corr, text_auto=True, aspect="auto", title=title)
                    else:
                        return "Error: heatmap requires multiple numeric columns"
                else:
                    return f"Error: Unsupported chart type: {chart_type}"
                
                fig.write_html(str(html_path))
                self.visualizations.append(str(html_path))
                return f"Interactive visualization saved to: {html_path}"
                
            except Exception as e:
                return f"Error creating Plotly visualization: {str(e)}"
        
        def build_dashboard(title: str = "Data Analysis Dashboard") -> str:
            """Build an interactive dashboard with all visualizations.
            
            Args:
                title: Dashboard title
                
            Returns:
                Path to dashboard HTML file
            """
            try:
                if not self.visualizations:
                    return "Error: No visualizations created yet"
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dashboard_path = self.output_dir / f"dashboard_{timestamp}.html"
                
                # Create simple HTML dashboard
                html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .viz-container {{ margin: 20px 0; }}
        iframe {{ width: 100%; height: 600px; border: none; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
"""
                
                # Add visualizations
                for viz_path in self.visualizations:
                    if viz_path.endswith('.html'):
                        html_content += f'<div class="viz-container"><iframe src="{viz_path}"></iframe></div>\n'
                    elif viz_path.endswith('.png'):
                        html_content += f'<div class="viz-container"><img src="{viz_path}" style="max-width:100%;"></div>\n'
                
                html_content += """
</body>
</html>
"""
                
                with open(dashboard_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                return f"Dashboard saved to: {dashboard_path}"
                
            except Exception as e:
                return f"Error building dashboard: {str(e)}"
        
        def generate_report(report_type: str = "markdown") -> str:
            """Generate a comprehensive analysis report.
            
            Args:
                report_type: Type of report (markdown, html)
                
            Returns:
                Path to generated report
            """
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if report_type == "markdown":
                    report_path = self.output_dir / f"report_{timestamp}.md"
                    
                    report_content = f"""# Data Analysis Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Data Source:** {self.current_file_path or "Unknown"}

## Executive Summary

This report contains comprehensive analysis of the provided dataset.

## Dataset Overview

"""
                    if self.current_df is not None:
                        report_content += f"""
- **Total Rows:** {len(self.current_df):,}
- **Total Columns:** {len(self.current_df.columns)}
- **Columns:** {', '.join(self.current_df.columns.tolist())}

"""
                    
                    report_content += "## Key Insights\n\n"
                    for i, insight in enumerate(self.insights, 1):
                        report_content += f"### Insight {i}: {insight['type'].title()}\n\n"
                        report_content += f"```json\n{json.dumps(insight['results'], indent=2, default=str)}\n```\n\n"
                    
                    report_content += "## Visualizations\n\n"
                    for viz_path in self.visualizations:
                        report_content += f"- [{Path(viz_path).name}]({viz_path})\n\n"
                    
                    report_content += "## Recommendations\n\n"
                    report_content += "Based on the analysis, consider the following:\n"
                    report_content += "- Review data quality issues if any\n"
                    report_content += "- Explore relationships between variables\n"
                    report_content += "- Consider additional data sources if needed\n"
                    
                    with open(report_path, 'w', encoding='utf-8') as f:
                        f.write(report_content)
                    
                    return f"Report saved to: {report_path}"
                
                else:
                    return f"Error: Unsupported report type: {report_type}"
                    
            except Exception as e:
                return f"Error generating report: {str(e)}"
        
        # Create tool instances
        tools = [
            Tool(
                name="load_csv",
                func=load_csv,
                description="Load a CSV file and get basic information about it. Use this first before any analysis."
            ),
            Tool(
                name="analyze_data",
                func=analyze_data,
                description="Perform various data analyses: statistics, correlation, missing values, distribution, outliers, trends. Returns JSON results."
            ),
            Tool(
                name="create_visualization",
                func=create_visualization,
                description="Create static visualizations (matplotlib/seaborn): histogram, scatter, bar, line, box, heatmap. Returns path to saved image."
            ),
            Tool(
                name="create_plotly_viz",
                func=create_plotly_viz,
                description="Create interactive Plotly visualizations: scatter, bar, line, histogram, box, heatmap. Returns path to saved HTML."
            ),
            Tool(
                name="build_dashboard",
                func=build_dashboard,
                description="Build an interactive HTML dashboard combining all created visualizations."
            ),
            Tool(
                name="generate_report",
                func=generate_report,
                description="Generate a comprehensive markdown report with insights, visualizations, and recommendations."
            )
        ]
        
        return tools
    
    def _find_strong_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict]:
        """Find strong correlations in correlation matrix"""
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                val = corr_matrix.iloc[i, j]
                if abs(val) >= threshold:
                    strong_corr.append({
                        "column1": corr_matrix.columns[i],
                        "column2": corr_matrix.columns[j],
                        "correlation": float(val)
                    })
        return strong_corr
    
    def _create_agent(self) -> AgentExecutor:
        """Create the agent executor"""
        
        system_prompt = """You are an expert data analyst and data scientist. Your role is to:
1. Analyze CSV data files thoroughly
2. Generate meaningful insights
3. Create appropriate visualizations
4. Build interactive dashboards when requested
5. Generate comprehensive reports

Always follow this workflow:
1. First, load the CSV file using load_csv
2. Understand the data structure
3. Perform relevant analyses based on the user's context
4. Create visualizations that best represent the data
5. Build dashboards if requested
6. Generate a final report

Be thorough, analytical, and provide actionable insights."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)
        
        return agent_executor
    
    def analyze(self, csv_path: str, context: str = "") -> Dict[str, Any]:
        """
        Main method to analyze a CSV file
        
        Args:
            csv_path: Path to CSV file
            context: Additional context about the data or analysis requirements
            
        Returns:
            Dictionary with analysis results
        """
        # Reset state
        self.insights = []
        self.visualizations = []
        
        # Construct prompt
        prompt = f"""Analyze the CSV file at: {csv_path}

Context: {context if context else "Perform a comprehensive data analysis including statistics, data quality checks, correlations, and create appropriate visualizations."}

Please:
1. Load the CSV file
2. Perform comprehensive analysis
3. Create relevant visualizations
4. Generate a report

Be thorough and provide actionable insights."""
        
        # Run agent
        result = self.agent.invoke({"input": prompt})
        
        return {
            "result": result,
            "insights": self.insights,
            "visualizations": self.visualizations,
            "output_dir": str(self.output_dir)
        }


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Data Analyst Agent")
    parser.add_argument("csv_file", help="Path to CSV file")
    parser.add_argument("--context", default="", help="Additional context about the data")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--output-dir", default="agent_outputs", help="Output directory")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = DataAnalystAgent(
        model_name=args.model,
        output_dir=args.output_dir,
        api_key=args.api_key
    )
    
    # Run analysis
    print(f"Analyzing {args.csv_file}...")
    results = agent.analyze(args.csv_file, args.context)
    
    print("\n" + "="*50)
    print("Analysis Complete!")
    print("="*50)
    print(f"Output directory: {results['output_dir']}")
    print(f"Visualizations created: {len(results['visualizations'])}")
    print(f"Insights generated: {len(results['insights'])}")
    print("\nCheck the output directory for results!")


if __name__ == "__main__":
    main()

