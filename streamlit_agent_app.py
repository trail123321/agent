"""
Streamlit Web App for AI Data Analyst Agent
Deploy this to Streamlit Cloud for easy access
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import os
from ai_data_analyst_agent import DataAnalystAgent

# Page config
st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Data Analyst Agent")
st.markdown("Upload a CSV file and get comprehensive data analysis, visualizations, and insights!")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Enter your OpenAI API key or set OPENAI_API_KEY environment variable"
    )
    
    model_name = st.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"],
        index=0,
        help="Choose the LLM model (gpt-4o-mini is cost-effective)"
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Lower = more deterministic, Higher = more creative"
    )

# Main area
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=['csv'],
    help="Upload your CSV file for analysis"
)

context = st.text_area(
    "Additional Context (Optional)",
    height=100,
    help="Provide any context about the data, specific questions, or analysis requirements",
    placeholder="Example: Focus on identifying trends in sales data and correlations between features..."
)

if st.button("🚀 Analyze Data", type="primary"):
    if uploaded_file is None:
        st.error("Please upload a CSV file first!")
    elif not api_key:
        st.error("Please enter your OpenAI API key in the sidebar!")
    else:
        with st.spinner("Analyzing data... This may take a few minutes."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w') as tmp_file:
                    # Read and save CSV
                    df = pd.read_csv(uploaded_file)
                    df.to_csv(tmp_file.name, index=False)
                    csv_path = tmp_file.name
                
                # Initialize agent
                agent = DataAnalystAgent(
                    model_name=model_name,
                    temperature=temperature,
                    output_dir="agent_outputs",
                    api_key=api_key
                )
                
                # Run analysis
                results = agent.analyze(csv_path, context)
                
                # Display results
                st.success("Analysis Complete! 🎉")
                
                # Show insights
                st.header("📈 Insights")
                if results['insights']:
                    for i, insight in enumerate(results['insights'], 1):
                        with st.expander(f"Insight {i}: {insight['type'].title()}"):
                            st.json(insight['results'])
                else:
                    st.info("No insights generated yet. Check the agent output above.")
                
                # Show visualizations
                st.header("📊 Visualizations")
                if results['visualizations']:
                    cols = st.columns(2)
                    for idx, viz_path in enumerate(results['visualizations']):
                        col = cols[idx % 2]
                        with col:
                            if viz_path.endswith('.png'):
                                st.image(viz_path, caption=Path(viz_path).name)
                            elif viz_path.endswith('.html'):
                                with open(viz_path, 'r', encoding='utf-8') as f:
                                    html_content = f.read()
                                st.components.v1.html(html_content, height=600)
                else:
                    st.info("No visualizations created yet.")
                
                # Show output directory
                st.header("📁 Output Files")
                st.info(f"All outputs saved to: `{results['output_dir']}`")
                
                # Download report if available
                report_files = list(Path(results['output_dir']).glob("report_*.md"))
                if report_files:
                    latest_report = max(report_files, key=lambda p: p.stat().st_mtime)
                    with open(latest_report, 'r', encoding='utf-8') as f:
                        st.download_button(
                            "📥 Download Report",
                            f.read(),
                            file_name=latest_report.name,
                            mime="text/markdown"
                        )
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.exception(e)
            finally:
                # Clean up temp file
                if 'csv_path' in locals():
                    try:
                        os.unlink(csv_path)
                    except:
                        pass

# Instructions
with st.expander("ℹ️ How to Use"):
    st.markdown("""
    1. **Upload CSV**: Click "Browse files" and select your CSV file
    2. **Add Context** (Optional): Provide any specific questions or analysis requirements
    3. **Configure**: Set your OpenAI API key and choose model in the sidebar
    4. **Analyze**: Click "Analyze Data" and wait for results
    5. **Review**: Check insights, visualizations, and download the report
    
    **Tips:**
    - Use `gpt-4o-mini` for cost-effective analysis
    - Use `gpt-4` for more complex analysis
    - Be specific in context for better results
    - Large files may take longer to process
    """)

# Footer
st.markdown("---")
st.markdown("Built with LangChain, OpenAI, and Streamlit")

