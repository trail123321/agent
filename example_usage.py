"""
Example usage of the AI Data Analyst Agent
"""

from ai_data_analyst_agent import DataAnalystAgent
import os

# Example 1: Basic usage
def example_basic():
    """Basic analysis example"""
    print("="*60)
    print("Example 1: Basic CSV Analysis")
    print("="*60)
    
    # Initialize agent
    agent = DataAnalystAgent(
        model_name="gpt-4o-mini",
        output_dir="example_outputs"
    )
    
    # Analyze a CSV file
    csv_path = "data/sample_data.csv"  # Replace with your CSV path
    
    if os.path.exists(csv_path):
        results = agent.analyze(
            csv_path=csv_path,
            context="Perform comprehensive analysis including statistics, correlations, and create visualizations"
        )
        
        print(f"\n✅ Analysis complete!")
        print(f"📊 Insights: {len(results['insights'])}")
        print(f"📈 Visualizations: {len(results['visualizations'])}")
        print(f"📁 Output directory: {results['output_dir']}")
    else:
        print(f"❌ CSV file not found: {csv_path}")
        print("Please provide a valid CSV file path")


# Example 2: With specific context
def example_with_context():
    """Analysis with specific context"""
    print("\n" + "="*60)
    print("Example 2: Analysis with Specific Context")
    print("="*60)
    
    agent = DataAnalystAgent(
        model_name="gpt-4o-mini",
        output_dir="example_outputs_context"
    )
    
    csv_path = "data/sales_data.csv"  # Replace with your CSV path
    
    context = """
    Focus on:
    1. Sales trends over time
    2. Correlation between marketing spend and sales
    3. Identify top performing products
    4. Create visualizations for key metrics
    5. Generate insights for business recommendations
    """
    
    if os.path.exists(csv_path):
        results = agent.analyze(csv_path, context)
        print(f"\n✅ Context-specific analysis complete!")
    else:
        print(f"❌ CSV file not found: {csv_path}")


# Example 3: Create sample data for testing
def create_sample_data():
    """Create a sample CSV for testing"""
    import pandas as pd
    import numpy as np
    
    print("\n" + "="*60)
    print("Creating Sample Data for Testing")
    print("="*60)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'date': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
        'sales': np.random.normal(1000, 200, n_samples),
        'marketing_spend': np.random.normal(500, 100, n_samples),
        'product_category': np.random.choice(['A', 'B', 'C'], n_samples),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'customer_satisfaction': np.random.uniform(1, 5, n_samples),
        'units_sold': np.random.poisson(50, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add some correlation
    df['sales'] = df['sales'] + df['marketing_spend'] * 0.5 + np.random.normal(0, 50, n_samples)
    
    # Save to CSV
    os.makedirs("data", exist_ok=True)
    output_path = "data/sample_data.csv"
    df.to_csv(output_path, index=False)
    
    print(f"✅ Sample data created: {output_path}")
    print(f"📊 Shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    
    return output_path


if __name__ == "__main__":
    # Create sample data first
    sample_csv = create_sample_data()
    
    # Run basic example
    example_basic()
    
    # Uncomment to run context example
    # example_with_context()
    
    print("\n" + "="*60)
    print("✨ Examples complete! Check the output directories for results.")
    print("="*60)
