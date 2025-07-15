import pandas as pd
from transformers import pipeline
import re

# Load the data
df = pd.read_csv('sales_data.csv')

# =============================================================================
# TRADITIONAL PROGRAMMING APPROACH - Rule-based processing
# =============================================================================

def process_traditional_data(df):
    """Process data using traditional programming methods"""

    # 1. Calculate sales metrics
    df['revenue_category'] = df['purchase_amount'].apply(lambda x:
        'High' if x > 50000 else 'Medium' if x > 10000 else 'Low')

    # 2. Standardize payment methods
    payment_mapping = {
        'Credit Card': 'Card',
        'Bank Transfer': 'Transfer',
        'Wire Transfer': 'Transfer',
        'PayPal': 'Digital'
    }
    df['payment_type'] = df['payment_method'].map(payment_mapping)

    # 3. Extract state from address
    df['state'] = df['delivery_address_area'].str.extract(r'([A-Z]{2})$')

    # 4. Categorize customer segments
    df['segment_priority'] = df['customer_segment'].map({
        'Enterprise': 1,
        'SMB': 2,
        'Individual': 3
    })

    return df

# =============================================================================
# AI/LLM APPROACH - Using Transformers for text processing
# =============================================================================

def process_ai_data(df):
    """Process text data using AI/LLM"""

    # Initialize sentiment analysis pipeline
    sentiment_analyzer = pipeline("sentiment-analysis",
                                model="cardiffnlp/twitter-roberta-base-sentiment-latest")

    # Initialize text classification pipeline
    classifier = pipeline("zero-shot-classification",
                         model="facebook/bart-large-mnli")

    # 1. Sentiment analysis on customer feedback
    def analyze_sentiment(text):
        try:
            result = sentiment_analyzer(text)[0]
            return result['label'], round(result['score'], 2)
        except:
            return 'NEUTRAL', 0.5

    df[['sentiment', 'sentiment_score']] = df['customer_feedback'].apply(
        lambda x: pd.Series(analyze_sentiment(x))
    )

    # 2. Classify sales notes into categories
    note_categories = ['technical_issue', 'pricing_negotiation', 'customer_relationship', 'product_feedback']

    def classify_note(text):
        try:
            result = classifier(text, note_categories)
            return result['labels'][0], round(result['scores'][0], 2)
        except:
            return 'general', 0.5

    df[['note_category', 'note_confidence']] = df['sales_note'].apply(
        lambda x: pd.Series(classify_note(x))
    )

    # 3. Extract key issues from feedback
    def extract_issues(text):
        try:
            issue_keywords = ['delay', 'problem', 'difficult', 'improve', 'issue', 'slow', 'complicated']
            issues = [word for word in issue_keywords if word in text.lower()]
            return ', '.join(issues) if issues else 'none'
        except:
            return 'none'

    df['identified_issues'] = df['customer_feedback'].apply(extract_issues)

    return df

# =============================================================================
# PRIVACY PROTECTION - Data anonymization
# =============================================================================

def anonymize_data(df):
    """Apply privacy protection measures"""

    # 1. Mask specific locations (keep only state)
    df['delivery_area_masked'] = df['delivery_address_area'].str.extract(r'([A-Z]{2})$')

    # 2. Remove sensitive information from notes
    def clean_sensitive_info(text):
        # Remove specific company names, numbers, and personal details
        text = re.sub(r'\b[A-Z][a-z]+ \d+\b', '[COMPANY_ID]', text)
        text = re.sub(r'\$\d+', '[AMOUNT]', text)
        text = re.sub(r'\d{2,}%', '[PERCENTAGE]', text)
        return text

    df['sales_note_cleaned'] = df['sales_note'].apply(clean_sensitive_info)

    # 3. Generalize customer feedback
    def generalize_feedback(text):
        # Remove specific product names and replace with generic terms
        text = re.sub(r'\b[A-Z][a-z]+\s+(team|service|support)\b', '[DEPARTMENT]', text)
        return text

    df['feedback_generalized'] = df['customer_feedback'].apply(generalize_feedback)

    return df

# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def main():
    # Load and process data
    print("Loading sales data...")
    df = pd.read_csv('sales_data.csv')

    print("Processing with traditional methods...")
    df = process_traditional_data(df)

    print("Processing with AI/LLM...")
    df = process_ai_data(df)

    print("Applying privacy protection...")
    df = anonymize_data(df)

    # Save processed data
    df.to_csv('processed_sales_data.csv', index=False)

    # Display summary
    print("\n=== PROCESSING SUMMARY ===")
    print(f"Total orders processed: {len(df)}")
    print(f"Revenue categories: {df['revenue_category'].value_counts().to_dict()}")
    print(f"Sentiment distribution: {df['sentiment'].value_counts().to_dict()}")
    print(f"Note categories: {df['note_category'].value_counts().to_dict()}")

    # Show sample results
    print("\n=== SAMPLE RESULTS ===")
    sample_cols = ['order_id', 'revenue_category', 'sentiment', 'note_category', 'identified_issues']
    print(df[sample_cols].head())

if __name__ == "__main__":
    main()
