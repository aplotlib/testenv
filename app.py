"""
Amazon Return Analysis Tool - Quality Management Edition
"""

import streamlit as st

# THIS MUST BE FIRST - NO EXCEPTIONS
st.set_page_config(
    page_title="Vive Health Return Analysis Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Now safe to import everything else
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import io
import re
from collections import defaultdict
from io import BytesIO
import csv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
RETURN_CATEGORIES = {
    'SIZE_FIT_ISSUES': {
        'keywords': ['too small', 'too large', 'doesnt fit', "doesn't fit", 'wrong size', 'size', 'fit', 'tight', 'loose', 'big', 'little'],
        'color': '#FF6B35',
        'icon': '📏'
    },
    'QUALITY_DEFECTS': {
        'keywords': ['defective', 'broken', 'damaged', 'doesnt work', "doesn't work", 'poor quality', 'defect', 'malfunction', 'faulty', 'dead', 'not working'],
        'color': '#FF0054',
        'icon': '⚠️'
    },
    'WRONG_PRODUCT': {
        'keywords': ['wrong item', 'not as described', 'inaccurate', 'different', 'not what', 'incorrect', 'mislabeled'],
        'color': '#FF006E',
        'icon': '📦'
    },
    'BUYER_MISTAKE': {
        'keywords': ['bought by mistake', 'accidentally', 'wrong order', 'my mistake', 'ordered wrong', 'accident'],
        'color': '#666680',
        'icon': '🤷'
    },
    'NO_LONGER_NEEDED': {
        'keywords': ['no longer needed', 'changed mind', 'dont need', "don't need", 'not needed', 'patient died', 'cancelled'],
        'color': '#666680',
        'icon': '❌'
    },
    'FUNCTIONALITY_ISSUES': {
        'keywords': ['not comfortable', 'hard to use', 'unstable', 'difficult', 'uncomfortable', 'awkward', 'complicated'],
        'color': '#FFB700',
        'icon': '🔧'
    },
    'COMPATIBILITY_ISSUES': {
        'keywords': ['doesnt fit toilet', "doesn't fit", 'not compatible', 'incompatible', 'wont fit', "won't fit", 'wrong type'],
        'color': '#00D9FF',
        'icon': '🔌'
    },
    'UNCATEGORIZED': {
        'keywords': [],
        'color': '#1A1A2E',
        'icon': '❓'
    }
}

def parse_fba_return_report(file_content: str):
    """Parse FBA Return Report TSV file"""
    try:
        # Parse TSV
        df = pd.read_csv(io.StringIO(file_content), sep='\t', encoding='utf-8')
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Convert to standard format
        returns_data = []
        for _, row in df.iterrows():
            return_data = {
                'order_id': str(row.get('order-id', '')).strip(),
                'asin': str(row.get('asin', '')).strip(),
                'sku': str(row.get('sku', '')).strip(),
                'product_name': str(row.get('product-name', '')).strip(),
                'return_reason': str(row.get('reason', '')).strip(),
                'buyer_comment': str(row.get('customer-comments', '')).strip() if pd.notna(row.get('customer-comments')) else '',
                'return_date': str(row.get('return-date', '')).strip(),
                'quantity': int(row.get('quantity', 1)) if pd.notna(row.get('quantity')) else 1
            }
            returns_data.append(return_data)
        
        return {
            'success': True,
            'returns': returns_data,
            'total_count': len(returns_data)
        }
        
    except Exception as e:
        logger.error(f"FBA report parsing error: {e}")
        return {
            'success': False,
            'returns': [],
            'total_count': 0,
            'error': str(e)
        }

def categorize_return(return_data):
    """Categorize a return based on reason and comment"""
    reason = str(return_data.get('return_reason', '')).lower()
    comment = str(return_data.get('buyer_comment', '')).lower()
    combined_text = f"{reason} {comment}"
    
    # Keyword matching
    for category, info in RETURN_CATEGORIES.items():
        if category == 'UNCATEGORIZED':
            continue
        for keyword in info['keywords']:
            if keyword.lower() in combined_text:
                return category
    
    return 'UNCATEGORIZED'

def process_returns_data(returns):
    """Process and categorize all returns"""
    categorized = defaultdict(list)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, return_data in enumerate(returns):
        status_text.text(f"Categorizing return {i+1} of {len(returns)}...")
        progress_bar.progress((i + 1) / len(returns))
        
        category = categorize_return(return_data)
        return_data['category'] = category
        categorized[category].append(return_data)
    
    progress_bar.empty()
    status_text.empty()
    
    # Calculate statistics
    stats = {}
    for category, returns_list in categorized.items():
        stats[category] = {
            'count': len(returns_list),
            'percentage': (len(returns_list) / len(returns)) * 100 if returns else 0,
            'returns': returns_list
        }
    
    return {
        'categorized': dict(categorized),
        'stats': stats,
        'total_returns': len(returns)
    }

def generate_excel_report(results):
    """Generate Excel report"""
    try:
        import xlsxwriter
        buffer = BytesIO()
        
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # Summary sheet
            summary_data = []
            for category, stats in sorted(results['stats'].items(), key=lambda x: x[1]['count'], reverse=True):
                if stats['count'] > 0:
                    summary_data.append({
                        'Category': category.replace('_', ' ').title(),
                        'Count': stats['count'],
                        'Percentage': f"{stats['percentage']:.1f}%"
                    })
            
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # All Returns sheet
            all_returns_data = []
            for category, returns_list in results['categorized'].items():
                for ret in returns_list:
                    all_returns_data.append({
                        'Order ID': ret.get('order_id', ''),
                        'ASIN': ret.get('asin', ''),
                        'SKU': ret.get('sku', ''),
                        'Product Name': ret.get('product_name', ''),
                        'Category': category.replace('_', ' ').title(),
                        'Return Reason': ret.get('return_reason', ''),
                        'Customer Comment': ret.get('buyer_comment', ''),
                        'Date': ret.get('return_date', ''),
                        'Quantity': ret.get('quantity', 1)
                    })
            
            pd.DataFrame(all_returns_data).to_excel(writer, sheet_name='All Returns', index=False)
            
            # By Product sheet
            product_returns = defaultdict(lambda: defaultdict(int))
            asin_details = {}
            
            for category, returns_list in results['categorized'].items():
                for ret in returns_list:
                    asin = ret.get('asin', 'Unknown')
                    if asin not in asin_details:
                        asin_details[asin] = {
                            'product_name': ret.get('product_name', 'Unknown'),
                            'sku': ret.get('sku', '')
                        }
                    product_returns[asin]['total'] += 1
                    product_returns[asin][category] += 1
            
            product_data = []
            for asin, counts in product_returns.items():
                row = {
                    'ASIN': asin,
                    'SKU': asin_details.get(asin, {}).get('sku', ''),
                    'Product Name': asin_details.get(asin, {}).get('product_name', 'Unknown'),
                    'Total Returns': counts['total']
                }
                
                # Add quality defect info
                quality_count = counts.get('QUALITY_DEFECTS', 0)
                quality_rate = (quality_count / counts['total'] * 100) if counts['total'] > 0 else 0
                row['Quality Defects'] = quality_count
                row['Quality Rate'] = f"{quality_rate:.1f}%"
                row['Action'] = 'YES' if quality_rate > 20 else 'Monitor' if quality_rate > 10 else 'OK'
                
                product_data.append(row)
            
            pd.DataFrame(product_data).to_excel(writer, sheet_name='By Product', index=False)
        
        buffer.seek(0)
        return buffer
    except ImportError:
        return None

def generate_csv_report(results):
    """Generate CSV report"""
    rows = []
    
    rows.append(['Return Analysis Report', datetime.now().strftime('%Y-%m-%d %H:%M')])
    rows.append([])
    rows.append(['Category', 'Count', 'Percentage'])
    
    for category, stats in results['stats'].items():
        if stats['count'] > 0:
            rows.append([
                category.replace('_', ' ').title(),
                stats['count'],
                f"{stats['percentage']:.1f}%"
            ])
    
    rows.append([])
    rows.append(['Order ID', 'ASIN', 'SKU', 'Product', 'Category', 'Return Reason', 'Customer Comment'])
    
    for category, returns_list in results['categorized'].items():
        for ret in returns_list:
            rows.append([
                ret.get('order_id', ''),
                ret.get('asin', ''),
                ret.get('sku', ''),
                ret.get('product_name', '')[:100],
                category.replace('_', ' ').title(),
                ret.get('return_reason', ''),
                ret.get('buyer_comment', '')
            ])
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(rows)
    
    return output.getvalue()

def main():
    st.title("🔍 Vive Health Return Analysis Tool")
    st.markdown("### Quality Management Return Categorization System")
    
    # Initialize session state
    if 'return_data' not in st.session_state:
        st.session_state.return_data = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    # File upload
    st.markdown("---")
    st.markdown("### 📤 Upload Return Data")
    
    uploaded_file = st.file_uploader(
        "Upload FBA Return Report (.txt TSV format)",
        type=['txt', 'tsv', 'csv'],
        help="Export from Seller Central > Reports > Fulfillment > FBA Returns"
    )
    
    if uploaded_file:
        content = uploaded_file.read().decode('utf-8')
        with st.spinner("🔍 Parsing FBA report..."):
            result = parse_fba_return_report(content)
            
            if result['success']:
                st.success(f"✅ Found {result['total_count']} returns")
                st.session_state.return_data = result['returns']
                
                # Show sample data
                if st.checkbox("Show sample data"):
                    sample_df = pd.DataFrame(result['returns'][:5])
                    st.dataframe(sample_df)
            else:
                st.error(f"❌ Error: {result['error']}")
    
    # Analyze button
    if st.session_state.return_data:
        if st.button("🚀 ANALYZE RETURNS", type="primary"):
            with st.spinner("🔍 Categorizing returns..."):
                results = process_returns_data(st.session_state.return_data)
                st.session_state.analysis_results = results
                st.success("✅ Analysis complete!")
    
    # Display results
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        st.markdown("---")
        st.markdown("### 📊 Analysis Results")
        st.info(f"Total Returns Analyzed: **{results['total_returns']}**")
        
        # Quality alert
        quality_defects = results['stats'].get('QUALITY_DEFECTS', {})
        if quality_defects.get('percentage', 0) > 20:
            st.error(f"⚠️ QUALITY ALERT: {quality_defects['percentage']:.1f}% of returns are quality-related!")
        
        # Category breakdown
        st.markdown("#### Return Categories")
        
        # Sort categories by count
        sorted_categories = sorted(
            results['stats'].items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        # Display as columns
        cols = st.columns(2)
        for i, (category, stats) in enumerate(sorted_categories):
            if stats['count'] > 0:
                col = cols[i % 2]
                with col:
                    info = RETURN_CATEGORIES[category]
                    st.metric(
                        f"{info['icon']} {category.replace('_', ' ').title()}",
                        f"{stats['count']} ({stats['percentage']:.1f}%)"
                    )
        
        # Export options
        st.markdown("---")
        st.markdown("### 💾 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Excel export
            excel_buffer = generate_excel_report(results)
            if excel_buffer:
                st.download_button(
                    "📊 Download Excel Report",
                    data=excel_buffer,
                    file_name=f"return_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.info("Excel export requires xlsxwriter")
        
        with col2:
            # CSV export
            csv_data = generate_csv_report(results)
            st.download_button(
                "📄 Download CSV Report",
                data=csv_data,
                file_name=f"return_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Show detailed breakdown
        if st.checkbox("Show detailed category breakdown"):
            category = st.selectbox(
                "Select category",
                [cat for cat, stats in sorted_categories if stats['count'] > 0]
            )
            
            if category:
                st.markdown(f"#### {RETURN_CATEGORIES[category]['icon']} {category.replace('_', ' ').title()} Details")
                category_returns = results['categorized'][category]
                
                # Convert to dataframe for display
                df_data = []
                for ret in category_returns[:20]:  # Show first 20
                    df_data.append({
                        'Order ID': ret.get('order_id', ''),
                        'ASIN': ret.get('asin', ''),
                        'Product': ret.get('product_name', '')[:50] + '...',
                        'Reason': ret.get('return_reason', ''),
                        'Comment': ret.get('buyer_comment', '')[:50] + '...'
                    })
                
                st.dataframe(pd.DataFrame(df_data))
                
                if len(category_returns) > 20:
                    st.info(f"Showing 20 of {len(category_returns)} returns")

if __name__ == "__main__":
    main()
