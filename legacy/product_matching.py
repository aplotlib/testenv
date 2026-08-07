"""
Fuzzy Product Matching and Historical Comparison Module

Uses AI to identify similar products for intelligent comparison:
- Fuzzy name matching
- Category-based similarity
- AI-powered semantic similarity
- Historical return rate benchmarking
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from difflib import SequenceMatcher
import re
import logging

logger = logging.getLogger(__name__)


class ProductMatcher:
    """
    Intelligent product matching system using fuzzy logic and AI
    """

    def __init__(self, historical_data: pd.DataFrame, ai_analyzer=None):
        """
        Initialize with historical product data

        Args:
            historical_data: DataFrame with columns: Name, SKU, Category, Return_Rate, etc.
            ai_analyzer: Optional EnhancedAIAnalyzer for semantic matching
        """
        self.historical_data = historical_data
        self.ai_analyzer = ai_analyzer

        # Prepare historical data
        self._prepare_historical_data()

        # Build category index
        self.category_index = self._build_category_index()

    def _prepare_historical_data(self):
        """Clean and prepare historical data"""
        if 'Name' in self.historical_data.columns:
            # Normalize product names
            self.historical_data['name_normalized'] = self.historical_data['Name'].apply(
                self._normalize_product_name
            )

        # Parse return rate if it's a string
        if 'Return Rate' in self.historical_data.columns:
            self.historical_data['return_rate_numeric'] = self.historical_data['Return Rate'].apply(
                self._parse_percentage
            )
        elif 'Refund Rate' in self.historical_data.columns:
            self.historical_data['return_rate_numeric'] = self.historical_data['Refund Rate'].apply(
                self._parse_percentage
            )

        # Extract product type keywords
        self.historical_data['product_keywords'] = self.historical_data['Name'].apply(
            self._extract_product_keywords
        )

    def _build_category_index(self) -> Dict[str, List[int]]:
        """Build index of products by category"""
        category_index = {}

        if 'Category' in self.historical_data.columns:
            for idx, row in self.historical_data.iterrows():
                category = row.get('Category', 'Unknown')
                if category not in category_index:
                    category_index[category] = []
                category_index[category].append(idx)

        return category_index

    def find_similar_products(
        self,
        product_name: str,
        product_category: Optional[str] = None,
        top_n: int = 5,
        similarity_threshold: float = 0.3,
        use_ai: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Find similar products from historical data

        Args:
            product_name: Name of product to match
            product_category: Optional category for filtering
            top_n: Number of similar products to return
            similarity_threshold: Minimum similarity score (0-1)
            use_ai: Whether to use AI for semantic matching

        Returns:
            List of similar products with similarity scores
        """

        similar_products = []

        # Method 1: Fuzzy string matching
        fuzzy_matches = self._fuzzy_string_matching(
            product_name, product_category, top_n * 2
        )
        similar_products.extend(fuzzy_matches)

        # Method 2: Keyword-based matching
        keyword_matches = self._keyword_matching(
            product_name, product_category, top_n * 2
        )
        similar_products.extend(keyword_matches)

        # Method 3: AI semantic matching (if available and requested)
        if use_ai and self.ai_analyzer:
            ai_matches = self._ai_semantic_matching(
                product_name, product_category, top_n * 2
            )
            similar_products.extend(ai_matches)

        # Deduplicate and rank by combined score
        unique_products = {}
        for match in similar_products:
            idx = match['index']
            if idx not in unique_products:
                unique_products[idx] = match
            else:
                # Average scores from different methods
                existing_score = unique_products[idx]['similarity_score']
                new_score = match['similarity_score']
                unique_products[idx]['similarity_score'] = (existing_score + new_score) / 2

        # Filter by threshold and sort
        filtered = [
            p for p in unique_products.values()
            if p['similarity_score'] >= similarity_threshold
        ]
        sorted_products = sorted(
            filtered,
            key=lambda x: x['similarity_score'],
            reverse=True
        )[:top_n]

        return sorted_products

    def get_benchmark_stats(
        self,
        product_name: str,
        product_category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get benchmark statistics for similar products

        Returns metrics like average return rate, percentiles, etc.
        """

        similar_products = self.find_similar_products(
            product_name, product_category, top_n=10, use_ai=False
        )

        if not similar_products:
            return {
                'similar_product_count': 0,
                'avg_return_rate': None,
                'median_return_rate': None,
                'percentile_25': None,
                'percentile_75': None,
                'best_in_class': None,
                'worst_in_class': None
            }

        return_rates = [p['return_rate'] for p in similar_products if p.get('return_rate') is not None]

        if not return_rates:
            return {
                'similar_product_count': len(similar_products),
                'avg_return_rate': None,
                'median_return_rate': None,
                'percentile_25': None,
                'percentile_75': None,
                'best_in_class': None,
                'worst_in_class': None
            }

        return {
            'similar_product_count': len(similar_products),
            'avg_return_rate': np.mean(return_rates),
            'median_return_rate': np.median(return_rates),
            'percentile_25': np.percentile(return_rates, 25),
            'percentile_75': np.percentile(return_rates, 75),
            'best_in_class': min(return_rates),
            'worst_in_class': max(return_rates),
            'similar_products': similar_products
        }

    def compare_to_similar_products(
        self,
        product_name: str,
        product_return_rate: float,
        product_category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare a product's performance against similar products

        Returns:
            Comparison metrics and ranking
        """

        benchmark = self.get_benchmark_stats(product_name, product_category)

        if benchmark['similar_product_count'] == 0:
            return {
                'comparison_available': False,
                'message': 'No similar products found for comparison'
            }

        avg_return_rate = benchmark['avg_return_rate']
        median_return_rate = benchmark['median_return_rate']

        # Calculate performance metrics
        vs_average_pct = ((product_return_rate - avg_return_rate) / avg_return_rate * 100) if avg_return_rate > 0 else 0
        vs_median_pct = ((product_return_rate - median_return_rate) / median_return_rate * 100) if median_return_rate > 0 else 0

        # Determine performance category
        if product_return_rate <= benchmark['percentile_25']:
            performance = 'Excellent'
            performance_desc = 'Top 25% performer among similar products'
        elif product_return_rate <= benchmark['median_return_rate']:
            performance = 'Good'
            performance_desc = 'Above median performance'
        elif product_return_rate <= benchmark['percentile_75']:
            performance = 'Fair'
            performance_desc = 'Below median but within normal range'
        else:
            performance = 'Needs Improvement'
            performance_desc = 'Bottom 25% performer among similar products'

        return {
            'comparison_available': True,
            'product_return_rate': product_return_rate,
            'benchmark_average': avg_return_rate,
            'benchmark_median': median_return_rate,
            'vs_average_pct': vs_average_pct,
            'vs_median_pct': vs_median_pct,
            'performance_category': performance,
            'performance_description': performance_desc,
            'similar_product_count': benchmark['similar_product_count'],
            'best_in_class': benchmark['best_in_class'],
            'worst_in_class': benchmark['worst_in_class'],
            'similar_products': benchmark.get('similar_products', [])
        }

    def _fuzzy_string_matching(
        self,
        product_name: str,
        product_category: Optional[str],
        top_n: int
    ) -> List[Dict[str, Any]]:
        """Fuzzy string matching using SequenceMatcher"""

        normalized_name = self._normalize_product_name(product_name)
        matches = []

        # Filter by category first if provided
        if product_category and product_category in self.category_index:
            candidate_indices = self.category_index[product_category]
        else:
            candidate_indices = self.historical_data.index

        for idx in candidate_indices:
            row = self.historical_data.loc[idx]
            hist_name_normalized = row.get('name_normalized', '')

            # Calculate similarity ratio
            similarity = SequenceMatcher(None, normalized_name, hist_name_normalized).ratio()

            if similarity > 0:  # Will be filtered by threshold later
                matches.append({
                    'index': idx,
                    'product_name': row.get('Name', ''),
                    'sku': row.get('SKU', ''),
                    'category': row.get('Category', ''),
                    'return_rate': row.get('return_rate_numeric'),
                    'similarity_score': similarity,
                    'match_method': 'fuzzy_string'
                })

        # Sort and return top N
        matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        return matches[:top_n]

    def _keyword_matching(
        self,
        product_name: str,
        product_category: Optional[str],
        top_n: int
    ) -> List[Dict[str, Any]]:
        """Keyword-based matching"""

        product_keywords = self._extract_product_keywords(product_name)
        matches = []

        # Filter by category if provided
        if product_category and product_category in self.category_index:
            candidate_indices = self.category_index[product_category]
        else:
            candidate_indices = self.historical_data.index

        for idx in candidate_indices:
            row = self.historical_data.loc[idx]
            hist_keywords = row.get('product_keywords', set())

            # Calculate keyword overlap
            if product_keywords and hist_keywords:
                intersection = product_keywords & hist_keywords
                union = product_keywords | hist_keywords
                jaccard_similarity = len(intersection) / len(union) if union else 0

                if jaccard_similarity > 0:
                    matches.append({
                        'index': idx,
                        'product_name': row.get('Name', ''),
                        'sku': row.get('SKU', ''),
                        'category': row.get('Category', ''),
                        'return_rate': row.get('return_rate_numeric'),
                        'similarity_score': jaccard_similarity,
                        'match_method': 'keyword'
                    })

        matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        return matches[:top_n]

    def _ai_semantic_matching(
        self,
        product_name: str,
        product_category: Optional[str],
        top_n: int
    ) -> List[Dict[str, Any]]:
        """AI-powered semantic similarity matching"""

        if not self.ai_analyzer:
            return []

        # Get candidate products
        if product_category and product_category in self.category_index:
            candidate_indices = self.category_index[product_category][:20]  # Limit for API efficiency
        else:
            candidate_indices = list(self.historical_data.index)[:20]

        candidate_names = [
            self.historical_data.loc[idx].get('Name', '')
            for idx in candidate_indices
        ]

        # Build AI prompt
        prompt = f"""Compare the following product to the list of historical products and identify which ones are most similar in function and use case.

Target Product: {product_name}

Historical Products:
{chr(10).join(f"{i+1}. {name}" for i, name in enumerate(candidate_names))}

For each historical product, provide a similarity score from 0.0 to 1.0 where:
- 1.0 = Identical or nearly identical product
- 0.8-0.9 = Very similar (same product type, minor variations)
- 0.6-0.7 = Similar (same category, different features)
- 0.4-0.5 = Somewhat related
- 0.0-0.3 = Different products

Examples:
- "4 Wheel Mobility Scooter" vs "3 Wheel Mobility Scooter" = 0.85 (both mobility scooters)
- "Upright Walker" vs "Core Carbon Rollator" = 0.90 (both rollators)
- "Knee Brace" vs "Elbow Brace" = 0.60 (same category, different body part)

Respond with only the scores in format: "1:0.85,2:0.60,3:0.90..." (no spaces, product_number:score pairs separated by commas)
"""

        system_prompt = "You are a medical device product categorization expert. Analyze product similarity based on function, use case, and target user needs."

        try:
            response = self.ai_analyzer.generate_text(prompt, system_prompt, mode='chat')

            if response:
                # Parse response
                scores = self._parse_similarity_scores(response)

                matches = []
                for product_num, score in scores.items():
                    try:
                        idx = candidate_indices[product_num - 1]
                        row = self.historical_data.loc[idx]

                        matches.append({
                            'index': idx,
                            'product_name': row.get('Name', ''),
                            'sku': row.get('SKU', ''),
                            'category': row.get('Category', ''),
                            'return_rate': row.get('return_rate_numeric'),
                            'similarity_score': score,
                            'match_method': 'ai_semantic'
                        })
                    except (IndexError, KeyError):
                        continue

                matches.sort(key=lambda x: x['similarity_score'], reverse=True)
                return matches[:top_n]

        except Exception as e:
            logger.error(f"AI semantic matching failed: {e}")

        return []

    def _normalize_product_name(self, name: str) -> str:
        """Normalize product name for comparison"""
        if not isinstance(name, str):
            return ""

        # Convert to lowercase
        normalized = name.lower()

        # Remove special characters but keep spaces
        normalized = re.sub(r'[^\w\s]', ' ', normalized)

        # Remove extra whitespace
        normalized = ' '.join(normalized.split())

        return normalized

    def _extract_product_keywords(self, name: str) -> set:
        """Extract meaningful keywords from product name"""
        if not isinstance(name, str):
            return set()

        # Common medical device keywords
        keywords = {
            'walker', 'rollator', 'cane', 'crutch', 'wheelchair', 'scooter',
            'knee', 'ankle', 'elbow', 'wrist', 'shoulder', 'back', 'neck',
            'brace', 'support', 'splint', 'wrap', 'sleeve', 'compression',
            'cushion', 'pad', 'pillow', 'mattress',
            'toilet', 'commode', 'shower', 'bath', 'safety', 'rail', 'grab bar',
            'mobility', 'lift', 'transfer', 'assist',
            '3 wheel', '4 wheel', 'folding', 'adjustable'
        }

        normalized = self._normalize_product_name(name)
        found_keywords = set()

        for keyword in keywords:
            if keyword in normalized:
                found_keywords.add(keyword)

        # Also add significant words (3+ chars, not common words)
        words = normalized.split()
        common_words = {'the', 'and', 'for', 'with', 'by', 'in', 'on', 'at', 'to', 'from'}

        for word in words:
            if len(word) >= 3 and word not in common_words:
                found_keywords.add(word)

        return found_keywords

    def _parse_percentage(self, value) -> Optional[float]:
        """Parse percentage string to float"""
        if pd.isna(value):
            return None

        if isinstance(value, (int, float)):
            return abs(float(value))

        if isinstance(value, str):
            # Remove % sign and convert
            clean_value = value.replace('%', '').replace(',', '').strip()
            try:
                num = float(clean_value)
                # If value is like -23.14%, convert to 0.2314
                return abs(num) / 100 if abs(num) > 1 else abs(num)
            except ValueError:
                return None

        return None

    def _parse_similarity_scores(self, response: str) -> Dict[int, float]:
        """Parse AI response containing similarity scores"""
        scores = {}

        try:
            # Expected format: "1:0.85,2:0.60,3:0.90"
            pairs = response.strip().split(',')

            for pair in pairs:
                if ':' in pair:
                    product_num_str, score_str = pair.split(':', 1)
                    product_num = int(product_num_str.strip())
                    score = float(score_str.strip())
                    scores[product_num] = max(0.0, min(1.0, score))  # Clamp to [0, 1]

        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to parse similarity scores: {e}")

        return scores


__all__ = ['ProductMatcher']
