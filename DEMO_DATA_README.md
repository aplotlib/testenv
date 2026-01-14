# Advanced Demo Dataset Documentation

## 📊 Overview

The **Advanced Demo Dataset** (`demo_quality_screening_data_advanced.csv`) contains 70 real products from the Vive Health catalog with realistic quality issues, designed to showcase all AI-powered features of the Quality Suite.

**Data Snapshot Date:** January 14, 2026 (30-day sales window)

## 🎯 Purpose

This dataset demonstrates how the AI-powered quality management system:
1. **Saves Time**: Screens 70 products in seconds vs. hours of manual review
2. **AI Analysis**: Automatically identifies 30+ products exceeding quality thresholds
3. **Fuzzy Matching**: Compares against 231 historical products for benchmarking
4. **Multilingual**: Generates vendor emails in 9 languages with cultural adaptation
5. **Deep Dive**: AI-powered investigation methodology recommendations
6. **Risk Detection**: Flags safety concerns and critical issues

## 📈 Dataset Statistics

- **Total Products**: 70 real products from Vive Health catalog
- **Total Units Sold**: ~400,000 units
- **Total Units Returned**: ~60,000 units
- **Average Return Rate**: 15.2%
- **Products Exceeding Threshold**: 35+ (50%)
- **Critical Severity**: 12 products
- **High Severity**: 28 products
- **Safety Risks Flagged**: 8 products

## 🏷️ Product Categories

| Category | Code | Products | Threshold | Description |
|----------|------|----------|-----------|-------------|
| Support Products | SUP | 24 | 11-20% | Braces, wraps, splints |
| Rehabilitation | RHB | 8 | 10.5-24% | PT equipment, transfer aids |
| Living Aids | LVA | 20 | 9-18.5% | Bathroom safety, mobility |
| Cushions | CSH | 10 | 11-18% | Comfort products, padding |
| Mobility | MOB | 6 | 10-15% | Walkers, scooters, wheelchairs |
| Insoles | INS | 2 | 12-14% | Foot support products |

## 🔍 Realistic Quality Issues by Category

### Support Products (SUP)
- **Common Issues**: Sizing problems (too small/large), velcro failures, stitching quality
- **Example**: "Post Op Shoe" - 23.4% return rate (threshold: 24%)
- **Complaint Pattern**: "Sizing issues - too small, Material quality - stitching came apart, Velcro doesn't hold"

### Rehabilitation (RHB)
- **Common Issues**: Durability, unclear instructions, material quality
- **Example**: "Closed Toe Post OP Shoe" - 30.5% return rate (threshold: 24%)
- **Complaint Pattern**: "Sizing inconsistent, Physical therapy ineffective, Material quality substandard"

### Living Aids (LVA)
- **Common Issues**: Assembly difficulties, stability concerns, missing parts
- **Example**: "Portable Stand Assist" - 50% return rate (threshold: 18.5%) - **CRITICAL**
- **Complaint Pattern**: "Assembly too difficult, Stability concerns - wobbly, Parts missing from box"
- **Safety Flag**: Yes ⚠️

### Mobility (MOB)
- **Common Issues**: Battery problems, motor noise, brake failures
- **Example**: "Folding Mobility Scooter" - 15.4% return rate (threshold: 10%)
- **Complaint Pattern**: "Battery doesn't hold charge, Motor makes loud noise, Seat uncomfortable for long use"

## 💡 Key Features Demonstrated

### 1. AI-Powered Screening
Upload the dataset and watch the AI:
- Calculate risk scores (0-100 scale)
- Identify SPC control chart violations
- Determine action recommendations
- Flag safety concerns

### 2. Fuzzy Product Matching
The app compares each product against 231 historical products:
- **"4 Wheel Mobility Scooter"** matches **"3 Wheel Mobility Scooter"** (0.85 similarity)
- **"Post Op Shoe"** matches similar footwear products
- **Benchmarking**: See if your 15% return rate is normal (historical avg: 12%)

### 3. Multilingual Vendor Communications
Select any flagged product and generate:
- **English proficiency levels**: Native, Fluent, Intermediate, Basic, Minimal
- **9 languages**: Chinese (Simplified/Traditional), Spanish, Portuguese, Hindi, German, French, Italian
- **Cultural adaptation**: High-context for China, direct for USA, formal for India
- **Both versions**: Original English + translation side-by-side

Example for Chinese vendor:
```
English Level: Intermediate
Target Language: Chinese (Simplified)
Region: China

Output:
✅ English version (clear, simple sentences)
✅ Chinese translation (culturally appropriate)
✅ Professional tone maintaining face-saving
```

### 4. Deep Dive Analysis
Upload product manuals/specs for AI recommendations:
- **5 Whys** for simple linear problems
- **Fishbone** for complex multi-factor issues
- **Formal RCA** for critical/safety concerns
- **FMEA** for proactive risk assessment
- **8D** for customer-facing team responses

### 5. Statistical Analysis
- **ANOVA/MANOVA**: Compare return rates across categories
- **SPC Charts**: Identify statistical outliers (>2σ, >3σ)
- **Trend Analysis**: 30/60/90/180-day comparisons
- **Risk Scoring**: Weighted algorithm (40% threshold, 30% safety, 20% value, 10% trend)

## 🚀 How to Use

### Step 1: Download the Dataset
In the Quality Suite app:
1. Go to **Tab 3: Quality Screening**
2. Click **"🚀 Download Advanced Demo Dataset"**
3. Save `demo_quality_screening_advanced.csv`

### Step 2: Upload and Screen
1. Click **"Upload CSV/Excel for Mass Analysis"**
2. Select the downloaded demo file
3. Review **Data Validation Report**
4. Click **"Run Full Screening Analysis"**
5. Wait 5-10 seconds for AI analysis

### Step 3: Explore Results
You'll see:
- **35+ products** flagged (exceeding thresholds)
- **12 critical** severity items needing immediate attention
- **8 safety risks** flagged automatically
- **Color-coded table**: Red (critical), Orange (cases), Yellow (monitor)

### Step 4: Test AI Features

#### Generate Multilingual Email
1. Find "Portable Stand Assist (Black)" - 50% return rate (CRITICAL)
2. Scroll to **"Generate AI-Powered Vendor Email"**
3. Select:
   - English Level: **Intermediate**
   - Target Language: **Chinese (Simplified)**
   - Vendor Region: **China**
4. Click **"🚀 Generate AI Email"**
5. Review both English and Chinese versions

#### Bulk Operations
1. Scroll to **"Advanced Operations & Tools"**
2. Tab: **"📧 Bulk Vendor Emails"**
3. Select 10-15 critical products
4. Choose: **CAPA Request**
5. Click **"🚀 Generate All AI Emails"**
6. Get 10-15 customized professional emails in seconds

#### Product Comparison
1. Scroll to results section
2. Expand **"🔍 Product Comparison vs. Historical Data"**
3. See benchmarking:
   - "Post Op Shoe: Your 23.4% vs Historical Avg 21.5% (Good)"
   - "Portable Stand Assist: Your 50% vs Historical Avg 18.5% (Needs Improvement - Bottom 25%)"

### Step 5: Export & Take Action
1. Download **Excel with formulas** (includes dashboard tab)
2. Download **Comparison Report** (benchmarking data)
3. Generate **CAPA Project Plans** for Smartsheet import
4. Create **Investigation Plans** with timelines

## 📊 Expected Results

### Products to Watch (Critical Priority)

1. **Shoulder Brace - Rotator Cuff** (SUP1041BGEFBM)
   - Return Rate: **62.96%** (Threshold: 14%)
   - Severity: **Critical**
   - Issues: Stitching failures, velcro problems, inadequate support
   - Action: Immediate escalation + vendor CAPA

2. **Portable Stand Assist** (LVA3016BLK)
   - Return Rate: **50.01%** (Threshold: 18.5%)
   - Severity: **Critical**
   - Safety Risk: **⚠️ YES** (stability concerns)
   - Issues: Assembly difficulty, wobbly, missing parts
   - Action: Safety investigation + possible recall

3. **Closed Toe Post OP Shoe** (RHB2096BLKL)
   - Return Rate: **30.52%** (Threshold: 24%)
   - Severity: **High**
   - Issues: Sizing inconsistent, unclear instructions
   - Action: Quality case + vendor investigation

### Time Savings Demonstration

**Manual Process** (without AI):
- Review 70 products manually: **3-4 hours**
- Identify threshold exceedances: **1-2 hours**
- Research similar products: **2-3 hours**
- Write 15 vendor emails: **2-3 hours**
- Create investigation plans: **1-2 hours**
- **Total: 9-14 hours**

**With AI-Powered Quality Suite:**
- Upload file: **10 seconds**
- AI screening: **5-10 seconds**
- Review results: **10 minutes**
- Generate 15 emails: **30 seconds**
- Create plans: **2 minutes**
- **Total: 15 minutes**

**Time Saved: 8-13 hours per screening session** ⚡

## 🎓 Educational Value

This dataset teaches:
1. **Quality Thresholds**: Different categories have different acceptable return rates
2. **Risk Prioritization**: Not all high return rates are equal (safety > cost > volume)
3. **Pattern Recognition**: Similar complaints indicate systemic issues
4. **Benchmarking**: Context matters - 15% might be normal or terrible depending on product type
5. **Cultural Communication**: Same quality issue, different communication style per region

## 🔧 Technical Details

### Column Definitions

| Column | Type | Description |
|--------|------|-------------|
| SKU | String | Product identifier from Vive catalog |
| Name | String | Product name (actual from catalog) |
| Category | String | SUP, RHB, LVA, CSH, MOB, INS |
| Sold | Integer | Units sold in 30-day period |
| Returned | Integer | Units returned in period |
| Return_Rate | Float | Calculated return rate (0.0-1.0) |
| Return_Rate_Threshold | Float | Category-specific SOP threshold |
| Landed_Cost | Float | Estimated unit cost (USD) |
| Complaint_Text | String | Comma-separated complaint reasons |
| Safety_Risk | Boolean | Safety concern flag |
| Severity | String | Critical / High / Medium / Low |
| Data_Snapshot_Date | Date | 2026-01-14 |

### Data Generation Method

1. **Base Products**: Top 100 from trailing 12-month Amazon return report
2. **Return Rates**: 70% within threshold, 30% exceeding (realistic distribution)
3. **Complaints**: Category-specific patterns based on actual product types
4. **Safety Flags**: Rule-based on complaints (brakes, stability, etc.)
5. **Costs**: Estimated from sales data (30-50% of retail)

## 📝 Notes

- **Not Real Quality Data**: Scenarios are realistic but synthetically generated for demo purposes
- **SKUs are Real**: Product names and SKUs from actual Vive Health catalog
- **Thresholds are Actual**: From Vive Health SOPs (VREC-001)
- **Snapshot Date**: Fixed date for consistency in demos
- **Historical Matching**: Works with actual `Trailing 12 Month Returns on Amazon - Use This.csv`

## 🎯 Success Metrics

After using this demo, users should understand:
- ✅ How AI reduces screening time from hours to minutes
- ✅ How fuzzy matching provides industry context
- ✅ How multilingual support enables global supply chain management
- ✅ How automated risk scoring prioritizes limited resources
- ✅ How the tool integrates with existing workflows (Smartsheet, Excel)

---

**Generated by**: Vive Health Quality Suite v20.0
**For Support**: See Interactive Help Guide in the app
**Last Updated**: January 14, 2026
