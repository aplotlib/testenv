# Test Data Guide

This folder contains comprehensive, realistic test datasets for the Vive Health Quality Suite.

## 📁 Test Files

### 1. `test_data_comprehensive.csv` - Main Quality Screening Dataset

**Purpose:** Comprehensive test data showing various realistic scenarios for quality screening

**Records:** 30 products covering diverse situations

**Use For:** Quality Screening (Lite Mode or Pro Mode)

#### Scenarios Included:

##### 🚨 **Critical Safety Issues (4 products)**
- **RHB2045** - Transfer Belt with Metal Buckle RECALL
  - Metal buckle breaking causing falls
  - Confirmed injuries (hip fracture)
  - FDA notification initiated
  - Shows: Critical safety response workflow

- **MB1088** - Hospital Bed Rail Entrapment
  - FDA compliance violation (gap exceeds standards)
  - Serious injuries reported
  - Regulatory alert scenario
  - Shows: Regulatory non-compliance handling

- **CSH3012** - Rollator Brake Failure
  - Brake cables snapping
  - Patient injury from walker rolling away
  - Shows: Safety retrofit program

- **RHB1134** - Patient Lift Sling Stitching Failure
  - Stitching failure during transfer
  - Patients dropped (minor injuries)
  - Shows: Weight rating verification issues

##### ⚠️ **Minor Safety Concerns (3 products)**
- **RHB1098** - Transfer Board Splintering (skin punctures)
- **MB2089** - IV Pole Tip-Over Risk (stability issue)
- **MB1223** - Raised Toilet Seat Pinch Point Hazard

##### 📈 **High Returns - Quality Issues (8 products)**
- **RHB1011** - Transfer Belt (35% return rate, quality not safety)
  - Velcro wearing out, handles tearing
  - Actionable: Switch to metal D-rings and industrial Velcro

- **CSH4023** - Shower Chair Rust (17% returns)
  - Material coating inadequate for humid environment
  - Actionable: Upgrade to stainless or marine-grade coating

- **MB1201** - Commode Chair Odor Retention (18% returns)
  - Plastic absorbing odors
  - Actionable: Switch to HDPE or antimicrobial plastic

- **MB3012** - Bed Wedge CertiPUR Violation (15% returns)
  - False certification claim (FTC compliance issue)
  - Actionable: Source certified foam immediately

- **RHB2156** - Posture Corrector (21% returns)
  - Sizing inconsistent, comfort issues
  - Actionable: Ergonomic redesign, QC tightening

- **CSH4056** - Bath Transfer Bench (20% returns)
  - Suction cups failing on textured tubs
  - Actionable: Replace with clamp system

- **CSH3045** - Rollator Seat Fabric Tears (12% returns)
  - Premium product with mid-grade materials
  - Actionable: Upgrade to 600D reinforced nylon

- **CSH2190** - Posture Corrector Sizing (21% returns)
  - 4-inch variance in same labeled size
  - Actionable: QC standardization

##### 🎯 **High Returns - Fixable Non-Product Issues (5 products)**
- **MB2034** - Overbed Table (13% returns, 78% cosmetic/shipping)
  - Actionable: Better packaging, save $4.50/unit in returns

- **MB1145** - Patient Lift Sling (30% returns, 85% sizing errors)
  - Actionable: Interactive sizing tool, compatibility chart

- **MB3067** - Bed Assist Rail (15% returns, 72% installation difficulty)
  - Actionable: Video guide, better instructions, longer straps

- **RHB2088** - Lumbar Support Belt (18% returns, 89% sizing chart error)
  - Actionable: Fix sizing chart (Asian to US sizing)

- **RHB3034** - Back Brace Insurance Issues (12% returns, 67% reimbursement)
  - Actionable: Pursue FDA 510(k) clearance, open insurance billing

##### ✅ **Low Returns - Benchmark Products (6 products)**
- **RHB3021** - Gait Belt Premium (3% returns) - GOLD STANDARD
- **RHB1167** - Arm Sling (3% returns) - BENCHMARK
- **CSH2190** - Walker Glides (3% returns, high volume)
- **RHB1245** - Cervical Collar (4% returns, compliance meets quality)
- **CSH3089** - Upright Walker Premium (4% returns, $399 retail success)
- **CSH1178** - Crutches (9% returns, quality good but pricing question)

##### 💰 **Cost/Margin Scenarios**
- **CSH2156** - Walking Cane ($3.25 cost, 5% returns)
  - Shows: Budget product where quality improvement exceeds margin
  - Decision: Market segmentation strategy

- **CSH3089** - Upright Walker ($125 cost, 4% returns, $399 retail)
  - Shows: Premium pricing justified by quality

- **RHB1167** - Arm Sling ($4.50 cost, 3% returns)
  - Shows: Optimal cost/quality/returns balance

##### 🔧 **Engineering/Design Issues (4 products)**
- **CSH1092** - Knee Walker Veering Left (15% returns)
  - Fork alignment tolerance issue
  - Shows: Engineering tolerance problem

- **MB1156** - Toilet Frame Stripped Screws (13% returns)
  - Cheap hardware causing assembly frustration
  - Shows: Component quality impact

- **CSH1178** - Knee Walker Ergonomics (15% returns)
  - Knee pad foam wrong density, steering QC
  - Shows: Ergonomic specification issue

---

### 2. `test_voc_totals_sheet.csv` - VoC Analysis Import (Critical Safety Product)

**Purpose:** Test Voice of Customer (VoC) analysis import feature with detailed return reason breakdown

**Product:** RHB2045 - Transfer Belt with Metal Buckle (RECALLED)

**Format:** Analysis Totals Sheet format (horizontal category layout)

**Stats:**
- Total Returns: 340
- Top Issue: Product Defects/Quality (45.88% - 156 returns)
- Safety Critical: Yes

**Use For:**
1. Testing VoC import auto-detection
2. Demonstrating category breakdown visualization
3. Testing root cause recommendations engine
4. Showing critical safety product analysis

**Location to Import:**
- Quality Screening → Pro Mode
- Scroll down to "📊 VoC Analysis Import (Auto-Detect Format)"
- Upload this CSV file
- Click "🔄 Import & Analyze Totals Data"

---

### 3. `test_voc_totals_good_product.csv` - VoC Analysis Import (Benchmark Product)

**Purpose:** Test VoC import with a low-return, high-quality product for comparison

**Product:** RHB3021 - Gait Belt Premium Quality

**Format:** Analysis Totals Sheet format (horizontal category layout)

**Stats:**
- Total Returns: 195
- Top Issue: Customer Error/Changed Mind (45.64% - 89 returns)
- Safety Critical: No
- Return Rate: 3% (BENCHMARK)

**Use For:**
1. Comparing good vs bad products in VoC analysis
2. Testing multi-product comparison features
3. Demonstrating low-defect product analysis
4. Showing that returns don't always mean product issues

---

## 🎯 How to Use Test Data

### For Quality Screening:

#### **Lite Mode (Manual Entry):**
1. Go to Quality Screening → Lite Mode
2. Manually enter 3-5 products from `test_data_comprehensive.csv`
3. Suggested products to test:
   - RHB2045 (critical safety)
   - RHB1011 (high returns but fixable)
   - RHB3021 (benchmark quality)

#### **Pro Mode (File Upload):**
1. Go to Quality Screening → Pro Mode → File Upload tab
2. Upload `test_data_comprehensive.csv`
3. Click "🔍 Run AI Screening"
4. Review all 30 products with various scenarios

### For VoC Analysis:

1. Go to Quality Screening → Pro Mode
2. Scroll to "📊 VoC Analysis Import"
3. Upload `test_voc_totals_sheet.csv` (critical product)
4. Review category breakdowns, visualizations, recommendations
5. Upload `test_voc_totals_good_product.csv` (benchmark product)
6. Click "📊 View Product Comparison" to compare both products
7. Explore:
   - Category distribution charts
   - Defect analysis
   - Root cause recommendations
   - Multi-product comparison heatmap

---

## 📊 Expected AI Analysis Results

### Critical Safety Products (RHB2045, MB1088, CSH3012, RHB1134):
- **Priority:** Critical
- **Action:** Immediate recall/retrofit
- **AI Insights:** Safety pattern recognition, liability assessment
- **Cost Impact:** High ($28K - $125K+ range)

### High Return - Quality Issues (RHB1011, CSH4023, MB1201):
- **Priority:** High
- **Action:** Product redesign, material upgrade
- **AI Insights:** Root cause analysis, supplier issues
- **Cost Impact:** Medium ($2.50 - $8.50/unit)
- **ROI:** Positive (return rate reduction justifies investment)

### High Return - Non-Product Issues (MB1145, RHB2088, MB3067):
- **Priority:** Medium
- **Action:** Educational materials, better documentation
- **AI Insights:** Customer error patterns, sizing confusion
- **Cost Impact:** Low ($0.15 - $8.50)
- **ROI:** Excellent (simple fixes with high impact)

### Low Return - Benchmark (RHB3021, RHB1167, CSH3089):
- **Priority:** Low (monitor)
- **Action:** Continue current standards
- **AI Insights:** Quality benchmarking, best practices
- **Cost Impact:** None (maintain status quo)

---

## 💡 Test Scenarios to Explore

### 1. **Safety Alert Workflow**
Load RHB2045, MB1088, or CSH3012 and observe:
- Safety risk flagging
- Regulatory compliance alerts
- Recall recommendation logic
- FDA notification triggers

### 2. **Cost-Benefit Analysis**
Compare CSH2156 (budget product) vs CSH3089 (premium):
- Quality improvement costs vs margin impact
- Market segmentation strategy
- Pricing elasticity analysis

### 3. **Root Cause Prioritization**
Load MB1145 (sizing errors) or RHB2088 (wrong chart):
- AI identifies non-product issues
- Low-cost, high-impact solutions
- Customer education recommendations

### 4. **Multi-Product Portfolio View**
Load full `test_data_comprehensive.csv`:
- See portfolio risk distribution
- Identify systemic issues (e.g., multiple products with suction cup failures)
- Resource allocation prioritization

### 5. **VoC Deep Dive**
Import both VoC files and compare:
- Category trend analysis
- Defect vs customer error ratios
- Emerging issues detection
- Product comparison heatmaps

---

## 🔍 Key Insights from Test Data

### Pattern Recognition:
- **Suction Cup Failures:** Multiple products (CSH4056, others) = systemic design issue
- **Sizing Chart Problems:** RHB2088, RHB2156 = need standardized sizing process
- **Hardware Quality:** MB1156, others = need supplier audit
- **Bathroom Products Corroding:** CSH4023 = material spec issue for humid environments

### Cost Opportunities:
- **Packaging Improvements:** MB2034 - $2.50 investment saves $4.50 in returns
- **Sizing Tools:** MB1145 - $8,500 development eliminates 18% of returns
- **Documentation:** MB3067 - Video guide solves 72% of returns

### Compliance Risks:
- **False Certification:** MB3012 - FTC violation exposure
- **FDA Non-Compliance:** MB1088 - regulatory action risk
- **Weight Rating Violations:** RHB1134 - liability exposure

---

## 📝 Notes

### Data Authenticity:
This test data is realistic and based on:
- Actual medical equipment return patterns
- Real customer feedback themes
- Industry-standard return rates (3-35% range)
- Authentic regulatory scenarios
- Realistic cost structures

### Data Completeness:
All required columns present:
- ✅ SKU, Name, Category
- ✅ Sold, Returned (for return rate calculation)
- ✅ Landed Cost (for financial impact)
- ✅ Complaint_Text (detailed, realistic feedback)
- ✅ Safety Risk flag
- ✅ Channel information
- ✅ Context notes

### Update Date:
Test data created: January 26, 2026
Reflects current pricing, regulations, and market conditions

---

## 🚀 Quick Start

**Want to see everything in 5 minutes?**

1. Upload `test_data_comprehensive.csv` to Pro Mode
2. Run AI Screening
3. Review the Critical priority products (top of list)
4. Import `test_voc_totals_sheet.csv` in VoC section
5. Click through the visualization tabs
6. Check out the root cause recommendations

**You'll see:**
- ✅ Safety issues flagged immediately
- ✅ Cost-benefit analysis for each fix
- ✅ Actionable recommendations
- ✅ Prioritized action list
- ✅ Beautiful visualizations
- ✅ Multi-product comparisons

---

Need help? Check the main README.md for full application documentation.
