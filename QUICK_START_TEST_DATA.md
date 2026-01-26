# 🚀 Quick Start - Test Data Usage

## Ready-to-Use CSV Files

All test data files are in **CSV format** and ready to upload directly to the app!

---

## 📁 Available Files

### 1. **test_data_comprehensive.csv** ⭐ MAIN FILE
- **Size:** 30 products
- **Use For:** Quality Screening (Pro Mode)
- **Contains:** All scenarios (safety issues, high returns, benchmark products)
- **Format:** CSV (ready to upload)

### 2. **test_voc_totals_sheet.csv**
- **Size:** 1 product (340 returns analyzed)
- **Use For:** VoC Analysis Import
- **Contains:** Critical safety product with detailed return categories
- **Format:** CSV (VoC Totals Sheet format)

### 3. **test_voc_totals_good_product.csv**
- **Size:** 1 product (195 returns analyzed)
- **Use For:** VoC Analysis Import
- **Contains:** Benchmark quality product for comparison
- **Format:** CSV (VoC Totals Sheet format)

### 4. **TEST_DATA_README.md** 📖
- **Format:** Markdown (documentation only, not for upload)
- **Contains:** Detailed guide, scenario explanations, expected results

---

## ⚡ 2-Minute Quick Test

### Step 1: Test Quality Screening (30 seconds)

1. Open the app → **Quality Screening** tab
2. Click **Pro Mode** (top right)
3. Click **File Upload** tab
4. Click "Upload Product Data"
5. Select **`test_data_comprehensive.csv`**
6. Click **"🔍 Run AI Screening"**

**You'll see:**
- 30 products analyzed instantly
- Critical safety issues at the top (red flags)
- High return products with actionable fixes
- Benchmark products (3-4% returns)
- Cost-benefit analysis for each issue

---

### Step 2: Test VoC Analysis (1 minute)

1. Stay in **Pro Mode**
2. Scroll down to **"📊 VoC Analysis Import (Auto-Detect Format)"**
3. Click to expand the section
4. Upload **`test_voc_totals_sheet.csv`**
5. Click **"🔄 Import & Analyze Totals Data"**

**You'll see:**
- Product: Transfer Belt (CRITICAL - recalled)
- Return category breakdown (45% defects)
- Interactive pie charts
- Root cause recommendations
- Export buttons

---

### Step 3: Compare Products (30 seconds)

1. Upload **`test_voc_totals_good_product.csv`**
2. Click **"📊 View Product Comparison"**
3. Explore the tabs:
   - Comparison Table
   - Visual Comparison (heatmaps, charts)
   - Emerging Issues Detection

**You'll see:**
- Side-by-side comparison
- Critical product (40% returns) vs Benchmark (3% returns)
- Category heatmap
- Risk distribution

---

## 📱 Mobile/Desktop Compatibility

✅ **All CSV files work on:**
- Windows Desktop App
- Mac Desktop App
- Web Browser (Chrome, Edge, Safari, Firefox)
- Mobile browsers (iOS Safari, Android Chrome)

---

## 🎯 What Each File Tests

### test_data_comprehensive.csv Tests:
- ✅ File upload and parsing
- ✅ AI screening with multiple models
- ✅ Safety risk detection
- ✅ Return rate calculations
- ✅ Cost-benefit analysis
- ✅ Priority sorting (Critical → Low)
- ✅ Regulatory compliance flagging
- ✅ Action recommendations

### test_voc_totals_sheet.csv Tests:
- ✅ VoC format auto-detection
- ✅ Category breakdown parsing
- ✅ Return reason analysis
- ✅ Defect vs customer error classification
- ✅ Pie chart visualization
- ✅ Bar chart visualization
- ✅ Root cause recommendations
- ✅ Export functionality

### test_voc_totals_good_product.csv Tests:
- ✅ Multi-product comparison
- ✅ Benchmark vs critical product analysis
- ✅ Heatmap generation
- ✅ Emerging issues detection
- ✅ Portfolio risk distribution

---

## 💡 Expected Results

### Quality Screening Results:

**Critical Priority (4 products):**
- RHB2045 - Metal buckle recall (injuries confirmed)
- MB1088 - Bed rail entrapment (FDA violation)
- CSH3012 - Brake failure (retrofit needed)
- RHB1134 - Sling stitching failure (drops patients)

**High Priority (13 products):**
- Products with 15-35% return rates
- Actionable quality issues
- Cost-effective fixes available

**Low Priority (6 products):**
- 3-4% return rates (benchmark)
- Gold standard quality
- Continue current standards

### VoC Analysis Results:

**Critical Product (RHB2045):**
- 45.88% Product Defects
- 22.94% Customer Error
- Safety recommendations generated
- Recall advised

**Benchmark Product (RHB3021):**
- 0.51% Product Defects (only 1 return!)
- 45.64% Customer Error (not product issues)
- No safety concerns
- "Continue excellence" recommendation

---

## 🔍 File Format Details

### CSV Format Requirements:
✅ Comma-separated values
✅ UTF-8 encoding
✅ Headers in first row
✅ No special formatting needed

### Required Columns (Quality Screening):
- `SKU` - Product identifier
- `Category` - Product category code
- `Sold` - Units sold
- `Returned` - Units returned

### Optional Columns (Enhanced Analysis):
- `Name` - Product name (highly recommended)
- `Landed Cost` - Cost per unit (for financial impact)
- `Complaint_Text` - Customer feedback (for AI analysis)
- `Safety Risk` - Yes/No flag
- `Primary_Channel` - Amazon/B2B/Both
- Other context columns

### VoC CSV Format:
- Row 0: Product metadata (ID, Product, SKU, dates)
- Rows 4+: Category data (Category, Count, Percentage, Comments)
- Horizontal layout with return comments across columns

---

## ⚠️ Common Issues

### "File not found" Error:
**Solution:** Make sure you're selecting the CSV file, not the README.md

### "Invalid format" Error:
**Solution:** Use the CSV files (test_data*.csv), not the markdown file

### "Missing columns" Warning:
**Solution:** The CSV files already have all required columns - this shouldn't happen with our test data

### VoC Import not showing:
**Solution:**
1. Make sure you're in **Pro Mode**
2. Scroll down below the file upload section
3. Look for expandable section "📊 VoC Analysis Import"
4. Click to expand it

---

## 📞 Need Help?

**Can't find files?**
- Files are in the main `testenv` folder
- Same location as `app.py`
- Look for filenames starting with `test_`

**Upload not working?**
- Make sure you're uploading `.csv` files, not `.md`
- Check file size (should be 3-21 KB)
- Try refreshing the page

**Want more details?**
- Read `TEST_DATA_README.md` for comprehensive guide
- See scenario breakdowns and expected AI insights

---

## 🎉 You're Ready!

Just upload **`test_data_comprehensive.csv`** and click **"Run Screening"** to see the quality suite in action!

All files are production-ready CSVs compatible with the web app.
