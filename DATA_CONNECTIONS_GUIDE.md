# Data Connections Setup Guide

The Vive Health Quality Suite now supports direct connections to databases, Google Sheets, and Smartsheet. This eliminates the need for manual file exports and enables real-time data access.

## 🔗 Supported Data Sources

- **Databases**: PostgreSQL, MySQL, SQLite
- **Google Sheets**: Import/export with service account authentication
- **Smartsheet**: Direct API integration

---

## 📊 Google Sheets Setup (RECOMMENDED - Easiest)

Google Sheets is the simplest option if you've had trouble with Google APIs before. This method uses service accounts which are more reliable than OAuth.

### Step 1: Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Select a Project" → "New Project"
3. Name it "Vive Quality Suite" → Click "Create"

### Step 2: Enable Google Sheets API

1. In your new project, go to "APIs & Services" → "Library"
2. Search for "Google Sheets API"
3. Click on it → Click "Enable"
4. Also search for "Google Drive API" and enable it

### Step 3: Create Service Account

1. Go to "APIs & Services" → "Credentials"
2. Click "Create Credentials" → "Service Account"
3. Name: `quality-suite-service`
4. Description: "Service account for quality suite data access"
5. Click "Create and Continue"
6. Skip roles (click "Continue", then "Done")

### Step 4: Create JSON Key

1. In Credentials, find your new service account
2. Click on it to open details
3. Go to "Keys" tab
4. Click "Add Key" → "Create New Key"
5. Choose "JSON" format
6. Click "Create" - JSON file will download
7. **Save this file securely** (e.g., `vive-quality-suite-credentials.json`)

### Step 5: Share Your Google Sheet

1. Open your Google Sheet with quality data
2. Click "Share" button
3. In the service account credentials JSON, find the email (format: `quality-suite-service@...iam.gserviceaccount.com`)
4. Paste this email in the share dialog
5. Give it "Editor" permissions
6. Uncheck "Notify people"
7. Click "Share"

### Step 6: Connect in App

1. Open Vive Health Quality Suite
2. In sidebar, click "➕ Add Connection"
3. Select "Google Sheets"
4. Name: "Production Data" (or your preference)
5. Upload the JSON key file
6. Click "💾 Save Connection"

### Step 7: Import Data

1. Go to Quality Screening → Pro Mode → "Connected Source" tab
2. Select your connection
3. Paste your Google Sheet ID (from URL: `docs.google.com/spreadsheets/d/[THIS_PART]/`)
4. Enter worksheet name (or leave blank for first sheet)
5. Click "📥 Import from Google Sheets"

**✅ Done!** Your data is now imported and ready for analysis.

---

## 🗄️ Database Setup

### SQLite (Simplest - Local File)

Perfect for local testing or small teams.

```python
# No server needed - just a file path
Connection Name: Local Quality DB
Database Type: SQLite
Database File: C:\data\quality_suite.db
```

**No additional setup required!**

### PostgreSQL (Production-Ready)

Best for enterprise deployments.

#### Setup on Local Machine:

1. Install PostgreSQL: https://www.postgresql.org/download/
2. During install, set password for `postgres` user
3. Open pgAdmin or terminal
4. Create database:
   ```sql
   CREATE DATABASE quality_suite;
   ```

#### Setup on Cloud (Heroku Postgres):

1. Create Heroku account
2. Create new app
3. Add "Heroku Postgres" add-on
4. Get connection details from Settings → Database Credentials

#### Connect in App:

```
Connection Name: Production Database
Database Type: PostgreSQL
Host: localhost (or cloud hostname)
Port: 5432
Database Name: quality_suite
Username: postgres
Password: [your password]
```

### MySQL (Alternative to PostgreSQL)

Similar to PostgreSQL, port is 3306 instead of 5432.

---

## 📋 Smartsheet Setup

### Step 1: Generate Access Token

1. Go to [Smartsheet](https://app.smartsheet.com/)
2. Click your profile icon → "Apps & Integrations"
3. Click "API Access"
4. Click "Generate new access token"
5. Name it "Vive Quality Suite"
6. Copy the token (save it securely - you can't see it again!)

### Step 2: Connect in App

1. In sidebar, click "➕ Add Connection"
2. Select "Smartsheet"
3. Name: "Smartsheet Production"
4. Paste Access Token
5. Click "💾 Save Connection"

### Step 3: Import Data

1. Go to Quality Screening → Pro Mode → "Connected Source" tab
2. Select your Smartsheet connection
3. Choose sheet from dropdown
4. Click "📥 Import from Smartsheet"

---

## 📝 Data Requirements

Your data source must have these columns:

### Required:
- `SKU` - Product SKU
- `Category` - Product category (CSH, MB, RHB, etc.)
- `Sold` - Units sold
- `Returned` - Units returned

### Optional (but recommended):
- `Name` - Product name
- `Landed Cost` - Cost per unit
- `Complaint_Text` - Customer feedback
- `Safety Risk` - Yes/No
- `Primary_Channel` - Amazon/B2B
- `B2B_Feedback` - Partner feedback
- `Amazon_Feedback` - Amazon-specific issues

---

## 🔄 Auto-Sync Workflow

Once connected, you can set up automatic data syncing:

### Option 1: Manual Refresh
- Keep connection active
- Click "Import" whenever you need fresh data
- No additional setup

### Option 2: Scheduled Sync (Advanced)
- Use external schedulers (cron, Windows Task Scheduler)
- API endpoints available for automation
- Contact support for enterprise setup

---

## 🚨 Troubleshooting

### Google Sheets: "Permission Denied"
- **Solution**: Make sure you shared the sheet with the service account email
- Check that the email matches exactly what's in the JSON file
- Service account email format: `name@project-id.iam.gserviceaccount.com`

### Google Sheets: "API Not Enabled"
- **Solution**: Go back to Google Cloud Console
- Enable both "Google Sheets API" AND "Google Drive API"
- Wait 1-2 minutes for activation

### Database: "Connection Refused"
- **Solution**: Check that database is running
- Verify host/port are correct
- Ensure firewall allows connection
- For cloud databases, check if your IP is whitelisted

### Smartsheet: "Invalid Token"
- **Solution**: Generate new token in Smartsheet
- Make sure you copied the entire token
- Tokens are case-sensitive

### Missing Dependency Errors
Install required packages:
```bash
# For Google Sheets:
pip install gspread google-auth

# For PostgreSQL:
pip install psycopg2

# For MySQL:
pip install mysql-connector-python

# For Smartsheet:
pip install smartsheet-python-sdk
```

---

## 💡 Best Practices

### Security:
- ✅ Never commit credentials to git
- ✅ Use environment variables for production
- ✅ Rotate access tokens quarterly
- ✅ Use read-only database users when possible
- ✅ Keep service account JSON files secure

### Performance:
- ✅ Use database queries to filter data server-side
- ✅ Index frequently queried columns
- ✅ Import only necessary date ranges
- ✅ Cache results for repeated analysis

### Data Quality:
- ✅ Validate data schemas before import
- ✅ Handle missing values appropriately
- ✅ Document column mappings
- ✅ Set up data quality checks

---

## 📞 Support

Having trouble? Here are resources:

1. **Google Sheets Issues**: Check service account email is shared correctly
2. **Database Issues**: Verify connection string and firewall rules
3. **Smartsheet Issues**: Regenerate access token
4. **General Help**: Check application logs for error details

---

## 🎯 Quick Start Recommendations

### For Small Teams (1-10 users):
**→ Use Google Sheets**
- Easy to set up (15 minutes)
- Familiar interface
- Real-time collaboration
- Free tier sufficient

### For Medium Teams (10-50 users):
**→ Use PostgreSQL**
- Better performance
- More robust querying
- Good for growing data
- Can start with free Heroku tier

### For Enterprise (50+ users):
**→ Use Smartsheet or PostgreSQL**
- Smartsheet: Enterprise features, governance
- PostgreSQL: Maximum flexibility, performance
- Both support advanced workflows

---

## 🔧 Advanced: Environment Variables

For production deployments, use environment variables instead of storing credentials in files:

```bash
# Linux/Mac
export GOOGLE_SHEETS_CREDENTIALS='{"type":"service_account",...}'
export SMARTSHEET_ACCESS_TOKEN='your_token_here'

# Windows
set GOOGLE_SHEETS_CREDENTIALS={"type":"service_account",...}
set SMARTSHEET_ACCESS_TOKEN=your_token_here
```

The app will automatically detect and use these environment variables.

---

## 📚 Example Queries

### PostgreSQL/MySQL:

```sql
-- Get high return rate products from last 30 days
SELECT
    SKU,
    Name,
    Sold,
    Returned,
    (Returned::float / Sold * 100) as Return_Rate
FROM products
WHERE date >= CURRENT_DATE - INTERVAL '30 days'
    AND (Returned::float / Sold) > 0.05
ORDER BY Return_Rate DESC;

-- Category performance summary
SELECT
    Category,
    COUNT(*) as Products,
    SUM(Sold) as Total_Sold,
    SUM(Returned) as Total_Returned,
    AVG(Returned::float / Sold * 100) as Avg_Return_Rate
FROM products
GROUP BY Category
ORDER BY Avg_Return_Rate DESC;
```

---

**🎉 You're all set!** Your quality suite is now connected to live data sources.
