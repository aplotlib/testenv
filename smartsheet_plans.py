"""
Smartsheet Project Plan Generator Module
Version 1.0

Generates Smartsheet-compatible project plans for:
- CAPA (Corrective and Preventive Action)
- Critical Case Investigation
- Rework Operations (AI-customized)

Export format: CSV compatible with Smartsheet import
Includes: Task hierarchy, dates, assignments, dependencies, status tracking
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import io

logger = logging.getLogger(__name__)


class SmartsheetProjectPlan:
    """
    Base class for Smartsheet project plan generation
    """

    def __init__(self, project_name: str, start_date: datetime = None):
        self.project_name = project_name
        self.start_date = start_date or datetime.now()
        self.tasks = []

    def add_task(self, task_name: str, duration_days: int,
                 assigned_to: str = "", dependencies: str = "",
                 status: str = "Not Started", priority: str = "Medium",
                 indent_level: int = 0, notes: str = ""):
        """
        Add a task to the project plan

        Args:
            task_name: Task description
            duration_days: How many days the task takes
            assigned_to: Who is responsible
            dependencies: Task IDs this depends on (e.g., "1,2")
            status: Not Started, In Progress, Complete, On Hold
            priority: High, Medium, Low
            indent_level: 0=parent, 1=child, 2=grandchild (for hierarchy)
            notes: Additional context
        """
        task_id = len(self.tasks) + 1

        # Calculate start date based on dependencies
        if dependencies and self.tasks:
            dep_ids = [int(d.strip()) for d in dependencies.split(',') if d.strip().isdigit()]
            if dep_ids:
                max_end_date = max([self.tasks[i-1]['End Date'] for i in dep_ids if i <= len(self.tasks)])
                start_date = max_end_date + timedelta(days=1)
            else:
                start_date = self.start_date
        else:
            start_date = self.start_date

        end_date = start_date + timedelta(days=duration_days) if duration_days > 0 else start_date

        self.tasks.append({
            'Task ID': task_id,
            'Task Name': '  ' * indent_level + task_name,  # Indent for hierarchy
            'Indent Level': indent_level,
            'Assigned To': assigned_to,
            'Status': status,
            'Priority': priority,
            'Start Date': start_date.strftime('%Y-%m-%d'),
            'End Date': end_date.strftime('%Y-%m-%d'),
            'Duration (Days)': duration_days,
            'Dependencies': dependencies,
            'Progress %': 0 if status == 'Not Started' else 50 if status == 'In Progress' else 100,
            'Notes': notes
        })

        return task_id

    def to_dataframe(self) -> pd.DataFrame:
        """Convert tasks to DataFrame"""
        return pd.DataFrame(self.tasks)

    def to_csv(self) -> str:
        """Export to CSV format for Smartsheet import"""
        df = self.to_dataframe()
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue()

    def to_excel(self) -> bytes:
        """Export to Excel format"""
        df = self.to_dataframe()
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Project Plan', index=False)

            # Format the worksheet
            workbook = writer.book
            worksheet = writer.sheets['Project Plan']

            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#4472C4',
                'font_color': 'white',
                'border': 1
            })

            # Apply header format
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)

            # Auto-fit columns
            for i, col in enumerate(df.columns):
                max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.set_column(i, i, min(max_len, 50))

        excel_buffer.seek(0)
        return excel_buffer.getvalue()


class CAPAProjectPlan(SmartsheetProjectPlan):
    """
    CAPA (Corrective and Preventive Action) Project Plan
    Based on 8D methodology and FDA requirements
    """

    def __init__(self, sku: str, product_name: str, issue_description: str,
                 return_rate: float, units_affected: int, severity: str = "Medium",
                 assigned_team_lead: str = "Quality Manager", start_date: datetime = None):

        project_name = f"CAPA - {sku} - {product_name}"
        super().__init__(project_name, start_date)

        self.sku = sku
        self.product_name = product_name
        self.issue_description = issue_description
        self.return_rate = return_rate
        self.units_affected = units_affected
        self.severity = severity
        self.team_lead = assigned_team_lead

        self._build_capa_plan()

    def _build_capa_plan(self):
        """Build the CAPA project plan following 8D methodology"""

        # Determine timeline based on severity
        urgency_multiplier = {
            'Critical': 0.5,
            'High': 0.75,
            'Medium': 1.0,
            'Low': 1.5
        }.get(self.severity, 1.0)

        def days(base_days):
            return max(1, int(base_days * urgency_multiplier))

        # PHASE 1: TEAM FORMATION (D1)
        p1 = self.add_task("Phase 1: Team Formation & Planning", 0, self.team_lead, "", "Not Started", self.severity, 0)
        self.add_task("Assemble cross-functional CAPA team", days(2), self.team_lead, "", "Not Started", self.severity, 1,
                     f"Include: Quality, Engineering, Production, Supplier Management")
        self.add_task("Schedule kickoff meeting", days(1), self.team_lead, f"{p1+1}", "Not Started", self.severity, 1)
        self.add_task("Define roles and responsibilities", days(1), self.team_lead, f"{p1+2}", "Not Started", self.severity, 1)

        # PHASE 2: PROBLEM DEFINITION (D2)
        p2 = self.add_task("Phase 2: Problem Definition & Quantification", 0, self.team_lead, f"{p1+3}", "Not Started", self.severity, 0)
        self.add_task("Document problem statement (5W2H)", days(2), "Quality Engineer", f"{p2}", "Not Started", self.severity, 1,
                     f"Issue: {self.issue_description}\nReturn Rate: {self.return_rate:.1%}\nUnits Affected: {self.units_affected:,}")
        self.add_task("Quantify impact (financial, customer, regulatory)", days(2), "Quality Analyst", f"{p2+1}", "Not Started", self.severity, 1)
        self.add_task("Review customer complaints & return data", days(3), "Quality Analyst", f"{p2+1}", "Not Started", self.severity, 1)
        self.add_task("Identify affected batches/lots", days(2), "Production Manager", f"{p2+1}", "Not Started", self.severity, 1)

        # PHASE 3: CONTAINMENT (D3)
        p3 = self.add_task("Phase 3: Immediate Containment Actions", 0, self.team_lead, f"{p2+4}", "Not Started", "High", 0)
        self.add_task("Quarantine affected inventory", days(1), "Warehouse Manager", f"{p3}", "Not Started", "High", 1)
        self.add_task("Issue stop-ship if necessary", days(1), "Quality Manager", f"{p3+1}", "Not Started", "High", 1)
        self.add_task("Notify customers (if required)", days(2), "Customer Service", f"{p3+1}", "Not Started", "High", 1)
        self.add_task("Implement sorting/inspection (if applicable)", days(3), "QA Inspector", f"{p3+2}", "Not Started", "High", 1)
        self.add_task("Document containment effectiveness", days(2), "Quality Engineer", f"{p3+4}", "Not Started", self.severity, 1)

        # PHASE 4: ROOT CAUSE ANALYSIS (D4)
        p4 = self.add_task("Phase 4: Root Cause Analysis", 0, "Quality Engineer", f"{p3+5}", "Not Started", self.severity, 0)
        self.add_task("Conduct 5 Whys analysis", days(3), "Quality Engineer", f"{p4}", "Not Started", self.severity, 1)
        self.add_task("Create Fishbone diagram", days(2), "Quality Engineer", f"{p4+1}", "Not Started", self.severity, 1)
        self.add_task("Analyze process data & trends", days(4), "Process Engineer", f"{p4+1}", "Not Started", self.severity, 1)
        self.add_task("Inspect returned units (if available)", days(3), "QA Inspector", f"{p4}", "Not Started", self.severity, 1)
        self.add_task("Verify root cause with testing", days(5), "R&D Engineer", f"{p4+2},{p4+3}", "Not Started", self.severity, 1)
        self.add_task("Document root cause findings", days(2), "Quality Engineer", f"{p4+5}", "Not Started", self.severity, 1)

        # PHASE 5: CORRECTIVE ACTIONS (D5)
        p5 = self.add_task("Phase 5: Develop Corrective Actions", 0, "Quality Engineer", f"{p4+6}", "Not Started", self.severity, 0)
        self.add_task("Brainstorm corrective action options", days(2), self.team_lead, f"{p5}", "Not Started", self.severity, 1)
        self.add_task("Evaluate feasibility & effectiveness", days(3), "Engineering Team", f"{p5+1}", "Not Started", self.severity, 1)
        self.add_task("Select best corrective actions", days(2), self.team_lead, f"{p5+2}", "Not Started", self.severity, 1)
        self.add_task("Create implementation plan", days(3), "Quality Engineer", f"{p5+3}", "Not Started", self.severity, 1)
        self.add_task("Get management approval", days(2), self.team_lead, f"{p5+4}", "Not Started", self.severity, 1)

        # PHASE 6: CORRECTIVE ACTION IMPLEMENTATION (D6)
        p6 = self.add_task("Phase 6: Implement Corrective Actions", 0, "Quality Manager", f"{p5+5}", "Not Started", self.severity, 0)
        self.add_task("Update process documentation", days(3), "Quality Engineer", f"{p6}", "Not Started", self.severity, 1)
        self.add_task("Train affected personnel", days(5), "Training Coordinator", f"{p6+1}", "Not Started", self.severity, 1)
        self.add_task("Implement process changes", days(7), "Production Manager", f"{p6+1}", "Not Started", self.severity, 1)
        self.add_task("Update inspection/test procedures", days(3), "QA Manager", f"{p6+1}", "Not Started", self.severity, 1)
        self.add_task("Communicate changes to vendors (if applicable)", days(3), "Supplier Quality", f"{p6+1}", "Not Started", self.severity, 1)

        # PHASE 7: PREVENTIVE ACTIONS (D7)
        p7 = self.add_task("Phase 7: Preventive Actions", 0, "Quality Manager", f"{p6+5}", "Not Started", self.severity, 0)
        self.add_task("Identify similar products/processes at risk", days(3), "Quality Engineer", f"{p7}", "Not Started", self.severity, 1)
        self.add_task("Implement preventive measures across organization", days(7), self.team_lead, f"{p7+1}", "Not Started", self.severity, 1)
        self.add_task("Update FMEA documentation", days(3), "Quality Engineer", f"{p7+1}", "Not Started", self.severity, 1)
        self.add_task("Revise quality control plan", days(3), "QA Manager", f"{p7+1}", "Not Started", self.severity, 1)

        # PHASE 8: VERIFICATION & CLOSURE (D8)
        p8 = self.add_task("Phase 8: Verify Effectiveness & Close", 0, self.team_lead, f"{p7+4}", "Not Started", self.severity, 0)
        self.add_task("Monitor return rate for 30 days", days(30), "Quality Analyst", f"{p8}", "Not Started", self.severity, 1)
        self.add_task("Collect verification data", days(14), "Quality Analyst", f"{p8+1}", "Not Started", self.severity, 1)
        self.add_task("Analyze effectiveness of actions", days(3), "Quality Engineer", f"{p8+2}", "Not Started", self.severity, 1)
        self.add_task("Document lessons learned", days(2), self.team_lead, f"{p8+3}", "Not Started", self.severity, 1)
        self.add_task("Prepare final CAPA report", days(3), "Quality Engineer", f"{p8+4}", "Not Started", self.severity, 1)
        self.add_task("Management review & sign-off", days(2), "Quality Manager", f"{p8+5}", "Not Started", self.severity, 1)
        self.add_task("Close CAPA in quality system", days(1), "Quality Manager", f"{p8+6}", "Not Started", self.severity, 1)


class CriticalInvestigationPlan(SmartsheetProjectPlan):
    """
    Critical Case Investigation Project Plan
    For safety concerns, regulatory issues, or high-impact quality problems
    """

    def __init__(self, sku: str, product_name: str, issue_description: str,
                 severity_level: str, regulatory_impact: bool = False,
                 safety_concern: bool = False, assigned_lead: str = "Quality Manager",
                 start_date: datetime = None):

        project_name = f"CRITICAL INVESTIGATION - {sku} - {product_name}"
        super().__init__(project_name, start_date)

        self.sku = sku
        self.product_name = product_name
        self.issue_description = issue_description
        self.severity_level = severity_level
        self.regulatory_impact = regulatory_impact
        self.safety_concern = safety_concern
        self.lead = assigned_lead

        self._build_critical_investigation()

    def _build_critical_investigation(self):
        """Build critical investigation timeline"""

        # IMMEDIATE ACTIONS (Within 24 hours)
        p1 = self.add_task("🚨 IMMEDIATE ACTIONS (24 Hours)", 0, self.lead, "", "Not Started", "Critical", 0)
        self.add_task("Notify senior management", 0.25, self.lead, "", "Not Started", "Critical", 1,
                     "Immediate escalation - same day notification required")
        self.add_task("Assemble crisis response team", 0.5, self.lead, f"{p1+1}", "Not Started", "Critical", 1,
                     "Include: Quality, Engineering, Regulatory, Legal, Operations")
        self.add_task("Emergency team meeting (within 2 hours)", 0.5, self.lead, f"{p1+2}", "Not Started", "Critical", 1)
        self.add_task("Implement immediate containment", 1, "Operations Manager", f"{p1+3}", "Not Started", "Critical", 1,
                     "Stop production/shipment, quarantine inventory")

        if self.safety_concern:
            self.add_task("⚠️ Assess patient/customer safety risk", 0.5, "Safety Officer", f"{p1+3}", "Not Started", "Critical", 1)
            self.add_task("Initiate customer notification (if required)", 1, "Regulatory Affairs", f"{p1+5}", "Not Started", "Critical", 1)

        if self.regulatory_impact:
            self.add_task("📋 Determine regulatory reporting obligations", 1, "Regulatory Affairs", f"{p1+3}", "Not Started", "Critical", 1)
            self.add_task("Prepare FDA/MDR notification (if required)", 2, "Regulatory Affairs", f"{p1+7 if self.safety_concern else p1+5}", "Not Started", "Critical", 1)

        # PHASE 2: INVESTIGATION SETUP (Days 1-2)
        last_task = p1 + 8 if self.regulatory_impact and self.safety_concern else p1 + 6 if self.safety_concern or self.regulatory_impact else p1 + 4
        p2 = self.add_task("Phase 2: Investigation Setup & Data Collection", 0, self.lead, f"{last_task}", "Not Started", "Critical", 0)

        self.add_task("Define investigation scope & objectives", 1, "Quality Manager", f"{p2}", "Not Started", "Critical", 1)
        self.add_task("Assign investigation team roles", 1, self.lead, f"{p2+1}", "Not Started", "Critical", 1)
        self.add_task("Secure all affected units for analysis", 1, "Warehouse Manager", f"{p2}", "Not Started", "Critical", 1)
        self.add_task("Preserve all relevant records/documentation", 1, "Quality Engineer", f"{p2}", "Not Started", "Critical", 1)
        self.add_task("Create investigation tracking log", 1, "Quality Analyst", f"{p2+1}", "Not Started", "High", 1)

        # PHASE 3: TECHNICAL INVESTIGATION (Days 3-7)
        p3 = self.add_task("Phase 3: Technical Investigation & Analysis", 0, "Quality Engineer", f"{p2+5}", "Not Started", "Critical", 0)

        self.add_task("Conduct failure analysis on returned units", 3, "QA Lab", f"{p3}", "Not Started", "Critical", 1,
                     "Destructive testing, microscopy, dimensional analysis")
        self.add_task("Review manufacturing records & traceability", 2, "Quality Engineer", f"{p3}", "Not Started", "Critical", 1)
        self.add_task("Analyze process data & SPC charts", 2, "Process Engineer", f"{p3}", "Not Started", "Critical", 1)
        self.add_task("Interview operators & witnesses", 2, "Quality Manager", f"{p3}", "Not Started", "High", 1)
        self.add_task("Review supplier documentation (if applicable)", 3, "Supplier Quality", f"{p3}", "Not Started", "High", 1)
        self.add_task("Conduct timeline reconstruction", 2, "Quality Engineer", f"{p3+1},{p3+2},{p3+3}", "Not Started", "Critical", 1)
        self.add_task("Perform risk assessment (FMEA update)", 2, "Quality Engineer", f"{p3+6}", "Not Started", "Critical", 1)

        # PHASE 4: ROOT CAUSE DETERMINATION (Days 8-10)
        p4 = self.add_task("Phase 4: Root Cause Analysis & Validation", 0, "Quality Engineer", f"{p3+7}", "Not Started", "Critical", 0)

        self.add_task("Conduct formal RCA session", 2, self.lead, f"{p4}", "Not Started", "Critical", 1,
                     "Use 5 Whys, Fishbone, Fault Tree Analysis")
        self.add_task("Identify contributing factors", 2, "Investigation Team", f"{p4+1}", "Not Started", "Critical", 1)
        self.add_task("Validate root cause with testing", 3, "R&D Engineer", f"{p4+2}", "Not Started", "Critical", 1)
        self.add_task("Document root cause evidence", 2, "Quality Engineer", f"{p4+3}", "Not Started", "Critical", 1)
        self.add_task("Peer review of findings", 1, "Quality Manager", f"{p4+4}", "Not Started", "Critical", 1)

        # PHASE 5: CORRECTIVE ACTIONS (Days 11-15)
        p5 = self.add_task("Phase 5: Corrective Action Planning", 0, "Quality Manager", f"{p4+5}", "Not Started", "Critical", 0)

        self.add_task("Develop corrective action plan", 2, "Quality Engineer", f"{p5}", "Not Started", "Critical", 1)
        self.add_task("Identify preventive actions", 2, "Quality Engineer", f"{p5+1}", "Not Started", "High", 1)
        self.add_task("Conduct risk/benefit analysis", 2, "Quality Manager", f"{p5+1}", "Not Started", "Critical", 1)
        self.add_task("Management review & approval", 1, "VP Quality", f"{p5+2},{p5+3}", "Not Started", "Critical", 1)

        # PHASE 6: IMPLEMENTATION & VERIFICATION (Days 16-30)
        p6 = self.add_task("Phase 6: Implementation & Verification", 0, "Quality Manager", f"{p5+4}", "Not Started", "High", 0)

        self.add_task("Implement corrective actions", 7, "Operations Manager", f"{p6}", "Not Started", "Critical", 1)
        self.add_task("Update all documentation & procedures", 3, "Quality Engineer", f"{p6+1}", "Not Started", "High", 1)
        self.add_task("Train all affected personnel", 5, "Training Manager", f"{p6+1}", "Not Started", "High", 1)
        self.add_task("Verify effectiveness (pilot run)", 7, "Production Manager", f"{p6+1},{p6+2}", "Not Started", "Critical", 1)
        self.add_task("Conduct final inspection & testing", 3, "QA Manager", f"{p6+4}", "Not Started", "Critical", 1)

        # PHASE 7: REPORTING & CLOSURE (Days 31-35)
        p7 = self.add_task("Phase 7: Documentation & Regulatory Closure", 0, self.lead, f"{p6+5}", "Not Started", "Critical", 0)

        self.add_task("Prepare investigation report", 3, "Quality Engineer", f"{p7}", "Not Started", "Critical", 1)
        self.add_task("Document lessons learned", 2, self.lead, f"{p7+1}", "Not Started", "High", 1)

        if self.regulatory_impact:
            self.add_task("📋 Submit regulatory follow-up report", 3, "Regulatory Affairs", f"{p7+1}", "Not Started", "Critical", 1)
            self.add_task("Archive investigation file per regulations", 1, "Quality Manager", f"{p7+3}", "Not Started", "Critical", 1)
        else:
            self.add_task("Archive investigation file", 1, "Quality Manager", f"{p7+2}", "Not Started", "High", 1)

        last_report_task = p7 + 4 if self.regulatory_impact else p7 + 3
        self.add_task("Executive briefing & closure meeting", 1, "VP Quality", f"{last_report_task}", "Not Started", "Critical", 1)
        self.add_task("Close investigation in quality system", 1, "Quality Manager", f"{last_report_task + 1}", "Not Started", "Critical", 1)

        # ONGOING MONITORING (30-90 days post-closure)
        p8 = self.add_task("Phase 8: Post-Closure Monitoring", 0, "Quality Analyst", f"{last_report_task + 2}", "Not Started", "High", 0)
        self.add_task("Monitor metrics for 30 days", 30, "Quality Analyst", f"{p8}", "Not Started", "High", 1,
                     "Track return rate, defect rate, customer complaints")
        self.add_task("Conduct 30-day effectiveness review", 1, "Quality Manager", f"{p8+1}", "Not Started", "High", 1)
        self.add_task("Conduct 90-day final verification", 1, "Quality Manager", f"{p8+2}", "Not Started", "Medium", 1)


class ReworkProjectPlan(SmartsheetProjectPlan):
    """
    Rework Operation Project Plan
    AI-customized based on specific rework requirements
    """

    def __init__(self, sku: str, product_name: str, units_to_rework: int,
                 rework_type: str, rework_details: Dict[str, Any],
                 assigned_lead: str = "Production Manager", start_date: datetime = None):

        project_name = f"REWORK - {sku} - {product_name}"
        super().__init__(project_name, start_date)

        self.sku = sku
        self.product_name = product_name
        self.units_to_rework = units_to_rework
        self.rework_type = rework_type
        self.rework_details = rework_details
        self.lead = assigned_lead

        self._build_rework_plan()

    def _build_rework_plan(self):
        """Build customized rework plan based on specifics"""

        details = self.rework_details

        # Calculate estimates
        batch_size = details.get('batch_size', 50)
        num_batches = (self.units_to_rework + batch_size - 1) // batch_size
        complexity = details.get('complexity', 'Medium')  # Low, Medium, High
        inspection_level = details.get('inspection_level', '100%')  # 100%, Sample, Skip-lot

        time_per_unit = {
            'Low': 0.1,
            'Medium': 0.25,
            'High': 0.5
        }.get(complexity, 0.25)  # Hours per unit

        rework_days_estimate = max(1, int((self.units_to_rework * time_per_unit) / 8))  # Assume 8-hour day

        # PHASE 1: PLANNING & SETUP
        p1 = self.add_task("Phase 1: Rework Planning & Setup", 0, self.lead, "", "Not Started", "High", 0)

        self.add_task("Review quality issue & rework requirements", 1, "Quality Engineer", "", "Not Started", "High", 1,
                     f"Rework Type: {self.rework_type}\nUnits: {self.units_to_rework:,}")
        self.add_task("Create detailed rework procedure", 2, "Quality Engineer", f"{p1+1}", "Not Started", "High", 1)
        self.add_task("Identify required tools & materials", 1, "Production Manager", f"{p1+1}", "Not Started", "High", 1,
                     details.get('required_materials', 'Standard rework materials'))
        self.add_task("Order/procure necessary parts", 3, "Procurement", f"{p1+3}", "Not Started", "High", 1)
        self.add_task("Designate rework area & equipment", 1, "Facility Manager", f"{p1+1}", "Not Started", "Medium", 1)
        self.add_task("Assign rework team personnel", 1, self.lead, f"{p1+5}", "Not Started", "High", 1,
                     f"Team size: {details.get('team_size', '2-3 operators')}")
        self.add_task("Train team on rework procedure", 2, "Training Coordinator", f"{p1+2},{p1+6}", "Not Started", "High", 1)
        self.add_task("Quality approval to proceed", 1, "Quality Manager", f"{p1+7}", "Not Started", "High", 1)

        # PHASE 2: REWORK EXECUTION
        p2 = self.add_task("Phase 2: Rework Execution", 0, self.lead, f"{p1+8}", "Not Started", "High", 0)

        if details.get('requires_disassembly', False):
            self.add_task("Disassemble affected units", max(2, num_batches), "Rework Technician", f"{p2}", "Not Started", "High", 1)
            disassembly_task = p2 + 1
        else:
            disassembly_task = p2

        if details.get('requires_cleaning', False):
            self.add_task("Clean components/units", max(1, int(num_batches * 0.5)), "Rework Technician", f"{disassembly_task}", "Not Started", "Medium", 1)
            cleaning_task = (disassembly_task + 1) if details.get('requires_disassembly') else disassembly_task
        else:
            cleaning_task = disassembly_task

        # Core rework operations
        self.add_task(f"Perform primary rework: {self.rework_type}", rework_days_estimate, "Rework Team", f"{cleaning_task}", "Not Started", "High", 1,
                     details.get('rework_steps', 'Follow approved procedure'))

        rework_task = cleaning_task + (1 if details.get('requires_cleaning') else 0) + (1 if details.get('requires_disassembly') else 0)

        if details.get('requires_testing', True):
            self.add_task("Functional testing post-rework", max(2, int(num_batches * 0.3)), "QA Technician", f"{rework_task}", "Not Started", "High", 1)
            testing_task = rework_task + 1
        else:
            testing_task = rework_task

        if details.get('requires_reassembly', False):
            self.add_task("Reassemble units", max(2, int(num_batches * 0.8)), "Rework Technician", f"{testing_task}", "Not Started", "High", 1)
            reassembly_task = testing_task + 1
        else:
            reassembly_task = testing_task

        # PHASE 3: INSPECTION & VERIFICATION
        p3 = self.add_task("Phase 3: Inspection & Verification", 0, "QA Manager", f"{reassembly_task}", "Not Started", "High", 0)

        if inspection_level == "100%":
            self.add_task("100% visual inspection", max(2, int(num_batches * 0.4)), "QA Inspector", f"{p3}", "Not Started", "High", 1,
                         f"Inspect all {self.units_to_rework:,} units")
            self.add_task("100% functional test", max(2, int(num_batches * 0.5)), "QA Technician", f"{p3+1}", "Not Started", "High", 1)
            inspection_tasks = 2
        elif inspection_level == "Sample":
            self.add_task("Sample inspection per AQL plan", max(1, int(num_batches * 0.2)), "QA Inspector", f"{p3}", "Not Started", "High", 1,
                         f"AQL {details.get('aql_level', '2.5')} sampling")
            inspection_tasks = 1
        else:  # Skip-lot
            self.add_task("First article inspection", 1, "QA Inspector", f"{p3}", "Not Started", "High", 1)
            inspection_tasks = 1

        last_inspection = p3 + inspection_tasks
        self.add_task("Document rework & inspection results", 1, "Quality Engineer", f"{last_inspection}", "Not Started", "High", 1)
        self.add_task("Quality sign-off for release", 1, "Quality Manager", f"{last_inspection + 1}", "Not Started", "High", 1)

        # PHASE 4: PACKAGING & RELEASE
        p4 = self.add_task("Phase 4: Repackaging & Release", 0, "Warehouse Manager", f"{last_inspection + 2}", "Not Started", "Medium", 0)

        if details.get('requires_relabeling', False):
            self.add_task("Relabel units with rework tracking", max(1, int(num_batches * 0.3)), "Warehouse Team", f"{p4}", "Not Started", "Medium", 1)
            relabel_task = p4 + 1
        else:
            relabel_task = p4

        if details.get('requires_repackaging', True):
            self.add_task("Repackage units", max(2, int(num_batches * 0.4)), "Packaging Team", f"{relabel_task}", "Not Started", "Medium", 1)
            repack_task = relabel_task + (1 if details.get('requires_relabeling') else 0)
        else:
            repack_task = relabel_task

        self.add_task("Update inventory & traceability records", 1, "Inventory Manager", f"{repack_task}", "Not Started", "High", 1)
        self.add_task("Return to saleable inventory", 1, "Warehouse Manager", f"{repack_task + 1}", "Not Started", "Medium", 1)

        # PHASE 5: DOCUMENTATION & CLOSEOUT
        p5 = self.add_task("Phase 5: Documentation & Closeout", 0, "Quality Engineer", f"{repack_task + 2}", "Not Started", "Medium", 0)

        self.add_task("Compile rework batch records", 2, "Quality Engineer", f"{p5}", "Not Started", "High", 1)
        self.add_task("Calculate rework costs", 1, "Finance Analyst", f"{p5}", "Not Started", "Medium", 1,
                     f"Labor + Materials for {self.units_to_rework:,} units")
        self.add_task("Update DHR/DMR if required", 1, "Quality Engineer", f"{p5+1}", "Not Started", "High", 1)
        self.add_task("Lessons learned documentation", 1, self.lead, f"{p5+1}", "Not Started", "Medium", 1)
        self.add_task("Close rework work order", 1, "Production Manager", f"{p5+4}", "Not Started", "Medium", 1)


# Export all classes
__all__ = [
    'SmartsheetProjectPlan',
    'CAPAProjectPlan',
    'CriticalInvestigationPlan',
    'ReworkProjectPlan'
]
