"""
Data Source Integrations
Supports: PostgreSQL, MySQL, SQLite, Google Sheets, Smartsheet
"""

import pandas as pd
from typing import Optional, Dict, List, Any
import json
import os
from pathlib import Path
from datetime import datetime


class DatabaseConnector:
    """Universal database connector supporting multiple database types"""

    def __init__(self, db_type: str, connection_params: Dict[str, str]):
        """
        Initialize database connection

        Args:
            db_type: 'postgresql', 'mysql', or 'sqlite'
            connection_params: Connection parameters (host, port, database, user, password)
        """
        self.db_type = db_type.lower()
        self.connection_params = connection_params
        self.connection = None

    def connect(self) -> bool:
        """Establish database connection"""
        try:
            if self.db_type == 'postgresql':
                import psycopg2
                self.connection = psycopg2.connect(
                    host=self.connection_params.get('host', 'localhost'),
                    port=self.connection_params.get('port', 5432),
                    database=self.connection_params.get('database'),
                    user=self.connection_params.get('user'),
                    password=self.connection_params.get('password')
                )
                return True

            elif self.db_type == 'mysql':
                import mysql.connector
                self.connection = mysql.connector.connect(
                    host=self.connection_params.get('host', 'localhost'),
                    port=self.connection_params.get('port', 3306),
                    database=self.connection_params.get('database'),
                    user=self.connection_params.get('user'),
                    password=self.connection_params.get('password')
                )
                return True

            elif self.db_type == 'sqlite':
                import sqlite3
                db_path = self.connection_params.get('database', 'quality_suite.db')
                self.connection = sqlite3.connect(db_path)
                return True

            else:
                raise ValueError(f"Unsupported database type: {self.db_type}")

        except Exception as e:
            print(f"Database connection error: {e}")
            return False

    def query(self, sql: str, params: tuple = None) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame"""
        try:
            df = pd.read_sql_query(sql, self.connection, params=params)
            return df
        except Exception as e:
            print(f"Query error: {e}")
            return pd.DataFrame()

    def execute(self, sql: str, params: tuple = None) -> bool:
        """Execute SQL statement (INSERT, UPDATE, DELETE)"""
        try:
            cursor = self.connection.cursor()
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            self.connection.commit()
            cursor.close()
            return True
        except Exception as e:
            print(f"Execute error: {e}")
            return False

    def write_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append') -> bool:
        """
        Write DataFrame to database table

        Args:
            df: DataFrame to write
            table_name: Target table name
            if_exists: 'fail', 'replace', or 'append'
        """
        try:
            df.to_sql(table_name, self.connection, if_exists=if_exists, index=False)
            return True
        except Exception as e:
            print(f"Write error: {e}")
            return False

    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()


class GoogleSheetsConnector:
    """
    Simplified Google Sheets integration using gspread

    Setup Instructions:
    1. Go to https://console.cloud.google.com/
    2. Create a new project or select existing
    3. Enable Google Sheets API
    4. Create Service Account credentials
    5. Download JSON key file
    6. Share your Google Sheet with the service account email
    """

    def __init__(self, credentials_file: str = None):
        """
        Initialize Google Sheets connector

        Args:
            credentials_file: Path to service account JSON file
        """
        self.credentials_file = credentials_file
        self.client = None

    def connect(self) -> bool:
        """Establish connection to Google Sheets"""
        try:
            import gspread
            from google.oauth2.service_account import Credentials

            # Define scopes
            scopes = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]

            # Load credentials
            if self.credentials_file and os.path.exists(self.credentials_file):
                creds = Credentials.from_service_account_file(self.credentials_file, scopes=scopes)
            else:
                # Try to load from environment variable
                creds_json = os.environ.get('GOOGLE_SHEETS_CREDENTIALS')
                if creds_json:
                    import json
                    creds_dict = json.loads(creds_json)
                    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
                else:
                    raise ValueError("No Google Sheets credentials found")

            self.client = gspread.authorize(creds)
            return True

        except ImportError:
            print("Missing dependencies. Install with: pip install gspread google-auth")
            return False
        except Exception as e:
            print(f"Google Sheets connection error: {e}")
            return False

    def read_sheet(self, spreadsheet_id: str, worksheet_name: str = None) -> pd.DataFrame:
        """
        Read data from Google Sheet

        Args:
            spreadsheet_id: Google Sheet ID (from URL)
            worksheet_name: Specific worksheet name (default: first sheet)
        """
        try:
            spreadsheet = self.client.open_by_key(spreadsheet_id)

            if worksheet_name:
                worksheet = spreadsheet.worksheet(worksheet_name)
            else:
                worksheet = spreadsheet.sheet1

            data = worksheet.get_all_records()
            df = pd.DataFrame(data)
            return df

        except Exception as e:
            print(f"Read error: {e}")
            return pd.DataFrame()

    def write_sheet(self, df: pd.DataFrame, spreadsheet_id: str,
                   worksheet_name: str = None, clear_first: bool = True) -> bool:
        """
        Write DataFrame to Google Sheet

        Args:
            df: DataFrame to write
            spreadsheet_id: Google Sheet ID
            worksheet_name: Target worksheet name (creates if doesn't exist)
            clear_first: Clear existing data before writing
        """
        try:
            spreadsheet = self.client.open_by_key(spreadsheet_id)

            # Get or create worksheet
            try:
                if worksheet_name:
                    worksheet = spreadsheet.worksheet(worksheet_name)
                else:
                    worksheet = spreadsheet.sheet1
            except:
                worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows=1000, cols=26)

            # Clear if requested
            if clear_first:
                worksheet.clear()

            # Write data
            data = [df.columns.tolist()] + df.values.tolist()
            worksheet.update('A1', data)

            return True

        except Exception as e:
            print(f"Write error: {e}")
            return False

    def append_rows(self, df: pd.DataFrame, spreadsheet_id: str,
                   worksheet_name: str = None) -> bool:
        """Append rows to existing Google Sheet"""
        try:
            spreadsheet = self.client.open_by_key(spreadsheet_id)

            if worksheet_name:
                worksheet = spreadsheet.worksheet(worksheet_name)
            else:
                worksheet = spreadsheet.sheet1

            # Append rows
            rows = df.values.tolist()
            worksheet.append_rows(rows)

            return True

        except Exception as e:
            print(f"Append error: {e}")
            return False


class SmartsheetConnector:
    """
    Smartsheet integration using smartsheet-python-sdk

    Setup Instructions:
    1. Go to https://app.smartsheet.com/
    2. Account > Apps & Integrations > API Access
    3. Generate new access token
    4. Copy token to use in connection
    """

    def __init__(self, access_token: str = None):
        """
        Initialize Smartsheet connector

        Args:
            access_token: Smartsheet API access token
        """
        self.access_token = access_token or os.environ.get('SMARTSHEET_ACCESS_TOKEN')
        self.client = None

    def connect(self) -> bool:
        """Establish connection to Smartsheet"""
        try:
            import smartsheet

            if not self.access_token:
                raise ValueError("No Smartsheet access token provided")

            self.client = smartsheet.Smartsheet(self.access_token)
            self.client.errors_as_exceptions(True)

            return True

        except ImportError:
            print("Missing dependencies. Install with: pip install smartsheet-python-sdk")
            return False
        except Exception as e:
            print(f"Smartsheet connection error: {e}")
            return False

    def list_sheets(self) -> List[Dict[str, Any]]:
        """List all accessible sheets"""
        try:
            response = self.client.Sheets.list_sheets(include_all=True)
            sheets = []
            for sheet in response.data:
                sheets.append({
                    'id': sheet.id,
                    'name': sheet.name,
                    'permalink': sheet.permalink
                })
            return sheets
        except Exception as e:
            print(f"List sheets error: {e}")
            return []

    def read_sheet(self, sheet_id: int) -> pd.DataFrame:
        """Read data from Smartsheet"""
        try:
            sheet = self.client.Sheets.get_sheet(sheet_id)

            # Extract column names
            columns = [col.title for col in sheet.columns]

            # Extract rows
            rows = []
            for row in sheet.rows:
                row_data = []
                for cell in row.cells:
                    row_data.append(cell.value)
                rows.append(row_data)

            df = pd.DataFrame(rows, columns=columns)
            return df

        except Exception as e:
            print(f"Read error: {e}")
            return pd.DataFrame()

    def write_sheet(self, df: pd.DataFrame, sheet_id: int, clear_first: bool = False) -> bool:
        """
        Write DataFrame to Smartsheet

        Note: Smartsheet has complex row/column structure.
        This is a simplified implementation that appends rows.
        """
        try:
            sheet = self.client.Sheets.get_sheet(sheet_id)

            # Get column IDs
            column_map = {col.title: col.id for col in sheet.columns}

            # Build rows
            import smartsheet
            rows_to_add = []

            for _, row_data in df.iterrows():
                new_row = smartsheet.models.Row()
                new_row.to_bottom = True

                for col_name, value in row_data.items():
                    if col_name in column_map:
                        cell = smartsheet.models.Cell()
                        cell.column_id = column_map[col_name]
                        cell.value = value
                        new_row.cells.append(cell)

                rows_to_add.append(new_row)

            # Add rows to sheet
            if rows_to_add:
                self.client.Sheets.add_rows(sheet_id, rows_to_add)
                return True

            return False

        except Exception as e:
            print(f"Write error: {e}")
            return False


class ConnectionManager:
    """Manage and persist data source connections"""

    def __init__(self, config_file: str = "data_connections.json"):
        """Initialize connection manager"""
        self.config_file = config_file
        self.connections = {}
        self.load_connections()

    def load_connections(self):
        """Load saved connections from config file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.connections = json.load(f)
            except:
                self.connections = {}

    def save_connections(self):
        """Save connections to config file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.connections, f, indent=2)
            return True
        except Exception as e:
            print(f"Save error: {e}")
            return False

    def add_connection(self, name: str, conn_type: str, params: Dict[str, Any]):
        """
        Add new connection configuration

        Args:
            name: Connection name (user-friendly)
            conn_type: 'database', 'google_sheets', or 'smartsheet'
            params: Connection parameters
        """
        self.connections[name] = {
            'type': conn_type,
            'params': params,
            'created': datetime.now().isoformat()
        }
        self.save_connections()

    def remove_connection(self, name: str):
        """Remove connection configuration"""
        if name in self.connections:
            del self.connections[name]
            self.save_connections()

    def get_connection(self, name: str):
        """Get connection object by name"""
        if name not in self.connections:
            return None

        conn_config = self.connections[name]
        conn_type = conn_config['type']
        params = conn_config['params']

        try:
            if conn_type == 'database':
                connector = DatabaseConnector(params['db_type'], params)
                if connector.connect():
                    return connector

            elif conn_type == 'google_sheets':
                connector = GoogleSheetsConnector(params.get('credentials_file'))
                if connector.connect():
                    return connector

            elif conn_type == 'smartsheet':
                connector = SmartsheetConnector(params.get('access_token'))
                if connector.connect():
                    return connector
        except Exception as e:
            print(f"Connection error: {e}")

        return None

    def list_connections(self) -> List[str]:
        """List all saved connection names"""
        return list(self.connections.keys())

    def test_connection(self, name: str) -> bool:
        """Test if connection works"""
        conn = self.get_connection(name)
        if conn:
            if hasattr(conn, 'close'):
                conn.close()
            return True
        return False
